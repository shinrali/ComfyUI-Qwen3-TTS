import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
from threading import Barrier
import types
import unittest
from unittest.mock import patch
import wave


class SaveAudioTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent
        package = types.ModuleType("tts_save_test")
        package.__path__ = [str(root)]
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name).resolve()
        self.output = self.temp / "output"
        paths = types.ModuleType("folder_paths")
        paths.get_output_directory = lambda: str(self.output)
        modules = patch.dict(sys.modules, {"tts_save_test": package, "folder_paths": paths})
        modules.start()
        self.addCleanup(modules.stop)
        spec = importlib.util.spec_from_file_location("tts_save_test.nodes", root / "nodes.py")
        nodes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nodes)
        self.node = nodes.SaveQwen3TTSAudio()

    def make_batch(self, *samples):
        run = Path(tempfile.mkdtemp(dir=self.temp))
        files = []
        for index, sample in enumerate(samples):
            source = run / f"candidate_{index}.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(bytes([sample, 0]) * 240)
            files.append(source)
        return {"files": files, "root": run}

    def save(self, *samples, prefix="voices/audio", number_candidates=False):
        batch = self.make_batch(*samples)
        result = self.node.save(batch, prefix, number_candidates)
        self.assertFalse(batch["root"].exists())
        return result["ui"]["audio"]

    def assert_wav(self, preview, sample):
        self.assertEqual(preview["type"], "output")
        target = self.output / preview["subfolder"] / preview["filename"]
        with wave.open(str(target), "rb") as wav:
            self.assertEqual(wav.getnframes(), 240)
            self.assertEqual(wav.readframes(240), bytes([sample, 0]) * 240)

    def test_repeated_chinese_wav_outputs_have_distinct_preview_paths(self):
        prefix = "pixion_voice/八字视频_女声_T04"
        for number_candidates in (False, True):
            with self.subTest(number_candidates=number_candidates):
                first, = self.save(1, prefix=prefix, number_candidates=number_candidates)
                second, = self.save(2, prefix=prefix, number_candidates=number_candidates)
                self.assertNotEqual(first["filename"], second["filename"])
                self.assertEqual(first["subfolder"], "pixion_voice")
                self.assertEqual(second["subfolder"], "pixion_voice")
                self.assert_wav(first, 1)
                self.assert_wav(second, 2)

    def test_multiple_candidates_share_a_generation_number(self):
        for number in (1, 2):
            previews = self.save(3, 4, 5, prefix=r"voices\audio", number_candidates=True)
            self.assertEqual(
                [preview["filename"] for preview in previews],
                [f"audio_{number:04d}_{index:02d}.wav" for index in (1, 2, 3)],
            )
            for preview, sample in zip(previews, (3, 4, 5)):
                self.assertEqual(preview["subfolder"], "voices")
                self.assert_wav(preview, sample)

    def test_multiple_candidates_still_require_numbering(self):
        with self.assertRaisesRegex(ValueError, "require number_candidates=true"):
            self.save(3, 4)
        self.assertFalse(self.output.exists())

    def test_existing_legacy_and_numbered_wavs_are_preserved(self):
        directory = self.output / "voices"
        directory.mkdir(parents=True)
        existing = ("audio.wav", "audio_01.wav", "audio_0001.wav", "audio_0007_02.wav")
        for name in existing:
            (directory / name).write_bytes(name.encode())
        preview, = self.save(6)
        self.assertEqual(preview["filename"], "audio_0008.wav")
        self.assert_wav(preview, 6)
        for name in existing:
            self.assertEqual((directory / name).read_bytes(), name.encode())

    def test_late_collision_on_second_candidate_keeps_existing_file(self):
        directory = self.output / "voices"
        directory.mkdir(parents=True)
        existing = directory / "audio_0001_02.wav"

        def scan_then_create(path):
            self.assertEqual(path, directory)
            existing.write_bytes(b"already saved")
            return iter(())

        with patch.object(Path, "iterdir", scan_then_create):
            previews = self.save(7, 8, number_candidates=True)
        self.assertEqual(
            sorted(path.name for path in directory.iterdir()),
            ["audio_0001_02.wav", "audio_0002_01.wav", "audio_0002_02.wav"],
        )
        self.assertEqual(existing.read_bytes(), b"already saved")
        for preview, sample in zip(previews, (7, 8)):
            self.assert_wav(preview, sample)

    def test_concurrent_writes_from_the_same_directory_snapshot(self):
        workers = 8
        barrier = Barrier(workers, timeout=10)
        original_iterdir = Path.iterdir

        def concurrent_scan(path):
            snapshot = list(original_iterdir(path))
            barrier.wait()
            return iter(snapshot)

        def save_batch(index):
            return self.save(index, index + workers, number_candidates=True)

        with patch.object(Path, "iterdir", concurrent_scan):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(save_batch, range(workers)))
        filenames = set()
        for index, previews in enumerate(results):
            for preview, sample in zip(previews, (index, index + workers)):
                filenames.add(preview["filename"])
                self.assert_wav(preview, sample)
        self.assertEqual(len(filenames), workers * 2)
        self.assertEqual({path.name for path in (self.output / "voices").iterdir()}, filenames)

    def test_failed_copy_removes_only_new_outputs(self):
        prior, = self.save(9)
        batch = self.make_batch(10, 11)
        batch["files"][1].unlink()
        with self.assertRaises(FileNotFoundError):
            self.node.save(batch, "voices/audio", True)
        self.assertFalse(batch["root"].exists())
        self.assertEqual(
            [path.name for path in (self.output / "voices").iterdir()],
            [prior["filename"]],
        )
        self.assert_wav(prior, 9)

    def test_rejects_output_subfolder_symlink_outside_output_root(self):
        self.output.mkdir()
        outside = self.temp / "outside"
        outside.mkdir()
        try:
            (self.output / "voices").symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"Directory symlinks unavailable: {exc}")
        with self.assertRaisesRegex(ValueError, "outside the ComfyUI output directory"):
            self.save(12)
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
