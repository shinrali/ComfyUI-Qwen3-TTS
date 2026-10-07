import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import wave


class SaveAudioTests(unittest.TestCase):
    def test_chinese_wav_output(self):
        root = Path(__file__).resolve().parent
        package = types.ModuleType("tts_save_test")
        package.__path__ = [str(root)]
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            output, run = temp / "output", temp / "run"
            run.mkdir()
            source = run / "candidate.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b"\0\0" * 240)
            paths = types.ModuleType("folder_paths")
            paths.get_output_directory = lambda: str(output)
            with patch.dict(sys.modules, {"tts_save_test": package, "folder_paths": paths}):
                spec = importlib.util.spec_from_file_location("tts_save_test.nodes", root / "nodes.py")
                nodes = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(nodes)
                prefix = "pixion_voice/\u516b\u5b57\u89c6\u9891_\u5973\u58f0_T04"
                result = nodes.SaveQwen3TTSAudio().save(
                    {"files": [source], "root": run}, prefix, False,
                )
                target = output / (prefix + ".wav")
                self.assertTrue(target.is_file())
                with wave.open(str(target), "rb") as wav:
                    self.assertEqual(wav.getnframes(), 240)
                self.assertEqual(result["ui"]["audio"][0]["filename"], target.name)
                self.assertFalse(run.exists())


if __name__ == "__main__":
    unittest.main()
