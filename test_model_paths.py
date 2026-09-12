import tempfile
import unittest
from pathlib import Path

from model_paths import model_is_complete, resolve_model_dir


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def complete(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"weights")


class ModelPathTests(unittest.TestCase):
    def test_direct_model_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = root / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            complete(direct)
            self.assertTrue(model_is_complete(direct))
            self.assertEqual(resolve_model_dir(root, MODEL_ID), direct)

    def test_legacy_huggingface_cache_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "huggingface" / "hub" / "models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            revision = "abc123"
            (repo / "refs").mkdir(parents=True)
            (repo / "refs" / "main").write_text(revision, encoding="utf-8")
            snapshot = repo / "snapshots" / revision
            complete(snapshot)
            self.assertEqual(resolve_model_dir(root, MODEL_ID), snapshot)

    def test_incomplete_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = root / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            incomplete.mkdir()
            (incomplete / "config.json").write_text("{}", encoding="utf-8")
            self.assertFalse(model_is_complete(incomplete))
            self.assertIsNone(resolve_model_dir(root, MODEL_ID))


if __name__ == "__main__":
    unittest.main()
