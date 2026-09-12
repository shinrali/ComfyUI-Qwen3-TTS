import tempfile
import unittest
from pathlib import Path

from runtime_paths import runtime_python


class RuntimePathTests(unittest.TestCase):
    def test_posix_runtime_path(self):
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory) / ".venv"
            self.assertEqual(runtime_python(venv, os_name="posix"), venv / "bin" / "python")

    def test_windows_runtime_path(self):
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory) / ".venv"
            self.assertEqual(runtime_python(venv, os_name="nt"), venv / "Scripts" / "python.exe")


if __name__ == "__main__":
    unittest.main()
