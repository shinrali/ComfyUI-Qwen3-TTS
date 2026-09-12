"""Install Qwen3-TTS into a custom-node-scoped dependency layer.

The layer reuses ComfyUI's existing Torch/CUDA packages, while keeping the
Transformers version required by qwen-tts out of ComfyUI's main environment.
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def shared_site_packages() -> Path:
    """Return the site-packages directory of the Python running this installer."""
    return Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def main() -> None:
    if not (VENV / "bin" / "python").is_file():
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])

    layer_site = VENV / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    layer_site.mkdir(parents=True, exist_ok=True)
    shared = shared_site_packages()
    if not (shared / "torch").is_dir():
        raise RuntimeError(f"Torch was not found in the ComfyUI environment: {shared}")
    (layer_site / "comfyui-runtime.pth").write_text(f"{shared}\n", encoding="utf-8")

    python = str(VENV / "bin" / "python")
    pip = [python, "-m", "pip", "install", "--no-cache-dir"]
    subprocess.check_call([*pip, "--upgrade", "pip", "setuptools", "wheel"])
    # qwen-tts currently pins Transformers 4.57.3. Install the package itself
    # without pulling another Torch/CUDA stack; the runtime dependencies below
    # are installed locally and shadow only where needed.
    subprocess.check_call([*pip, "--no-deps", "qwen-tts==0.1.1"])
    subprocess.check_call([*pip, "-r", str(ROOT / "requirements-runtime.txt")])


if __name__ == "__main__":
    main()
