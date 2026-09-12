"""Install Qwen3-TTS into a custom-node-scoped dependency layer.

The layer reuses ComfyUI's existing Torch/CUDA packages, while keeping the
Transformers version required by qwen-tts out of ComfyUI's main environment.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"

# Windows portable/embeddable Python can run with an isolated ``python*._pth``
# configuration that does not add the script directory to ``sys.path``.  Make
# local custom-node modules importable before importing them.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_paths import runtime_python


def shared_site_packages() -> Path:
    """Return the site-packages directory of the Python running this installer."""
    purelib = sysconfig.get_path("purelib")
    if not purelib:
        raise RuntimeError("Could not resolve the ComfyUI site-packages directory")
    return Path(purelib).resolve()


def create_runtime() -> Path:
    """Create a venv, with a uv fallback for Windows portable Python builds."""

    python = runtime_python(VENV)
    if python.is_file():
        return python

    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    except subprocess.CalledProcessError as venv_error:
        uv = shutil.which("uv")
        if not uv and os.name == "nt":
            candidate = Path(sys.executable).resolve().parent / "Scripts" / "uv.exe"
            if candidate.is_file():
                uv = str(candidate)
        if not uv:
            raise RuntimeError(
                "Could not create the isolated Qwen3-TTS runtime. Install uv or "
                "use a Python build with the venv module, then run install.py again."
            ) from venv_error
        subprocess.check_call([uv, "venv", "--python", sys.executable, "--seed", str(VENV)])

    if not python.is_file():
        raise RuntimeError(f"The isolated Python runtime was not created at {python}")
    return python


def runtime_site_packages(python: Path) -> Path:
    """Ask the created runtime for its platform-specific site-packages path."""

    value = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip()
    if not value:
        raise RuntimeError("Could not resolve the isolated runtime site-packages directory")
    return Path(value).resolve()


def main() -> None:
    python_path = create_runtime()
    layer_site = runtime_site_packages(python_path)
    layer_site.mkdir(parents=True, exist_ok=True)
    shared = shared_site_packages()
    if not (shared / "torch").is_dir():
        raise RuntimeError(f"Torch was not found in the ComfyUI environment: {shared}")
    (layer_site / "comfyui-runtime.pth").write_text(f"{shared}\n", encoding="utf-8")

    python = str(python_path)
    pip = [python, "-m", "pip", "install", "--no-cache-dir"]
    subprocess.check_call([*pip, "--upgrade", "pip", "setuptools", "wheel"])
    # qwen-tts currently pins Transformers 4.57.3. Install the package itself
    # without pulling another Torch/CUDA stack; the runtime dependencies below
    # are installed locally and shadow only where needed.
    subprocess.check_call([*pip, "--no-deps", "qwen-tts==0.1.1"])
    subprocess.check_call([*pip, "-r", str(ROOT / "requirements-runtime.txt")])

    if os.environ.get("QWEN3_TTS_SKIP_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes"}:
        print("Skipping Qwen3-TTS model download by request.", flush=True)
        return

    default_model_root = ROOT.parent.parent / "models" / "qwen-voice"
    model_root = Path(
        os.environ.get("QWEN3_TTS_MODEL_ROOT", str(default_model_root))
    ).expanduser().resolve()
    subprocess.check_call([
        python,
        str(ROOT / "download_models.py"),
        "--model-root",
        str(model_root),
    ])


if __name__ == "__main__":
    main()
