"""Install Qwen3-TTS into a custom-node-scoped dependency layer.

The layer reuses ComfyUI's existing Torch/CUDA packages, while keeping the
Transformers version required by qwen-tts out of ComfyUI's main environment.
"""

import os
import re
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


def host_torch_version(shared: Path) -> str | None:
    """Read the torch version string from the shared environment without import."""

    version_file = shared / "torch" / "version.py"
    if not version_file.is_file():
        return None
    match = re.search(
        r"__version__\s*=\s*['\"]([^'\"]+)['\"]",
        version_file.read_text(encoding="utf-8", errors="replace"),
    )
    return match.group(1) if match else None


def check_gpu_runtime(python: str) -> None:
    """Fail early when the reused torch is not a CUDA build (wrong interpreter).

    install.py reuses the torch of whatever interpreter runs it, so launching it
    with a separate system Python instead of ComfyUI's own silently builds a CPU
    runtime that cannot run Qwen3-TTS. Detect that and point at the right Python.
    """

    try:
        output = subprocess.check_output(
            [python, "-c", "import torch; print(torch.__version__); print(torch.version.cuda)"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError):
        return
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    cuda = lines[-1] if lines else ""
    if cuda in {"", "None"}:
        version = lines[0] if lines else "unknown"
        raise RuntimeError(
            f"The reused PyTorch build has no CUDA (torch {version}). Qwen3-TTS needs "
            "a CUDA/BF16 build, so it must reuse the SAME Python that ComfyUI uses to "
            "run workflows. Re-run with that interpreter (ComfyUI Desktop: "
            "<ComfyUI base>\\.venv\\Scripts\\python.exe install.py) instead of a "
            "standalone system Python."
        )


def ensure_torchaudio(shared: Path, python: str) -> None:
    """Reuse the host torchaudio, or install a torch-matched build into the layer.

    ComfyUI Desktop builds (e.g. torch 2.10.0+cu130) no longer ship torchaudio,
    but qwen-tts imports torchaudio.compliance.kaldi at load time. Install the
    wheel with the host torch's exact version and CUDA suffix so its C extension
    links against the reused libtorch, never pulling a second GPU stack.
    """

    if (shared / "torchaudio").is_dir():
        return
    torch_version = host_torch_version(shared)
    if not torch_version:
        raise RuntimeError(
            "torchaudio is required by Qwen3-TTS but was neither found in the "
            "ComfyUI environment nor resolvable from its torch version. Install "
            "a ComfyUI build that bundles torchaudio."
        )
    base, _, suffix = torch_version.partition("+")
    if suffix.startswith("cu"):
        index, package = suffix, f"torchaudio=={base}+{suffix}"
    else:
        index, package = "cpu", f"torchaudio=={base}"
    print(f"Installing matched torchaudio {package} for torch {torch_version}", flush=True)
    subprocess.check_call([
        python, "-m", "pip", "install", "--no-cache-dir", "--no-deps",
        "--index-url", f"https://download.pytorch.org/whl/{index}", package,
    ])


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
    check_gpu_runtime(python)
    ensure_torchaudio(shared, python)

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
