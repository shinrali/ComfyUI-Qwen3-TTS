"""Platform-aware paths for the custom-node-scoped Python runtime."""

from __future__ import annotations

import os
from pathlib import Path


def runtime_python(venv: Path, *, os_name: str | None = None) -> Path:
    """Return the Python executable created by venv on this platform."""

    platform_name = os.name if os_name is None else os_name
    if platform_name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"
