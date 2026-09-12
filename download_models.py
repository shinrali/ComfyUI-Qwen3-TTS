"""Download the official Qwen3-TTS models without requiring symlinks."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from model_paths import MODEL_IDS, resolve_model_dir


def download_models(model_root: Path) -> None:
    """Download missing models as real files into ComfyUI's model directory."""

    model_root.mkdir(parents=True, exist_ok=True)
    for model_id in MODEL_IDS.values():
        existing = resolve_model_dir(model_root, model_id)
        if existing is not None:
            print(f"Qwen3-TTS model already available: {existing}", flush=True)
            continue

        target = model_root / model_id.rsplit("/", 1)[-1]
        print(f"Downloading {model_id} to {target}", flush=True)
        snapshot_download(repo_id=model_id, local_dir=str(target))
        resolved = resolve_model_dir(model_root, model_id)
        if resolved is None:
            raise RuntimeError(f"Downloaded Qwen3-TTS model is incomplete: {target}")
        print(f"Qwen3-TTS model ready: {resolved}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True, type=Path)
    args = parser.parse_args()
    download_models(args.model_root.resolve())
