"""Resolve Qwen3-TTS models from direct or legacy Hugging Face layouts."""

from __future__ import annotations

from pathlib import Path


MODEL_IDS = {
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}


def model_is_complete(model_dir: Path) -> bool:
    """Return whether the minimum files required by Qwen3-TTS are present."""

    return (
        model_dir.is_dir()
        and (model_dir / "config.json").is_file()
        and (model_dir / "model.safetensors").is_file()
    )


def resolve_model_dir(model_root: Path, model_id: str) -> Path | None:
    """Find a model in the portable direct layout or the legacy HF cache."""

    repo_name = model_id.rsplit("/", 1)[-1]
    direct = model_root / repo_name
    if model_is_complete(direct):
        return direct

    cache_repo = model_root / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
    ref = cache_repo / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        snapshot = cache_repo / "snapshots" / revision
        if revision and model_is_complete(snapshot):
            return snapshot
    return None
