"""Qwen3-TTS inference process launched synchronously by a ComfyUI node."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from quality import speech_token_limit, validate_speech


def write_status(path: Path, stage: str) -> None:
    """Atomically publish the worker's current inference phase."""

    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"stage": stage, "updated_at": time.time()}),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_model(kind: str, cache_dir: Path) -> Any:
    model_id = (
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        if kind == "design"
        else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    )
    repo_dir = cache_dir / "hub" / f"models--{model_id.replace('/', '--')}"
    ref = repo_dir / "refs" / "main"
    if not ref.is_file():
        raise FileNotFoundError(f"Qwen3-TTS model ref was not found: {ref}")
    model_dir = repo_dir / "snapshots" / ref.read_text(encoding="utf-8").strip()
    required = (model_dir / "config.json", model_dir / "model.safetensors")
    if not model_dir.is_dir() or any(not path.is_file() for path in required):
        raise FileNotFoundError(f"Qwen3-TTS model snapshot is incomplete: {model_dir}")
    return Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation=(
            "flash_attention_2"
            if importlib.util.find_spec("flash_attn") is not None
            else "sdpa"
        ),
    )


def run(request_path: Path, response_path: Path, output_dir: Path, status_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    kind = str(request.get("operation") or "")
    text = str(request.get("text") or "").strip()
    language = str(request.get("language") or "Auto")
    if kind not in {"design", "clone"} or not text:
        raise ValueError("Invalid Qwen3-TTS request")
    seed = int(request.get("seed") or 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    write_status(status_path, "loading_model")
    model = load_model(kind, Path(request["model_cache"]))
    if kind == "design":
        instruct = str(request.get("instruct") or "").strip()
        count = max(1, min(5, int(request.get("candidate_count") or 1)))
        if not instruct:
            raise ValueError("instruct is required")
        write_status(status_path, "generating")
        if count == 1:
            wavs, sample_rate = model.generate_voice_design(
                text=text, language=language, instruct=instruct
            )
        else:
            wavs, sample_rate = model.generate_voice_design(
                text=[text] * count,
                language=[language] * count,
                instruct=[instruct] * count,
            )
    else:
        reference = Path(request["reference_path"])
        write_status(status_path, "preparing_reference")
        prompt = model.create_voice_clone_prompt(
            ref_audio=str(reference), x_vector_only_mode=True
        )
        write_status(status_path, "generating")
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
            max_new_tokens=speech_token_limit(text),
        )
    write_status(status_path, "validating")
    validated = validate_speech(list(wavs), int(sample_rate), text)
    write_status(status_path, "saving")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for index, wav in enumerate(validated, 1):
        path = output_dir / f"candidate-{index:02d}.wav"
        sf.write(path, np.asarray(wav), int(sample_rate), format="WAV")
        files.append(str(path))
    response_path.write_text(
        json.dumps({"files": files, "sample_rate": int(sample_rate)}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    args = parser.parse_args()
    run(args.request, args.response, args.output_dir, args.status)
