"""ComfyUI workflow nodes for Qwen3-TTS Voice Design and Voice Clone."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import uuid
import wave
from pathlib import Path
from typing import Any

import folder_paths


LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
]
SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
ROOT = Path(__file__).resolve().parent


def _run(request: dict[str, Any]) -> dict[str, Any]:
    python = ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"Qwen3-TTS runtime is not installed. Run: python {ROOT / 'install.py'}")
    model_cache = Path(folder_paths.models_dir).resolve() / "qwen-voice" / "huggingface"
    if not model_cache.is_dir():
        raise RuntimeError(f"Qwen3-TTS model cache was not found at {model_cache}")
    run_root = Path(folder_paths.get_temp_directory()).resolve() / "qwen3-tts" / uuid.uuid4().hex
    run_root.mkdir(parents=True, exist_ok=False)
    request_path, response_path = run_root / "request.json", run_root / "response.json"
    log_path, output_dir = run_root / "worker.log", run_root / "outputs"
    request_path.write_text(
        json.dumps({**request, "model_cache": str(model_cache)}, ensure_ascii=False),
        encoding="utf-8",
    )
    command = [
        str(python), str(ROOT / "worker.py"), "--request", str(request_path),
        "--response", str(response_path), "--output-dir", str(output_dir),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 1200
        try:
            while process.poll() is None:
                try:
                    import comfy.model_management as model_management
                    model_management.throw_exception_if_processing_interrupted()
                except ImportError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Qwen3-TTS generation exceeded 1200 seconds")
                time.sleep(0.25)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
    if process.returncode:
        output = log_path.read_text(encoding="utf-8", errors="replace")
        shutil.rmtree(run_root, ignore_errors=True)
        raise RuntimeError(f"Qwen3-TTS failed: {output[-4000:]}")
    result = json.loads(response_path.read_text(encoding="utf-8"))
    files = [Path(value).resolve() for value in result.get("files", [])]
    if not files or any(not path.is_file() or run_root not in path.parents for path in files):
        shutil.rmtree(run_root, ignore_errors=True)
        raise RuntimeError("Qwen3-TTS returned invalid output")
    return {"files": files, "sample_rate": int(result["sample_rate"]), "root": run_root}


class Qwen3TTSVoiceDesign:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True}),
            "instruct": ("STRING", {"multiline": True}),
            "language": (LANGUAGES, {"default": "Auto"}),
            "candidate_count": ("INT", {"default": 1, "min": 1, "max": 5}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
        }}
    RETURN_TYPES = ("QWEN3_TTS_AUDIO_BATCH",)
    RETURN_NAMES = ("audio_batch",)
    FUNCTION = "generate"
    CATEGORY = "audio/Qwen3-TTS"

    def generate(self, text: str, instruct: str, language: str, candidate_count: int, seed: int):
        return (_run({
            "operation": "design", "text": text, "instruct": instruct,
            "language": language, "candidate_count": candidate_count, "seed": seed,
        }),)


class Qwen3TTSVoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True}),
            "language": (LANGUAGES, {"default": "Auto"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            "reference_audio": ("AUDIO",),
        }}
    RETURN_TYPES = ("QWEN3_TTS_AUDIO_BATCH",)
    RETURN_NAMES = ("audio_batch",)
    FUNCTION = "generate"
    CATEGORY = "audio/Qwen3-TTS"

    def generate(self, text: str, language: str, seed: int, reference_audio: dict[str, Any]):
        waveform = reference_audio.get("waveform")
        rate = int(reference_audio.get("sample_rate") or 0)
        if waveform is None or rate <= 0:
            raise ValueError("Reference AUDIO is invalid")
        reference_root = Path(folder_paths.get_temp_directory()).resolve() / "qwen3-tts-reference" / uuid.uuid4().hex
        reference_root.mkdir(parents=True, exist_ok=False)
        try:
            reference = reference_root / "reference.wav"
            samples = waveform.detach().cpu().float()
            if samples.ndim == 3:
                samples = samples[0]
            if samples.ndim != 2:
                raise ValueError(f"Reference AUDIO has an unexpected shape: {tuple(samples.shape)}")
            pcm = (
                samples.clamp(-1.0, 1.0)
                .mul(32767.0)
                .round()
                .short()
                .transpose(0, 1)
                .contiguous()
                .numpy()
            )
            with wave.open(str(reference), "wb") as wav:
                wav.setnchannels(int(samples.shape[0]))
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(pcm.tobytes())
            return (_run({
                "operation": "clone", "text": text, "language": language,
                "seed": seed, "reference_path": str(reference),
            }),)
        finally:
            shutil.rmtree(reference_root, ignore_errors=True)


class SaveQwen3TTSAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio_batch": ("QWEN3_TTS_AUDIO_BATCH",),
            "filename_prefix": ("STRING", {"default": "qwen3_tts/audio"}),
            "number_candidates": ("BOOLEAN", {"default": False}),
        }}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "audio/Qwen3-TTS"

    def save(self, audio_batch: dict[str, Any], filename_prefix: str, number_candidates: bool):
        parts = [part for part in filename_prefix.replace("\\", "/").split("/") if part]
        if not parts or any(not SAFE_NAME.fullmatch(part) for part in parts):
            raise ValueError("Invalid filename_prefix")
        files = list(audio_batch.get("files") or [])
        if len(files) > 1 and not number_candidates:
            raise ValueError("Multiple candidates require number_candidates=true")
        output_root = Path(folder_paths.get_output_directory()).resolve()
        subfolder = Path(*parts[:-1]) if len(parts) > 1 else Path()
        destination = (output_root / subfolder).resolve()
        if destination != output_root and output_root not in destination.parents:
            raise ValueError("Output path is outside the ComfyUI output directory")
        destination.mkdir(parents=True, exist_ok=True)
        outputs = []
        try:
            for index, source in enumerate(files, 1):
                suffix = f"_{index:02d}" if number_candidates else ""
                filename = f"{parts[-1]}{suffix}.wav"
                target = destination / filename
                temporary = destination / f".{filename}.{uuid.uuid4().hex}.tmp"
                shutil.copyfile(source, temporary)
                temporary.replace(target)
                outputs.append({"filename": filename, "subfolder": subfolder.as_posix(), "type": "output"})
        finally:
            shutil.rmtree(Path(audio_batch["root"]), ignore_errors=True)
        return {"ui": {"audio": outputs}, "result": ()}


NODE_CLASS_MAPPINGS = {
    "Qwen3TTSVoiceDesign": Qwen3TTSVoiceDesign,
    "Qwen3TTSVoiceClone": Qwen3TTSVoiceClone,
    "SaveQwen3TTSAudio": SaveQwen3TTSAudio,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTSVoiceDesign": "Qwen3-TTS Voice Design",
    "Qwen3TTSVoiceClone": "Qwen3-TTS Voice Clone / TTS",
    "SaveQwen3TTSAudio": "Save Qwen3-TTS Audio",
}
