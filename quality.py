"""Conservative output guards for generated speech."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


def speech_limit_seconds(text: str) -> float:
    cjk = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text))
    words = len(
        re.findall(
            r"[^\W_]+",
            re.sub(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", " ", text),
        )
    )
    limit = max(20.0, 12.0 + cjk * 0.6 + words * 1.2)
    if limit > 600:
        raise ValueError("TTS text is too long for one request; split it into segments.")
    return limit


def speech_token_limit(text: str) -> int:
    return math.ceil(speech_limit_seconds(text) * 12.5)


def validate_speech(wavs: list[Any], sample_rate: int, text: str) -> list[np.ndarray]:
    if not wavs or sample_rate <= 0:
        raise ValueError("TTS produced no audio.")
    limit = speech_limit_seconds(text)
    validated: list[np.ndarray] = []
    for wav in wavs:
        samples = np.asarray(wav)
        duration = len(samples) / sample_rate
        if not np.isfinite(samples).all() or duration < 0.1:
            raise ValueError("TTS produced empty or invalid audio.")
        if duration >= limit - 0.2:
            raise ValueError(
                "TTS abnormal duration: generation reached its text-based safety limit. "
                "Inspect inference and the reference audio; do not retry unchanged."
            )
        rms = float(np.sqrt(np.mean(samples.astype(float) ** 2)))
        if rms < 1e-5:
            raise ValueError("TTS produced silent or invalid audio; result was not accepted.")
        validated.append(samples)
    return validated
