"""Progress estimation shared by the ComfyUI Qwen3-TTS nodes."""

from __future__ import annotations

import math


STAGE_PROGRESS = {
    "queued": 5.0,
    "waiting_for_worker": 10.0,
    "loading_model": 18.0,
    "preparing_reference": 28.0,
    "generating": 35.0,
    "validating": 93.0,
    "saving": 95.0,
    "complete": 100.0,
}


def estimate_generation_seconds(
    operation: str,
    *,
    text_length: int,
    candidate_count: int = 1,
) -> float:
    """Match the former CUDA Voice service's conservative ETA model."""

    if operation == "design":
        configured = 12.0
        candidate_scale = 0.65 + (0.35 * max(1, candidate_count))
    else:
        configured = 20.0
        candidate_scale = 1.0
    text_scale = max(0.7, min(3.0, math.sqrt(max(1, text_length) / 90)))
    return max(5.0, configured * candidate_scale * text_scale)


def stage_percent(stage: str, *, elapsed: float = 0.0, estimate: float = 1.0) -> float:
    """Return phase progress; generation advances from 35% to at most 92%."""

    percent = STAGE_PROGRESS.get(stage, STAGE_PROGRESS["waiting_for_worker"])
    if stage == "generating":
        percent = min(92.0, percent + (max(0.0, elapsed) / max(1.0, estimate)) * 57.0)
    return round(max(0.0, min(100.0, percent)), 1)
