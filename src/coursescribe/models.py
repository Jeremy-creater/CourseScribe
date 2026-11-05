"""Transcription models the settings offer, with download sizes in GB."""

from __future__ import annotations

DEFAULT_TRANSCRIBE_MODEL = "large-v3"
DEFAULT_PROOFREAD_MODEL = "gemma4:12b"
DEFAULT_NOTES_MODEL = "qwen3.5:9b"

MODELS = {
    "large-v3": 3.1,
    "large-v3-turbo": 1.6,
    "medium": 1.5,
    "small": 0.5,
}


def model_size_label(name: str) -> str:
    size = MODELS.get(name)
    return f"约 {size} GB" if size else "大小未知"
