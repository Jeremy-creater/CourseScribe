"""Timed transcript pieces, their on-disk checkpoint, and the raw text form."""

from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


def append_segment(path: Path, segment: Segment) -> None:
    """Add one finished segment to the checkpoint and force it onto disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(segment), ensure_ascii=False) + "\n"
    with path.open("ab") as stream:
        stream.write(line.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def read_segments(path: Path) -> list[Segment]:
    """Read the checkpoint. A half-written last line is removed from the file."""
    if not path.exists():
        return []
    segments: list[Segment] = []
    good_bytes = 0
    with path.open("rb") as stream:
        for line in stream:
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("unterminated line")
                segments.append(Segment(**json.loads(line.decode("utf-8"))))
            except (ValueError, TypeError):
                break
            good_bytes += len(line)
    if good_bytes != path.stat().st_size:
        with path.open("r+b") as stream:
            stream.truncate(good_bytes)
    return segments


def _has_content(text: str) -> bool:
    return any(not unicodedata.category(ch).startswith("P") for ch in text)


def segments_to_text(segments: list[Segment]) -> str:
    """One line per spoken segment; punctuation alone joins the line before it."""
    lines: list[str] = []
    for segment in segments:
        text = " ".join(segment.text.split())
        if not text:
            continue
        if _has_content(text):
            lines.append(text)
        elif lines:
            lines[-1] += text
    return "".join(f"{line}\n" for line in lines)
