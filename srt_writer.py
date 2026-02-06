"""SRT formatting and writing."""
from __future__ import annotations

from typing import Iterable

from utils import SubtitleSegment, format_ts_ms, atomic_write


def format_srt(segments: Iterable[SubtitleSegment]) -> str:
    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{format_ts_ms(seg.start_ms)} --> {format_ts_ms(seg.end_ms)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_srt(path: str, segments: Iterable[SubtitleSegment]) -> None:
    content = format_srt(segments)
    atomic_write(path, content)
