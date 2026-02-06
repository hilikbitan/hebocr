"""Shared helpers for logging, timing, and file utilities."""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import os
import pathlib
import threading
from typing import Any, Callable, Iterable, Optional


@dataclasses.dataclass
class SubtitleSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str
    confidence: float


class CancelledError(RuntimeError):
    """Raised when a long-running operation is cancelled."""


def ensure_dir(path: str | os.PathLike[str]) -> str:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def timestamp_ms() -> int:
    return int(dt.datetime.utcnow().timestamp() * 1000)


def format_ts_ms(ts_ms: int) -> str:
    seconds, millis = divmod(max(ts_ms, 0), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def human_ms(ts_ms: int) -> str:
    return format_ts_ms(ts_ms)


def atomic_write(path: str, data: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(data)
    os.replace(tmp_path, path)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def log_csv(path: str, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def raise_if_cancelled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Operation cancelled")


ProgressCallback = Callable[[float, str], None]
LogCallback = Callable[[str], None]
