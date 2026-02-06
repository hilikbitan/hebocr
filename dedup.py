"""Deduplication and merging logic for subtitle segments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz


@dataclass
class DedupConfig:
    text_similarity_threshold: float
    max_merge_gap_ms: int
    min_line_duration_ms: int


@dataclass
class SubtitleCandidate:
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    image_hash: str


@dataclass
class DedupState:
    current: Optional[SubtitleCandidate] = None


def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def should_merge(current: SubtitleCandidate, new: SubtitleCandidate, cfg: DedupConfig) -> bool:
    gap = new.start_ms - current.end_ms
    if gap < 0:
        gap = 0
    if gap > cfg.max_merge_gap_ms:
        return False
    similarity = text_similarity(current.text, new.text)
    return similarity >= cfg.text_similarity_threshold


def finalize_candidate(candidate: SubtitleCandidate, cfg: DedupConfig) -> SubtitleCandidate:
    duration = candidate.end_ms - candidate.start_ms
    if duration < cfg.min_line_duration_ms:
        candidate.end_ms = candidate.start_ms + cfg.min_line_duration_ms
    return candidate
