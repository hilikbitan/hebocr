"""End-to-end subtitle extraction pipeline."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from dedup import DedupConfig, DedupState, SubtitleCandidate, finalize_candidate, should_merge
from ocr_backend import EasyOCROCRBackend
from preprocess import preprocess_subtitle_crop
from roi import auto_detect_roi, manual_select_roi
from srt_writer import write_srt
from utils import (CancelledError, LogCallback, ProgressCallback, SubtitleSegment,
                   configure_utf8_stdio, ensure_dir, log_csv, raise_if_cancelled,
                   timestamp_ms)


@dataclasses.dataclass
class ExtractConfig:
    sample_fps: float = 4.0
    roi_scan_seconds: int = 60
    search_vertical_band: tuple[float, float] = (0.55, 0.98)
    ocr_languages: tuple[str, ...] = ("he", "en")
    phash_hamming_threshold: int = 10
    min_ms_between_ocr: int = 160
    text_similarity_threshold: float = 0.82
    min_line_duration_ms: int = 220
    max_merge_gap_ms: int = 140
    min_confidence: float = 0.35
    debug: bool = False
    debug_dir: str = ""


def _ffmpeg_available() -> bool:
    return subprocess.call(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def _sample_frames_ffmpeg(video_path: str, fps: float, duration_s: Optional[int] = None) -> Iterable[np.ndarray]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
    ]
    if duration_s:
        cmd.extend(["-t", str(duration_s)])
    cmd.extend(["-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"])
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        return
    # Read frame size from ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "json", video_path],
        capture_output=True,
        text=True,
        check=False,
    )
    width = height = 0
    if probe.stdout:
        data = json.loads(probe.stdout)
        stream = data.get("streams", [{}])[0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
    if width == 0 or height == 0:
        process.kill()
        return
    frame_size = width * height * 3
    while True:
        raw = process.stdout.read(frame_size)
        if len(raw) != frame_size:
            break
        frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
        yield frame


def _sample_frames_opencv(video_path: str, fps: float, duration_s: Optional[int] = None) -> Iterable[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(int(round(source_fps / fps)), 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = total_frames
    if duration_s is not None:
        max_frames = min(max_frames, int(duration_s * source_fps))
    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            yield frame
        frame_idx += 1
    cap.release()


def sample_frames(video_path: str, fps: float, duration_s: Optional[int] = None) -> Iterable[np.ndarray]:
    if _ffmpeg_available():
        return _sample_frames_ffmpeg(video_path, fps, duration_s)
    return _sample_frames_opencv(video_path, fps, duration_s)


def phash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_low_freq = dct[:8, :8]
    median = np.median(dct_low_freq)
    bits = dct_low_freq > median
    return "".join("1" if b else "0" for b in bits.flatten())


def hamming_distance(hash_a: str, hash_b: str) -> int:
    if not hash_a or not hash_b:
        return 64
    return sum(ch1 != ch2 for ch1, ch2 in zip(hash_a, hash_b))


def _prepare_debug_dir(config: ExtractConfig) -> str:
    if not config.debug:
        return ""
    if config.debug_dir:
        return ensure_dir(config.debug_dir)
    session = timestamp_ms()
    return ensure_dir(os.path.join("logs", f"session_{session}"))


def _log(log_cb: Optional[LogCallback], message: str) -> None:
    if log_cb:
        log_cb(message)


def extract_subtitles(
    video_path: str,
    output_srt: str,
    config: ExtractConfig,
    progress_cb: Optional[ProgressCallback] = None,
    log_cb: Optional[LogCallback] = None,
    cancel_event: Optional[object] = None,
    manual_roi: bool = False,
) -> list[SubtitleSegment]:
    """Extract subtitles from video and write to SRT."""
    debug_dir = _prepare_debug_dir(config)
    _log(log_cb, "Loading OCR backend...")
    ocr = EasyOCROCRBackend(gpu=True, languages=list(config.ocr_languages))

    _log(log_cb, "Sampling frames for ROI detection...")
    frames_for_roi = list(sample_frames(video_path, config.sample_fps, config.roi_scan_seconds))
    if not frames_for_roi:
        raise RuntimeError("Unable to sample frames for ROI detection.")

    if manual_roi:
        roi = manual_select_roi(frames_for_roi[0])
    else:
        roi = auto_detect_roi(frames_for_roi, config.search_vertical_band)

    x, y, w, h = roi
    _log(log_cb, f"Using ROI: x={x}, y={y}, w={w}, h={h}")

    dedup_cfg = DedupConfig(
        text_similarity_threshold=config.text_similarity_threshold,
        max_merge_gap_ms=config.max_merge_gap_ms,
        min_line_duration_ms=config.min_line_duration_ms,
    )
    dedup_state = DedupState()

    segments: list[SubtitleSegment] = []
    decision_rows: list[dict[str, object]] = []
    last_hash = ""
    last_ocr_ts = 0
    frame_idx = 0

    for frame in sample_frames(video_path, config.sample_fps):
        raise_if_cancelled(cancel_event)
        time_ms = int((frame_idx / config.sample_fps) * 1000)
        frame_idx += 1

        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        current_hash = phash(crop)
        hash_delta = hamming_distance(last_hash, current_hash) if last_hash else 64
        should_ocr = hash_delta >= config.phash_hamming_threshold
        if time_ms - last_ocr_ts < config.min_ms_between_ocr:
            should_ocr = False

        if not should_ocr:
            decision_rows.append(
                {
                    "time_ms": time_ms,
                    "hash": current_hash,
                    "hash_delta": hash_delta,
                    "ocr": False,
                    "text": "",
                    "confidence": 0.0,
                    "merged": False,
                }
            )
            continue

        preprocessed = preprocess_subtitle_crop(crop)
        results = ocr.readtext(preprocessed)
        if results:
            best = max(results, key=lambda r: r.confidence)
        else:
            best = None

        ocr_text = best.text if best else ""
        confidence = best.confidence if best else 0.0
        last_hash = current_hash
        last_ocr_ts = time_ms

        if config.debug and debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"raw_{time_ms}.png"), crop)
            cv2.imwrite(os.path.join(debug_dir, f"bin_{time_ms}.png"), preprocessed)

        merged = False
        if confidence >= config.min_confidence and ocr_text:
            candidate = SubtitleCandidate(
                start_ms=time_ms,
                end_ms=time_ms,
                text=ocr_text,
                confidence=confidence,
                image_hash=current_hash,
            )
            if dedup_state.current is None:
                dedup_state.current = candidate
            else:
                if should_merge(dedup_state.current, candidate, dedup_cfg):
                    dedup_state.current.end_ms = candidate.start_ms
                    dedup_state.current.text = candidate.text
                    dedup_state.current.confidence = max(dedup_state.current.confidence, candidate.confidence)
                    merged = True
                else:
                    finalized = finalize_candidate(dedup_state.current, dedup_cfg)
                    segments.append(
                        SubtitleSegment(
                            index=len(segments) + 1,
                            start_ms=finalized.start_ms,
                            end_ms=finalized.end_ms,
                            text=finalized.text,
                            confidence=finalized.confidence,
                        )
                    )
                    dedup_state.current = candidate

        decision_rows.append(
            {
                "time_ms": time_ms,
                "hash": current_hash,
                "hash_delta": hash_delta,
                "ocr": True,
                "text": ocr_text,
                "confidence": confidence,
                "merged": merged,
            }
        )

        if progress_cb:
            progress_cb(min(frame_idx / max(len(frames_for_roi), 1), 1.0), f"Processed {frame_idx} frames")

    if dedup_state.current is not None:
        finalized = finalize_candidate(dedup_state.current, dedup_cfg)
        segments.append(
            SubtitleSegment(
                index=len(segments) + 1,
                start_ms=finalized.start_ms,
                end_ms=finalized.end_ms,
                text=finalized.text,
                confidence=finalized.confidence,
            )
        )

    if not segments:
        _log(log_cb, "No subtitles detected.")

    write_srt(output_srt, segments)
    _log(log_cb, f"Wrote SRT to {output_srt}")

    if config.debug and debug_dir:
        log_csv(os.path.join(debug_dir, "decisions.csv"), decision_rows)

    return segments


def load_config(path: str) -> ExtractConfig:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return ExtractConfig(
        sample_fps=float(data.get("sample_fps", 4.0)),
        roi_scan_seconds=int(data.get("roi_scan_seconds", 60)),
        search_vertical_band=tuple(data.get("search_vertical_band", [0.55, 0.98])),
        ocr_languages=tuple(data.get("ocr_languages", ["he", "en"])),
        phash_hamming_threshold=int(data.get("phash_hamming_threshold", 10)),
        min_ms_between_ocr=int(data.get("min_ms_between_ocr", 160)),
        text_similarity_threshold=float(data.get("text_similarity_threshold", 0.82)),
        min_line_duration_ms=int(data.get("min_line_duration_ms", 220)),
        max_merge_gap_ms=int(data.get("max_merge_gap_ms", 140)),
        min_confidence=float(data.get("min_confidence", 0.35)),
        debug=bool(data.get("debug", False)),
        debug_dir=str(data.get("debug_dir", "")),
    )


def save_config(path: str, config: ExtractConfig) -> None:
    import yaml

    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dataclasses.asdict(config), handle, sort_keys=False)


def run_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Hebrew Subtitle OCR extractor")
    parser.add_argument("video", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--manual-roi", action="store_true")
    args = parser.parse_args()

    configure_utf8_stdio()
    config = load_config(args.config)
    extract_subtitles(args.video, args.output, config, manual_roi=args.manual_roi)


if __name__ == "__main__":
    run_cli()
