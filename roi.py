"""ROI detection utilities."""
from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np


ROI = Tuple[int, int, int, int]


def auto_detect_roi(frames: Iterable[np.ndarray], search_vertical_band: tuple[float, float]) -> ROI:
    """Detect subtitle ROI by finding band with highest edge density."""
    frames = list(frames)
    if not frames:
        raise ValueError("No frames provided for ROI detection")

    height, width = frames[0].shape[:2]
    y_start = int(height * search_vertical_band[0])
    y_end = int(height * search_vertical_band[1])

    edge_accum = np.zeros((y_end - y_start,), dtype=np.float32)
    for frame in frames:
        band = frame[y_start:y_end, :]
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.mean(axis=1)
        edge_accum += edge_density

    edge_accum /= max(len(frames), 1)
    peak_idx = int(edge_accum.argmax())
    band_height = max(int(height * 0.18), 40)
    roi_top = max(y_start + peak_idx - band_height // 2, 0)
    roi_bottom = min(roi_top + band_height, height)

    return 0, roi_top, width, roi_bottom - roi_top


def manual_select_roi(frame: np.ndarray) -> ROI:
    """Allow manual ROI selection using OpenCV GUI."""
    if frame is None:
        raise ValueError("Frame is required for manual ROI selection")
    roi = cv2.selectROI("Select Subtitle ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Subtitle ROI")
    x, y, w, h = roi
    if w == 0 or h == 0:
        raise ValueError("No ROI selected")
    return int(x), int(y), int(w), int(h)
