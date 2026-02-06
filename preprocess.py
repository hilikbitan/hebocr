"""Image preprocessing for subtitle OCR."""
from __future__ import annotations

import cv2
import numpy as np


def preprocess_subtitle_crop(crop: np.ndarray, upscale: float = 2.0) -> np.ndarray:
    """Convert subtitle crop to a clean binary image for OCR."""
    if crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )
    return binary
