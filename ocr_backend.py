"""OCR backend abstraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class OCRResult:
    text: str
    confidence: float


class OCRBackend:
    def readtext(self, image: np.ndarray) -> List[OCRResult]:
        raise NotImplementedError


class EasyOCROCRBackend(OCRBackend):
    def __init__(self, gpu: bool = True) -> None:
        import easyocr

        self.reader = easyocr.Reader(["he", "en"], gpu=gpu)

    def readtext(self, image: np.ndarray) -> List[OCRResult]:
        results = self.reader.readtext(image)
        outputs: List[OCRResult] = []
        for _, text, conf in results:
            text = text.strip()
            if text:
                outputs.append(OCRResult(text=text, confidence=float(conf)))
        return outputs
