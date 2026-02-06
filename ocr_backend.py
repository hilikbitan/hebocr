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
    def __init__(self, gpu: bool = True, languages: List[str] | None = None) -> None:
        from utils import configure_utf8_stdio
        import easyocr

        configure_utf8_stdio()
        lang_list = languages or ["he", "en"]
        try:
            self.reader = easyocr.Reader(lang_list, gpu=gpu)
        except Exception as exc:  # pragma: no cover - passes through library errors
            msg = str(exc)
            if "is not supported" in msg:
                raise RuntimeError(
                    "EasyOCR does not support the requested languages. "
                    "Upgrade easyocr (>=1.7.1) or adjust ocr_languages in config.yaml."
                ) from exc
            raise

    def readtext(self, image: np.ndarray) -> List[OCRResult]:
        results = self.reader.readtext(image)
        outputs: List[OCRResult] = []
        for _, text, conf in results:
            text = text.strip()
            if text:
                outputs.append(OCRResult(text=text, confidence=float(conf)))
        return outputs
