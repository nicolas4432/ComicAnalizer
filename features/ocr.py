from __future__ import annotations

from pathlib import Path

from core.data import TextFeatures


class NoOpOCRExtractor:
    """Placeholder OCR module; keeps the pipeline contract stable."""

    def extract(self, path: Path) -> TextFeatures:
        return TextFeatures(backend="none")


class PaddleOCRExtractor:
    """Optional OCR extractor using PaddleOCR when enabled and installed."""

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._ocr = None

    def _load(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError(
                    "PaddleOCR is not installed. Disable OCR or install paddleocr."
                ) from exc
            self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)
        return self._ocr

    def extract(self, path: Path) -> TextFeatures:
        ocr = self._load()
        result = ocr.ocr(str(path), cls=True)
        blocks: list[dict[str, object]] = []
        texts: list[str] = []
        confidences: list[float] = []

        for page_result in result or []:
            for item in page_result or []:
                if len(item) < 2:
                    continue
                box, text_payload = item[0], item[1]
                text, confidence = text_payload[0], float(text_payload[1])
                texts.append(text)
                confidences.append(confidence)
                blocks.append({"box": box, "text": text, "confidence": confidence})

        confidence = sum(confidences) / len(confidences) if confidences else None
        return TextFeatures(
            text="\n".join(texts),
            confidence=confidence,
            blocks=blocks,
            backend="paddleocr",
        )

