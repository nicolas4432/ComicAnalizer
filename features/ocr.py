from __future__ import annotations

from pathlib import Path

from core.data import TextFeatures
from features.ocr_paddle import PaddleOCRComplementaryExtractor


class NoOpOCRExtractor:
    """Placeholder OCR module; keeps the pipeline contract stable."""

    def extract(self, path: Path) -> TextFeatures:
        return TextFeatures(backend="none")


class PaddleOCRExtractor:
    """Pipeline OCR adapter backed by the shared PaddleOCR implementation."""

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._extractor = PaddleOCRComplementaryExtractor(lang=lang)

    def extract(self, path: Path) -> TextFeatures:
        result = self._extractor.extract_page(path)
        blocks: list[dict[str, object]] = [block.to_dict() for block in result.blocks]
        if result.error:
            blocks.append({"error": result.error})
        return TextFeatures(
            text=result.text,
            confidence=result.average_confidence,
            blocks=blocks,
            backend="paddleocr",
        )
