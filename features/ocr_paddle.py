from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from PIL import Image

from features.magi_schema import BoundingBox, MagiPageAnalysis


@dataclass(frozen=True)
class PaddleOCRBlock:
    index: int
    text: str
    confidence: float | None
    polygon: list[list[float]]
    box: BoundingBox

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "confidence": self.confidence,
            "polygon": self.polygon,
            "box": self.box.to_dict(),
        }


@dataclass
class PaddleOCRPageResult:
    path: str
    backend: str
    lang: str
    elapsed_seconds: float
    blocks: list[PaddleOCRBlock] = field(default_factory=list)
    error: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if block.text)

    @property
    def average_confidence(self) -> float | None:
        confidences = [
            block.confidence for block in self.blocks if block.confidence is not None
        ]
        return sum(confidences) / len(confidences) if confidences else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "backend": self.backend,
            "lang": self.lang,
            "elapsed_seconds": self.elapsed_seconds,
            "text": self.text,
            "block_count": len(self.blocks),
            "average_confidence": self.average_confidence,
            "blocks": [block.to_dict() for block in self.blocks],
            "error": self.error,
        }


@dataclass(frozen=True)
class OCRComparison:
    page_id: str
    comic_id: str | None
    file_name: str
    magi_text_regions: int
    paddle_text_blocks: int
    matched_regions: int
    magi_only_regions: int
    paddle_only_blocks: int
    avg_iou: float | None
    paddle_elapsed_seconds: float
    paddle_avg_confidence: float | None
    paddle_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "comic_id": self.comic_id,
            "file_name": self.file_name,
            "magi_text_regions": self.magi_text_regions,
            "paddle_text_blocks": self.paddle_text_blocks,
            "matched_regions": self.matched_regions,
            "magi_only_regions": self.magi_only_regions,
            "paddle_only_blocks": self.paddle_only_blocks,
            "avg_iou": self.avg_iou,
            "paddle_elapsed_seconds": self.paddle_elapsed_seconds,
            "paddle_avg_confidence": self.paddle_avg_confidence,
            "paddle_error": self.paddle_error,
        }


class PaddleOCRComplementaryExtractor:
    """Small wrapper that keeps PaddleOCR behind a replaceable boundary."""

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = False,
        show_log: bool = False,
    ) -> None:
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.show_log = show_log
        self._ocr = None

    def load(self) -> Any:
        if self._ocr is not None:
            return self._ocr
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("FLAGS_enable_pir_api", "0")
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR could not be imported. Install/verify paddleocr and "
                f"paddlepaddle first. Original import error: {exc}"
            ) from exc

        try:
            self._ocr = PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=self.use_angle_cls,
                text_det_limit_side_len=1280,
            )
        except (TypeError, ValueError):
            # Older PaddleOCR versions use the legacy angle-classifier arguments.
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                show_log=self.show_log,
            )
        return self._ocr

    def extract_page(self, path: str | Path) -> PaddleOCRPageResult:
        path = Path(path).expanduser().resolve()
        start = time.perf_counter()
        try:
            raw = self._run_ocr(path)
            blocks = normalize_paddle_output(raw)
            error = None
        except Exception as exc:  # noqa: BLE001 - this is a diagnostic tool.
            blocks = []
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        return PaddleOCRPageResult(
            path=str(path),
            backend="paddleocr",
            lang=self.lang,
            elapsed_seconds=elapsed,
            blocks=blocks,
            error=error,
        )

    def extract_regions(
        self,
        path: str | Path,
        regions: list[BoundingBox],
    ) -> list[PaddleOCRPageResult]:
        path = Path(path).expanduser().resolve()
        image = Image.open(path).convert("RGB")
        results: list[PaddleOCRPageResult] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for index, box in enumerate(regions):
                crop_path = Path(tmp_dir) / f"region_{index:03d}.jpg"
                image.crop((box.x1, box.y1, box.x2, box.y2)).save(crop_path)
                result = self.extract_page(crop_path)
                results.append(result)
        return results

    def _run_ocr(self, path: Path) -> Any:
        ocr = self.load()
        if hasattr(ocr, "ocr"):
            try:
                return ocr.ocr(str(path), cls=self.use_angle_cls)
            except TypeError as exc:
                if "unexpected keyword argument 'cls'" not in str(exc):
                    raise
                return ocr.ocr(str(path))
        if hasattr(ocr, "predict"):
            return ocr.predict(str(path))
        raise RuntimeError("Unsupported PaddleOCR object: no ocr() or predict() method.")


def normalize_paddle_output(raw: Any) -> list[PaddleOCRBlock]:
    entries = flatten_legacy_paddle_output(raw)
    blocks: list[PaddleOCRBlock] = []
    for index, item in enumerate(entries):
        parsed = parse_legacy_item(item)
        if parsed is None:
            continue
        polygon, text, confidence = parsed
        blocks.append(
            PaddleOCRBlock(
                index=len(blocks),
                text=text,
                confidence=confidence,
                polygon=polygon,
                box=polygon_to_box(polygon),
            )
        )
    return blocks


def flatten_legacy_paddle_output(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return flatten_dict_output(raw)
    if isinstance(raw, (list, tuple)):
        if not raw:
            return []
        if is_ocr_item(raw):
            return [raw]
        flattened: list[Any] = []
        for item in raw:
            flattened.extend(flatten_legacy_paddle_output(item))
        return flattened
    if hasattr(raw, "json"):
        return flatten_legacy_paddle_output(raw.json)
    return []


def flatten_dict_output(raw: dict[str, Any]) -> list[Any]:
    candidates: list[Any] = []
    for key in ("rec_texts", "dt_polys", "rec_scores"):
        if key in raw:
            candidates.append(raw[key])
    if {"rec_texts", "dt_polys"}.issubset(raw):
        scores = raw.get("rec_scores") or [None] * len(raw["rec_texts"])
        return [
            [poly, [text, score]]
            for poly, text, score in zip(raw["dt_polys"], raw["rec_texts"], scores)
        ]
    for value in raw.values():
        candidates.extend(flatten_legacy_paddle_output(value))
    return candidates


def is_ocr_item(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and (isinstance(value[0], (list, tuple)) or hasattr(value[0], "tolist"))
        and isinstance(value[1], (list, tuple))
    )


def parse_legacy_item(item: Any) -> tuple[list[list[float]], str, float | None] | None:
    if not is_ocr_item(item):
        return None
    polygon_raw = item[0]
    payload = item[1]
    if len(payload) < 1:
        return None
    text = str(payload[0])
    confidence = None
    if len(payload) > 1 and payload[1] is not None:
        try:
            confidence = float(payload[1])
        except (TypeError, ValueError):
            confidence = None
    polygon = normalize_polygon(polygon_raw)
    if not polygon:
        return None
    return polygon, text, confidence


def normalize_polygon(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    points: list[list[float]] = []
    if not isinstance(value, (list, tuple)):
        return points
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            continue
    return points


def polygon_to_box(polygon: list[list[float]]) -> BoundingBox:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return BoundingBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))


def compare_magi_with_paddle(
    page: MagiPageAnalysis,
    ocr_result: PaddleOCRPageResult,
    iou_threshold: float = 0.15,
) -> OCRComparison:
    matched_texts: set[int] = set()
    matched_blocks: set[int] = set()
    ious: list[float] = []
    for text_region in page.texts:
        best_iou = 0.0
        best_block_index = None
        for block in ocr_result.blocks:
            score = box_iou(text_region.box, block.box)
            if score > best_iou:
                best_iou = score
                best_block_index = block.index
        if best_block_index is not None and best_iou >= iou_threshold:
            matched_texts.add(text_region.index)
            matched_blocks.add(best_block_index)
            ious.append(best_iou)

    return OCRComparison(
        page_id=page.page_id,
        comic_id=page.comic_id,
        file_name=page.file_name,
        magi_text_regions=page.text_count,
        paddle_text_blocks=len(ocr_result.blocks),
        matched_regions=len(matched_texts),
        magi_only_regions=page.text_count - len(matched_texts),
        paddle_only_blocks=len(ocr_result.blocks) - len(matched_blocks),
        avg_iou=sum(ious) / len(ious) if ious else None,
        paddle_elapsed_seconds=ocr_result.elapsed_seconds,
        paddle_avg_confidence=ocr_result.average_confidence,
        paddle_error=ocr_result.error,
    )


def box_iou(left: BoundingBox, right: BoundingBox) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    union = left.area + right.area - intersection
    return intersection / union if union > 0 else 0.0
