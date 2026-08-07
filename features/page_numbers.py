from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


PAGE_NUMBER_PATTERNS = (
    re.compile(r"^\s*(?:p(?:age|g)?\.?\s*)?([0-9]{1,4})\s*$", re.IGNORECASE),
    re.compile(r"^\s*[-–—]?\s*([0-9]{1,4})\s*[-–—]?\s*$"),
)


@dataclass(frozen=True)
class PageNumberCandidate:
    value: int
    raw_text: str
    score: float
    confidence: float | None
    block_index: int | None
    box: dict[str, Any]
    edge_zone: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "raw_text": self.raw_text,
            "score": self.score,
            "confidence": self.confidence,
            "block_index": self.block_index,
            "box": self.box,
            "edge_zone": self.edge_zone,
            "reasons": self.reasons,
        }


def detect_page_number_candidates(
    ocr_page: dict[str, Any] | None,
    image_size: tuple[int, int] | None = None,
    min_score: float = 0.55,
) -> list[PageNumberCandidate]:
    if not ocr_page:
        return []
    blocks = [block for block in ocr_page.get("blocks") or [] if str(block.get("text") or "").strip()]
    if not blocks:
        return []
    width, height = image_size or infer_image_size_from_blocks(blocks)
    candidates: list[PageNumberCandidate] = []
    for block in blocks:
        text = normalize_candidate_text(str(block.get("text") or ""))
        value = extract_page_number_value(text)
        if value is None:
            continue
        candidate = score_candidate(block=block, value=value, width=width, height=height)
        if is_weak_candidate(candidate):
            continue
        if candidate.score >= min_score:
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def normalize_candidate_text(text: str) -> str:
    return text.strip()


def extract_page_number_value(text: str) -> int | None:
    clean = text.strip()
    for pattern in PAGE_NUMBER_PATTERNS:
        match = pattern.match(clean)
        if not match:
            continue
        try:
            value = int(match.group(1))
        except ValueError:
            return None
        if 1 <= value <= 9999:
            return value
    return None


def score_candidate(
    block: dict[str, Any],
    value: int,
    width: int,
    height: int,
) -> PageNumberCandidate:
    box = dict(block.get("box") or {})
    x1 = float(box.get("x1", 0.0))
    y1 = float(box.get("y1", 0.0))
    x2 = float(box.get("x2", 0.0))
    y2 = float(box.get("y2", 0.0))
    block_width = max(1.0, x2 - x1)
    block_height = max(1.0, y2 - y1)
    area_ratio = (block_width * block_height) / max(1.0, float(width * height))
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    confidence = block.get("confidence")
    confidence_score = float(confidence) if isinstance(confidence, (int, float)) else 0.55
    edge_zone, edge_score, edge_reasons = edge_position_score(
        center_x=center_x,
        center_y=center_y,
        width=width,
        height=height,
    )
    size_score, size_reason = size_score_for_page_number(area_ratio, block_height, height)
    text_score = 1.0 if len(str(value)) <= 3 else 0.75
    score = (confidence_score * 0.35) + (edge_score * 0.35) + (size_score * 0.2) + (text_score * 0.1)
    reasons = edge_reasons + [size_reason, f"value={value}"]
    if confidence_score < 0.45:
        reasons.append("low_ocr_confidence")
    return PageNumberCandidate(
        value=value,
        raw_text=str(block.get("text") or ""),
        score=round(score, 4),
        confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        block_index=int(block["index"]) if isinstance(block.get("index"), int) else None,
        box=box,
        edge_zone=edge_zone,
        reasons=reasons,
    )


def is_weak_candidate(candidate: PageNumberCandidate) -> bool:
    if candidate.confidence is not None and candidate.confidence < 0.45:
        return candidate.edge_zone not in {"top", "bottom"}
    if candidate.edge_zone == "center":
        return True
    return False


def edge_position_score(
    center_x: float,
    center_y: float,
    width: int,
    height: int,
) -> tuple[str, float, list[str]]:
    x_ratio = center_x / max(1.0, float(width))
    y_ratio = center_y / max(1.0, float(height))
    horizontal_edge = min(x_ratio, 1.0 - x_ratio)
    vertical_edge = min(y_ratio, 1.0 - y_ratio)
    reasons: list[str] = []
    if y_ratio >= 0.86:
        reasons.append("bottom_edge")
        return "bottom", 1.0, reasons
    if y_ratio <= 0.14:
        reasons.append("top_edge")
        return "top", 0.85, reasons
    if horizontal_edge <= 0.08:
        reasons.append("side_edge")
        return "side", 0.65, reasons
    if vertical_edge <= 0.22:
        reasons.append("near_vertical_edge")
        return "near_edge", 0.45, reasons
    return "center", 0.15, ["center_page"]


def size_score_for_page_number(
    area_ratio: float,
    block_height: float,
    image_height: int,
) -> tuple[float, str]:
    height_ratio = block_height / max(1.0, float(image_height))
    if area_ratio <= 0.004 and height_ratio <= 0.035:
        return 1.0, "small_text"
    if area_ratio <= 0.012 and height_ratio <= 0.06:
        return 0.72, "medium_small_text"
    return 0.25, "large_text"


def infer_image_size_from_blocks(blocks: list[dict[str, Any]]) -> tuple[int, int]:
    max_x = 1.0
    max_y = 1.0
    for block in blocks:
        box = block.get("box") or {}
        max_x = max(max_x, float(box.get("x2", 1.0)))
        max_y = max(max_y, float(box.get("y2", 1.0)))
    return int(max_x), int(max_y)
