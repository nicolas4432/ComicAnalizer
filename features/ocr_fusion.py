from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from features.magi_schema import BoundingBox, MagiPageAnalysis
from features.ocr_paddle import box_iou


@dataclass(frozen=True)
class OCRTextOption:
    source: str
    text: str
    confidence: float | None
    box: BoundingBox | None
    target_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "text": self.text,
            "confidence": self.confidence,
            "box": self.box.to_dict() if self.box else None,
            "target_id": self.target_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class FusedOCRText:
    fusion_id: str
    display: int
    text: str
    confidence: float | None
    box: BoundingBox
    source: str
    options: list[OCRTextOption]
    linked_targets: list[str]
    review_flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.fusion_id,
            "kind": "ocr_fusion",
            "display": self.display,
            "text": self.text,
            "confidence": self.confidence,
            "box": self.box.to_dict(),
            "source": self.source,
            "options": [item.to_dict() for item in self.options],
            "linked_targets": self.linked_targets,
            "review_flags": self.review_flags,
        }


def fuse_ocr_texts(
    magi_page: MagiPageAnalysis,
    ocr_groups: list[dict[str, Any]],
    iou_threshold: float = 0.08,
) -> list[FusedOCRText]:
    magi_options = build_magi_options(magi_page)
    paddle_options = build_paddle_group_options(ocr_groups)
    used_paddle: set[int] = set()
    fused: list[FusedOCRText] = []

    for magi_index, magi_option in enumerate(magi_options):
        matches = [
            (idx, option)
            for idx, option in enumerate(paddle_options)
            if idx not in used_paddle and boxes_match(magi_option.box, option.box, iou_threshold)
        ]
        options = [magi_option] + [option for _idx, option in matches]
        for idx, _option in matches:
            used_paddle.add(idx)
        fused.append(build_fused_text(len(fused) + 1, options, anchor=magi_option.box, suffix=magi_index))

    for idx, option in enumerate(paddle_options):
        if idx in used_paddle:
            continue
        fused.append(build_fused_text(len(fused) + 1, [option], anchor=option.box, suffix=idx))

    return sorted(fused, key=lambda item: (item.box.y1, item.box.x1))


def build_magi_options(page: MagiPageAnalysis) -> list[OCRTextOption]:
    options: list[OCRTextOption] = []
    for region in page.texts:
        text = str(region.attributes.get("ocr_text") or "").strip()
        if not text:
            continue
        ocr_box = dict_to_box(region.attributes.get("ocr_box")) or region.box
        options.append(
            OCRTextOption(
                source="magi_ocr",
                text=text,
                confidence=None,
                box=ocr_box,
                target_id=f"magi_text:{region.index}",
                metadata={
                    "is_essential": region.attributes.get("is_essential"),
                    "region_index": region.index,
                },
            )
        )
    return options


def build_paddle_group_options(groups: list[dict[str, Any]]) -> list[OCRTextOption]:
    options: list[OCRTextOption] = []
    for index, group in enumerate(groups, 1):
        text = str(group.get("text") or "").strip()
        box = dict_to_box(group.get("box"))
        if not text or box is None:
            continue
        options.append(
            OCRTextOption(
                source="paddle_group",
                text=text,
                confidence=group.get("confidence"),
                box=box,
                target_id=f"ocr:group:{index}",
                metadata={
                    "group_id": group.get("group_id"),
                    "block_indices": group.get("block_indices") or [],
                    "group_source": group.get("source"),
                },
            )
        )
    return options


def build_fused_text(
    display: int,
    options: list[OCRTextOption],
    anchor: BoundingBox | None,
    suffix: int,
) -> FusedOCRText:
    chosen = choose_best_option(options)
    boxes = [item.box for item in options if item.box is not None]
    box = union_boxes(boxes) if boxes else anchor or BoundingBox(0, 0, 1, 1)
    flags = review_flags(options, chosen)
    return FusedOCRText(
        fusion_id=f"ocr:fusion:{suffix}:{display}",
        display=display,
        text=chosen.text,
        confidence=chosen.confidence,
        box=box,
        source=chosen.source,
        options=options,
        linked_targets=[item.target_id for item in options if item.target_id],
        review_flags=flags,
    )


def choose_best_option(options: list[OCRTextOption]) -> OCRTextOption:
    return max(options, key=option_score)


def option_score(option: OCRTextOption) -> float:
    text = option.text.strip()
    score = 0.0
    score += min(len(text), 120) / 120
    score += 0.2 if contains_space_or_sentence(text) else 0.0
    score -= 0.3 if looks_noisy(text) else 0.0
    if option.confidence is not None:
        score += max(0.0, min(float(option.confidence), 1.0)) * 0.35
    if option.source == "magi_ocr":
        score += 0.35
    return score


def review_flags(options: list[OCRTextOption], chosen: OCRTextOption) -> list[str]:
    flags: list[str] = []
    texts = {normalize_text(option.text) for option in options if option.text.strip()}
    if len(texts) > 1:
        flags.append("ocr_disagreement")
    if any(looks_noisy(option.text) for option in options):
        flags.append("possible_noise")
    if len(options) == 1 and chosen.source == "paddle_group":
        flags.append("missing_magi_text")
    if len(options) == 1 and chosen.source == "magi_ocr":
        flags.append("missing_paddle_text")
    return flags


def boxes_match(a: BoundingBox | None, b: BoundingBox | None, iou_threshold: float) -> bool:
    if a is None or b is None:
        return False
    if box_iou(a, b) >= iou_threshold:
        return True
    center = ((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2)
    return a.x1 <= center[0] <= a.x2 and a.y1 <= center[1] <= a.y2


def dict_to_box(value: Any) -> BoundingBox | None:
    if not isinstance(value, dict):
        return None
    try:
        return BoundingBox(
            x1=float(value.get("x1", 0.0)),
            y1=float(value.get("y1", 0.0)),
            x2=float(value.get("x2", 0.0)),
            y2=float(value.get("y2", 0.0)),
        )
    except (TypeError, ValueError):
        return None


def union_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        x1=min(box.x1 for box in boxes),
        y1=min(box.y1 for box in boxes),
        x2=max(box.x2 for box in boxes),
        y2=max(box.y2 for box in boxes),
    )


def contains_space_or_sentence(text: str) -> bool:
    return " " in text or any(char in text for char in ".?!,;:")


def looks_noisy(text: str) -> bool:
    clean = text.strip()
    if len(clean) <= 1:
        return True
    alnum = sum(char.isalnum() for char in clean)
    if alnum == 0:
        return True
    return (alnum / max(len(clean), 1)) < 0.45


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
