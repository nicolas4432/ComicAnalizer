from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from features.magi_schema import BoundingBox, MagiPageAnalysis, MagiRegion
from features.ocr_paddle import box_iou


@dataclass(frozen=True)
class OCRGroupedText:
    group_id: str
    text: str
    block_indices: list[int]
    box: BoundingBox
    confidence: float | None
    source: str
    magi_text_region: dict[str, Any] | None = None
    magi_panel: dict[str, Any] | None = None
    blocks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "text": self.text,
            "block_indices": self.block_indices,
            "box": self.box.to_dict(),
            "confidence": self.confidence,
            "source": self.source,
            "magi_text_region": self.magi_text_region,
            "magi_panel": self.magi_panel,
            "blocks": self.blocks,
        }


def group_ocr_blocks(
    ocr_page: dict[str, Any],
    magi_page: MagiPageAnalysis | None = None,
    iou_threshold: float = 0.08,
) -> list[OCRGroupedText]:
    blocks = [
        block
        for block in ocr_page.get("blocks") or []
        if str(block.get("text") or "").strip()
    ]
    if not blocks:
        return []
    if magi_page is not None and magi_page.texts:
        groups = group_with_magi_text_regions(blocks, magi_page, iou_threshold=iou_threshold)
    else:
        groups = group_by_geometry(blocks)
    return sort_groups(groups)


def group_with_magi_text_regions(
    blocks: list[dict[str, Any]],
    magi_page: MagiPageAnalysis,
    iou_threshold: float,
) -> list[OCRGroupedText]:
    assigned: set[int] = set()
    groups: list[OCRGroupedText] = []

    for region in magi_page.texts:
        region_blocks: list[dict[str, Any]] = []
        for block in blocks:
            block_index = int(block.get("index") or 0)
            if block_index in assigned:
                continue
            block_box = block_to_box(block)
            if block_matches_region(block_box, region.box, iou_threshold=iou_threshold):
                region_blocks.append(block)
                assigned.add(block_index)
        if not region_blocks:
            continue
        groups.append(build_group(region_blocks, source="magi_text_region", magi_page=magi_page, region=region))

    remaining = [block for block in blocks if int(block.get("index") or 0) not in assigned]
    groups.extend(group_by_geometry(remaining, prefix="geo_unmatched"))
    return groups


def block_matches_region(block_box: BoundingBox, region_box: BoundingBox, iou_threshold: float) -> bool:
    if box_iou(block_box, region_box) >= iou_threshold:
        return True
    center = box_center(block_box)
    return region_box.x1 <= center[0] <= region_box.x2 and region_box.y1 <= center[1] <= region_box.y2


def group_by_geometry(blocks: list[dict[str, Any]], prefix: str = "geo") -> list[OCRGroupedText]:
    if not blocks:
        return []
    ordered = sorted(blocks, key=lambda block: (block_to_box(block).y1, block_to_box(block).x1))
    groups: list[list[dict[str, Any]]] = []
    for block in ordered:
        box = block_to_box(block)
        placed = False
        for group in groups:
            group_box = union_boxes([block_to_box(item) for item in group])
            median_height = median([block_to_box(item).height for item in group]) or box.height
            vertical_gap = max(0.0, box.y1 - group_box.y2)
            horizontal_overlap = min(box.x2, group_box.x2) - max(box.x1, group_box.x1)
            same_column = horizontal_overlap > -max(median_height, box.height) * 1.2
            if same_column and vertical_gap <= max(median_height, box.height) * 1.6:
                group.append(block)
                placed = True
                break
        if not placed:
            groups.append([block])
    return [build_group(group, source=prefix, explicit_index=index) for index, group in enumerate(groups)]


def build_group(
    blocks: list[dict[str, Any]],
    source: str,
    magi_page: MagiPageAnalysis | None = None,
    region: MagiRegion | None = None,
    explicit_index: int | None = None,
) -> OCRGroupedText:
    ordered = sort_blocks_for_reading(blocks)
    boxes = [block_to_box(block) for block in ordered]
    group_box = union_boxes(boxes)
    confidences = [
        float(block["confidence"])
        for block in ordered
        if isinstance(block.get("confidence"), (int, float))
    ]
    group_index = region.index if region is not None else explicit_index or 0
    group_id = f"{source}:{group_index}"
    panel = best_panel(group_box, magi_page) if magi_page is not None else None
    return OCRGroupedText(
        group_id=group_id,
        text=join_group_text([str(block.get("text") or "").strip() for block in ordered]),
        block_indices=[int(block.get("index") or 0) for block in ordered],
        box=group_box,
        confidence=sum(confidences) / len(confidences) if confidences else None,
        source=source,
        magi_text_region=region.to_dict() if region is not None else None,
        magi_panel=panel.to_dict() if panel is not None else None,
        blocks=ordered,
    )


def sort_blocks_for_reading(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(blocks, key=lambda block: (block_to_box(block).y1, block_to_box(block).x1))


def join_group_text(texts: list[str]) -> str:
    joined = " ".join(text for text in texts if text)
    replacements = {
        " ?": "?",
        " !": "!",
        " .": ".",
        " ,": ",",
        " ;": ";",
        " :": ":",
    }
    for old, new in replacements.items():
        joined = joined.replace(old, new)
    return joined


def best_panel(box: BoundingBox, magi_page: MagiPageAnalysis | None) -> MagiRegion | None:
    if magi_page is None or not magi_page.panels:
        return None
    center = box_center(box)
    containing = [
        panel
        for panel in magi_page.panels
        if panel.box.x1 <= center[0] <= panel.box.x2 and panel.box.y1 <= center[1] <= panel.box.y2
    ]
    if containing:
        return containing[0]
    return max(magi_page.panels, key=lambda panel: box_iou(box, panel.box))


def block_to_box(block: dict[str, Any]) -> BoundingBox:
    box = block.get("box") or {}
    return BoundingBox(
        x1=float(box.get("x1", 0.0)),
        y1=float(box.get("y1", 0.0)),
        x2=float(box.get("x2", 0.0)),
        y2=float(box.get("y2", 0.0)),
    )


def union_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        x1=min(box.x1 for box in boxes),
        y1=min(box.y1 for box in boxes),
        x2=max(box.x2 for box in boxes),
        y2=max(box.y2 for box in boxes),
    )


def box_center(box: BoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)


def sort_groups(groups: list[OCRGroupedText]) -> list[OCRGroupedText]:
    return sorted(groups, key=lambda group: (group.box.y1, group.box.x1))


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
