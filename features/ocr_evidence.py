from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from features.magi_schema import BoundingBox, MagiPageAnalysis, MagiRegion
from features.ocr_paddle import box_iou, polygon_to_box
from reports.box_visualization import sanitize_filename


SCHEMA_VERSION = "ocr_evidence.v2"


@dataclass(frozen=True)
class EvidenceExportStats:
    page_count: int
    block_count: int
    skipped_empty_blocks: int
    assets_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "page_count": self.page_count,
            "block_count": self.block_count,
            "skipped_empty_blocks": self.skipped_empty_blocks,
            "assets_count": self.assets_count,
        }


def export_ocr_evidence(
    ocr_report: dict[str, Any],
    output_dir: Path,
    magi_pages: list[MagiPageAnalysis] | None = None,
    image_root: Path | None = None,
    dataset_name: str = "test_1_clean",
    include_empty_blocks: bool = False,
    context_padding: int = 56,
    limit_pages: int | None = None,
    asset_policy: str = "priority",
    max_asset_blocks: int | None = 500,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    magi_by_page = build_magi_page_map(magi_pages or [])
    comparisons_by_path = build_comparison_path_map(ocr_report)
    comparisons_by_page_key = build_comparison_page_map(ocr_report)

    evidence_path = output_dir / "evidence.jsonl"
    correction_template_path = output_dir / "correction_template.jsonl"
    page_items = list(ocr_report.get("ocr_results") or [])
    if limit_pages is not None:
        page_items = page_items[: max(0, limit_pages)]

    page_count = 0
    block_count = 0
    skipped_empty = 0
    assets_count = 0
    asset_block_count = 0
    per_page: list[dict[str, Any]] = []

    with evidence_path.open("w", encoding="utf-8") as evidence_file, correction_template_path.open(
        "w", encoding="utf-8"
    ) as correction_file:
        for ocr_page in page_items:
            comparison = match_comparison(ocr_page, comparisons_by_path, comparisons_by_page_key)
            image_path = resolve_ocr_image_path(
                ocr_page=ocr_page,
                comparison=comparison,
                image_root=image_root,
                dataset_name=dataset_name,
            )
            if image_path is None:
                continue
            comic_id = infer_comic_id(image_path, comparison)
            file_name = image_path.name
            page_key = build_page_key(comic_id, file_name)
            magi_page = magi_by_page.get(page_key)

            image = Image.open(image_path).convert("RGB")
            image_size = image.size
            image_sha = file_sha256(image_path)
            page_count += 1
            page_block_count = 0

            for block in ocr_page.get("blocks") or []:
                text = str(block.get("text") or "").strip()
                if not text and not include_empty_blocks:
                    skipped_empty += 1
                    continue
                block_index = int(block.get("index") or 0)
                evidence_id = build_evidence_id(comic_id, image_path.stem, block_index)
                block_box = box_from_block(block)
                block_polygon = block.get("polygon") or []
                magi_context = build_magi_context(block_box, magi_page)
                page_metrics = build_page_metrics(comparison)
                review = build_review_payload(
                    block=block,
                    block_box=block_box,
                    magi_context=magi_context,
                    page_metrics=page_metrics,
                )

                should_write_assets = should_export_assets(
                    asset_policy=asset_policy,
                    review=review,
                    assets_written_for_blocks=asset_block_count,
                    max_asset_blocks=max_asset_blocks,
                )
                asset_paths: dict[str, str] = {}
                if should_write_assets:
                    asset_paths = write_block_assets(
                        image=image,
                        output_root=output_dir,
                        assets_dir=assets_dir,
                        comic_id=comic_id,
                        page_stem=image_path.stem,
                        block_index=block_index,
                        block_box=block_box,
                        block_polygon=block_polygon,
                        context_padding=context_padding,
                        magi_context=magi_context,
                    )
                    assets_count += len(asset_paths)
                    asset_block_count += 1

                item = {
                    "schema_version": SCHEMA_VERSION,
                    "evidence_id": evidence_id,
                    "source": {
                        "image_path": str(image_path),
                        "image_sha256": image_sha,
                        "comic_id": comic_id,
                        "page_file": file_name,
                        "page_key": page_key,
                        "image_size": {"width": image_size[0], "height": image_size[1]},
                    },
                    "ocr": {
                        "backend": ocr_page.get("backend"),
                        "lang": ocr_page.get("lang"),
                        "page_elapsed_seconds": ocr_page.get("elapsed_seconds"),
                        "block": block,
                        "raw_text": text,
                        "confidence": block.get("confidence"),
                    },
                    "page_metrics": page_metrics,
                    "geometry": build_geometry_features(block_box, block_polygon, image_size),
                    "magi_context": magi_context,
                    "visual_assets": asset_paths,
                    "review": review,
                    "human_label": {
                        "status": "unreviewed",
                        "corrected_text": None,
                        "is_correct": None,
                        "error_type": None,
                        "is_false_positive": None,
                        "belongs_to_bubble": None,
                        "group_id": None,
                        "notes": None,
                    },
                    "training_tags": build_training_tags(block, magi_context),
                }
                evidence_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                correction_file.write(json.dumps(build_correction_template(item), ensure_ascii=False) + "\n")
                block_count += 1
                page_block_count += 1

            per_page.append(
                {
                    "comic_id": comic_id,
                    "page_file": file_name,
                    "page_key": page_key,
                    "block_count": page_block_count,
                    "magi_context_available": magi_page is not None,
                    "page_metrics": build_page_metrics(comparison),
                }
            )

    stats = EvidenceExportStats(
        page_count=page_count,
        block_count=block_count,
        skipped_empty_blocks=skipped_empty,
        assets_count=assets_count,
    )
    index = {
        "schema_version": SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "files": {
            "evidence": str(evidence_path),
            "correction_template": str(correction_template_path),
            "assets": str(assets_dir),
        },
        "export_options": {
            "dataset_name": dataset_name,
            "image_root": str(image_root.expanduser().resolve()) if image_root else None,
            "asset_policy": asset_policy,
            "max_asset_blocks": max_asset_blocks,
            "context_padding": context_padding,
        },
        "stats": stats.to_dict(),
        "review_summary": build_review_summary(evidence_path),
        "pages": per_page,
    }
    (output_dir / "evidence_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return index


def build_magi_page_map(pages: list[MagiPageAnalysis]) -> dict[str, MagiPageAnalysis]:
    return {
        build_page_key(page.comic_id or "unknown", page.file_name): page
        for page in pages
    }


def build_comparison_path_map(ocr_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for item in ocr_report.get("comparisons") or []:
        image_path = str(item.get("image_path") or "")
        if image_path:
            mapping[normalize_path_key(image_path)] = item
    return mapping


def build_comparison_page_map(ocr_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for item in ocr_report.get("comparisons") or []:
        comic_id = str(item.get("comic_id") or "unknown")
        file_name = str(item.get("file_name") or Path(str(item.get("image_path") or "")).name)
        if file_name:
            mapping[build_page_key(comic_id, file_name)] = item
    return mapping


def match_comparison(
    ocr_page: dict[str, Any],
    comparisons_by_path: dict[str, dict[str, Any]],
    comparisons_by_page_key: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    path_match = comparisons_by_path.get(normalize_path_key(str(ocr_page.get("path") or "")))
    if path_match:
        return path_match
    if comparisons_by_page_key:
        path = Path(str(ocr_page.get("path") or ""))
        inferred_comic = infer_comic_id(path, None)
        page_key = build_page_key(inferred_comic, path.name)
        return comparisons_by_page_key.get(page_key)
    return None


def resolve_ocr_image_path(
    ocr_page: dict[str, Any],
    comparison: dict[str, Any] | None,
    image_root: Path | None,
    dataset_name: str,
) -> Path | None:
    original_path = Path(str(ocr_page.get("path") or ""))
    if original_path.exists():
        return original_path
    if comparison:
        comparison_path = Path(str(comparison.get("image_path") or ""))
        if comparison_path.exists():
            return comparison_path
    if image_root is None:
        return None
    root = image_root.expanduser().resolve()
    comic_id = str((comparison or {}).get("comic_id") or infer_comic_id(original_path, comparison))
    file_name = str((comparison or {}).get("file_name") or original_path.name)
    candidates = [
        root / comic_id / dataset_name / file_name,
        root / comic_id / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def normalize_path_key(path: str) -> str:
    return path.replace("\\", "/").lower()


def infer_comic_id(image_path: Path, comparison: dict[str, Any] | None) -> str:
    if comparison and comparison.get("comic_id"):
        return str(comparison["comic_id"])
    parts = image_path.parts
    if "by_comic" in parts:
        index = parts.index("by_comic")
        if index + 1 < len(parts):
            return parts[index + 1]
    return image_path.parent.parent.name if image_path.parent.name.startswith("test_") else "unknown"


def build_page_key(comic_id: str | None, file_name: str) -> str:
    return f"{comic_id or 'unknown'}:{Path(file_name).stem}"


def build_evidence_id(comic_id: str, page_stem: str, block_index: int) -> str:
    return f"{sanitize_filename(comic_id)}_{sanitize_filename(page_stem)}_block_{block_index:03d}"


def box_from_block(block: dict[str, Any]) -> BoundingBox:
    box = block.get("box") or {}
    if box:
        return BoundingBox(
            x1=float(box.get("x1", 0.0)),
            y1=float(box.get("y1", 0.0)),
            x2=float(box.get("x2", 0.0)),
            y2=float(box.get("y2", 0.0)),
        )
    return polygon_to_box(block.get("polygon") or [])


def write_block_assets(
    image: Image.Image,
    output_root: Path,
    assets_dir: Path,
    comic_id: str,
    page_stem: str,
    block_index: int,
    block_box: BoundingBox,
    block_polygon: list[list[float]],
    context_padding: int,
    magi_context: dict[str, Any],
) -> dict[str, str]:
    block_dir = assets_dir / sanitize_filename(comic_id) / sanitize_filename(page_stem)
    block_dir.mkdir(parents=True, exist_ok=True)
    stem = f"block_{block_index:03d}"

    block_crop_box = clamp_box(block_box, image.size, padding=4)
    context_crop_box = clamp_box(block_box, image.size, padding=context_padding)

    block_crop_path = block_dir / f"{stem}_crop.jpg"
    context_crop_path = block_dir / f"{stem}_context.jpg"
    overlay_path = block_dir / f"{stem}_overlay.jpg"

    image.crop(box_tuple(block_crop_box)).save(block_crop_path)
    image.crop(box_tuple(context_crop_box)).save(context_crop_path)
    draw_block_overlay_crop(
        image=image,
        output_path=overlay_path,
        crop_box=context_crop_box,
        block_polygon=block_polygon,
        block_box=block_box,
        magi_context=magi_context,
    )

    assets = {
        "block_crop": relative_path(block_crop_path, output_root),
        "context_crop": relative_path(context_crop_path, output_root),
        "overlay_crop": relative_path(overlay_path, output_root),
    }

    magi_text = (magi_context.get("text_region") or {}).get("box")
    if magi_text:
        magi_box = dict_to_box(magi_text)
        magi_crop = clamp_box(magi_box, image.size, padding=8)
        magi_crop_path = block_dir / f"{stem}_magi_text_region.jpg"
        image.crop(box_tuple(magi_crop)).save(magi_crop_path)
        assets["magi_text_region_crop"] = relative_path(magi_crop_path, output_root)

    panel = (magi_context.get("panel") or {}).get("box")
    if panel:
        panel_box = clamp_box(dict_to_box(panel), image.size, padding=4)
        panel_crop_path = block_dir / f"{stem}_panel.jpg"
        image.crop(box_tuple(panel_box)).save(panel_crop_path)
        assets["panel_crop"] = relative_path(panel_crop_path, output_root)

    return assets


def draw_block_overlay_crop(
    image: Image.Image,
    output_path: Path,
    crop_box: BoundingBox,
    block_polygon: list[list[float]],
    block_box: BoundingBox,
    magi_context: dict[str, Any],
) -> None:
    crop = image.crop(box_tuple(crop_box)).convert("RGB")
    draw = ImageDraw.Draw(crop)

    def offset_box(box: BoundingBox) -> tuple[float, float, float, float]:
        return (
            box.x1 - crop_box.x1,
            box.y1 - crop_box.y1,
            box.x2 - crop_box.x1,
            box.y2 - crop_box.y1,
        )

    magi_text = (magi_context.get("text_region") or {}).get("box")
    if magi_text:
        draw.rectangle(offset_box(dict_to_box(magi_text)), outline="dodgerblue", width=3)

    if block_polygon:
        points = [(point[0] - crop_box.x1, point[1] - crop_box.y1) for point in block_polygon]
        draw.line(points + [points[0]], fill="orange", width=4)
    else:
        draw.rectangle(offset_box(block_box), outline="orange", width=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)


def build_magi_context(block_box: BoundingBox, page: MagiPageAnalysis | None) -> dict[str, Any]:
    if page is None:
        return {
            "available": False,
            "text_region": None,
            "panel": None,
            "tail": None,
            "character": None,
            "associations": [],
        }

    text_region, text_iou = best_region(block_box, page.texts)
    panel, panel_score = best_region(block_box, page.panels, prefer_center=True)
    tail, tail_iou = best_region(block_box, page.tails)
    character, character_iou = best_region(block_box, page.characters)
    associations = []
    if text_region is not None:
        associations = [
            item.to_dict()
            for item in page.text_character_associations + page.text_tail_associations
            if item.source_index == text_region.index
        ]

    return {
        "available": True,
        "page_id": page.page_id,
        "text_region": region_match(text_region, text_iou),
        "panel": region_match(panel, panel_score),
        "tail": region_match(tail, tail_iou),
        "character": region_match(character, character_iou),
        "associations": associations,
    }


def best_region(
    block_box: BoundingBox,
    regions: list[MagiRegion],
    prefer_center: bool = False,
) -> tuple[MagiRegion | None, float | None]:
    if not regions:
        return None, None
    best: tuple[MagiRegion | None, float] = (None, 0.0)
    center = box_center(block_box)
    for region in regions:
        score = box_iou(block_box, region.box)
        if prefer_center and point_inside_box(center, region.box):
            score = max(score, 1.0)
        if score > best[1]:
            best = (region, score)
    return best[0], best[1]


def region_match(region: MagiRegion | None, score: float | None) -> dict[str, Any] | None:
    if region is None:
        return None
    return {
        "id": region.id,
        "kind": region.kind,
        "index": region.index,
        "score": score,
        "box": region.box.to_dict(),
        "attributes": region.attributes,
    }


def build_geometry_features(
    block_box: BoundingBox,
    polygon: list[list[float]],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    width, height = image_size
    center_x, center_y = box_center(block_box)
    return {
        "box": block_box.to_dict(),
        "polygon": polygon,
        "center": {"x": center_x, "y": center_y},
        "normalized_box": {
            "x1": block_box.x1 / width if width else 0.0,
            "y1": block_box.y1 / height if height else 0.0,
            "x2": block_box.x2 / width if width else 0.0,
            "y2": block_box.y2 / height if height else 0.0,
        },
        "area_ratio": block_box.area / (width * height) if width and height else 0.0,
        "estimated_angle_degrees": estimate_polygon_angle(polygon),
    }


def build_training_tags(block: dict[str, Any], magi_context: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    confidence = block.get("confidence")
    if isinstance(confidence, (int, float)):
        if confidence < 0.5:
            tags.append("low_confidence")
        elif confidence >= 0.95:
            tags.append("high_confidence")
    text = str(block.get("text") or "")
    if len(text) <= 2:
        tags.append("short_text")
    if magi_context.get("text_region"):
        tags.append("inside_magi_text_region")
    else:
        tags.append("no_magi_text_match")
    if magi_context.get("panel"):
        tags.append("inside_panel")
    if looks_like_noise_text(text):
        tags.append("noise_like_text")
    return tags


def build_correction_template(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "ocr_correction.v1",
        "evidence_id": item["evidence_id"],
        "comic_id": item["source"]["comic_id"],
        "page_file": item["source"]["page_file"],
        "block_index": item["ocr"]["block"].get("index"),
        "raw_text": item["ocr"]["raw_text"],
        "corrected_text": item["ocr"]["raw_text"],
        "is_correct": None,
        "error_type": None,
        "is_false_positive": None,
        "belongs_to_bubble": None,
        "group_id": None,
        "review_priority": item.get("review", {}).get("priority"),
        "review_flags": item.get("review", {}).get("flags", []),
        "asset_hint": item.get("visual_assets", {}).get("overlay_crop"),
        "notes": None,
        "reviewer": None,
    }


def build_page_metrics(comparison: dict[str, Any] | None) -> dict[str, Any]:
    if not comparison:
        return {}
    keys = [
        "magi_text_regions",
        "paddle_text_blocks",
        "matched_regions",
        "magi_only_regions",
        "paddle_only_blocks",
        "paddle_avg_confidence",
        "paddle_elapsed_seconds",
        "visual_path",
    ]
    return {key: comparison.get(key) for key in keys if key in comparison}


def build_review_payload(
    block: dict[str, Any],
    block_box: BoundingBox,
    magi_context: dict[str, Any],
    page_metrics: dict[str, Any],
) -> dict[str, Any]:
    flags: list[str] = []
    priority = 0
    confidence = block.get("confidence")
    text = str(block.get("text") or "").strip()
    paddle_only_blocks = int(page_metrics.get("paddle_only_blocks") or 0)
    magi_text_regions = int(page_metrics.get("magi_text_regions") or 0)

    if isinstance(confidence, (int, float)) and confidence < 0.6:
        flags.append("low_confidence_ocr")
        priority += 30
    if not magi_context.get("text_region"):
        flags.append("outside_magi_text")
        priority += 20
    if paddle_only_blocks >= 40:
        flags.append("page_many_paddle_only_blocks")
        priority += 25
    if magi_text_regions == 0 and isinstance(confidence, (int, float)) and confidence >= 0.85:
        flags.append("magi_missed_text_candidate")
        priority += 20
    if looks_like_noise_text(text):
        flags.append("noise_like_text")
        priority += 35
    if block_box.area < 120:
        flags.append("tiny_text_box")
        priority += 5
    if len(text) <= 2:
        flags.append("short_text")
        priority += 5

    return {
        "priority": priority,
        "flags": flags,
        "recommended_action": recommend_action(flags),
    }


def recommend_action(flags: list[str]) -> str:
    if "noise_like_text" in flags or "low_confidence_ocr" in flags:
        return "review_false_positive"
    if "magi_missed_text_candidate" in flags:
        return "review_magi_missed_text"
    if "outside_magi_text" in flags:
        return "review_context"
    return "spot_check"


def looks_like_noise_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) <= 2:
        return False
    alpha = sum(char.isalpha() for char in stripped)
    digits = sum(char.isdigit() for char in stripped)
    symbols = len(stripped) - alpha - digits - sum(char.isspace() for char in stripped)
    repeated = max((stripped.count(char) for char in set(stripped)), default=0)
    alpha_ratio = alpha / max(1, len(stripped))
    symbol_ratio = symbols / max(1, len(stripped))
    return (
        alpha_ratio < 0.35
        or symbol_ratio > 0.45
        or repeated / max(1, len(stripped)) > 0.65
    )


def should_export_assets(
    asset_policy: str,
    review: dict[str, Any],
    assets_written_for_blocks: int,
    max_asset_blocks: int | None,
) -> bool:
    if asset_policy == "none":
        return False
    if max_asset_blocks is not None and assets_written_for_blocks >= max_asset_blocks:
        return False
    if asset_policy == "all":
        return True
    if asset_policy == "priority":
        return bool(review.get("flags")) or int(review.get("priority") or 0) > 0
    raise ValueError(f"Unsupported asset policy: {asset_policy}")


def build_review_summary(evidence_path: Path) -> dict[str, Any]:
    from collections import Counter

    flags: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    priorities: list[int] = []
    for line in evidence_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        review = item.get("review") or {}
        priorities.append(int(review.get("priority") or 0))
        flags.update(review.get("flags") or [])
        action = review.get("recommended_action")
        if action:
            actions[action] += 1
    return {
        "flag_counts": dict(flags.most_common()),
        "recommended_action_counts": dict(actions.most_common()),
        "max_priority": max(priorities) if priorities else 0,
        "avg_priority": sum(priorities) / len(priorities) if priorities else 0.0,
    }


def clamp_box(box: BoundingBox, image_size: tuple[int, int], padding: int = 0) -> BoundingBox:
    width, height = image_size
    return BoundingBox(
        x1=max(0.0, box.x1 - padding),
        y1=max(0.0, box.y1 - padding),
        x2=min(float(width), box.x2 + padding),
        y2=min(float(height), box.y2 + padding),
    )


def box_tuple(box: BoundingBox) -> tuple[int, int, int, int]:
    return (int(box.x1), int(box.y1), int(box.x2), int(box.y2))


def dict_to_box(value: dict[str, Any]) -> BoundingBox:
    return BoundingBox(
        x1=float(value["x1"]),
        y1=float(value["y1"]),
        x2=float(value["x2"]),
        y2=float(value["y2"]),
    )


def box_center(box: BoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)


def point_inside_box(point: tuple[float, float], box: BoundingBox) -> bool:
    return box.x1 <= point[0] <= box.x2 and box.y1 <= point[1] <= box.y2


def estimate_polygon_angle(polygon: list[list[float]]) -> float | None:
    if len(polygon) < 2:
        return None
    import math

    x1, y1 = polygon[0]
    x2, y2 = polygon[1]
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)
