from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from features.ocr_grouping import group_ocr_blocks
from reports.box_visualization import draw_ocr_grouped_overlay, sanitize_filename
from tools.analyze_magi_results import load_magi_bundle, normalize_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype OCR grouping calibration into bubble/sentence-level text."
    )
    parser.add_argument("--ocr-report", required=True)
    parser.add_argument("--magi-input", default=None)
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional local by_comic image root to remap Colab /content paths.",
    )
    parser.add_argument("--dataset-name", default="test_1_clean")
    parser.add_argument("--comic-id", required=True)
    parser.add_argument("--page-file", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/runs/ocr_grouping_calibration",
        help="Directory for grouped JSON and visual output.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ocr_report = json.loads(Path(args.ocr_report).expanduser().read_text(encoding="utf-8"))
    ocr_page = find_ocr_page(ocr_report, args.comic_id, args.page_file)
    if ocr_page is None:
        raise RuntimeError(f"OCR page not found: {args.comic_id}/{args.page_file}")
    ocr_page = dict(ocr_page)
    image_path = resolve_image_path(
        ocr_page=ocr_page,
        image_root=Path(args.image_root).expanduser() if args.image_root else None,
        comic_id=args.comic_id,
        page_file=args.page_file,
        dataset_name=args.dataset_name,
    )
    if image_path:
        ocr_page["path"] = str(image_path)

    magi_page = None
    if args.magi_input:
        pages = normalize_pages(load_magi_bundle(Path(args.magi_input).expanduser()))
        magi_page = next(
            (
                page
                for page in pages
                if page.comic_id == args.comic_id and page.file_name == args.page_file
            ),
            None,
        )

    groups = group_ocr_blocks(
        ocr_page=ocr_page,
        magi_page=magi_page,
        iou_threshold=args.iou_threshold,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    comic_dir = output_dir / sanitize_filename(args.comic_id)
    comic_dir.mkdir(parents=True, exist_ok=True)
    page_stem = Path(args.page_file).stem
    json_path = comic_dir / f"{sanitize_filename(page_stem)}_ocr_groups.json"
    visual_path = comic_dir / f"{sanitize_filename(page_stem)}_ocr_groups.jpg"

    payload = {
        "schema_version": "ocr_grouping_calibration.v1",
        "comic_id": args.comic_id,
        "page_file": args.page_file,
        "image_path": ocr_page.get("path"),
        "ocr_block_count": len(
            [block for block in ocr_page.get("blocks") or [] if str(block.get("text") or "").strip()]
        ),
        "group_count": len(groups),
        "magi_context_used": magi_page is not None,
        "iou_threshold": args.iou_threshold,
        "groups": [group.to_dict() for group in groups],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_ocr_grouped_overlay(
        image_path=ocr_page["path"],
        groups=groups,
        output_path=visual_path,
    )

    print(json.dumps(summary(payload), indent=2, ensure_ascii=False))
    print(f"Wrote grouped OCR JSON: {json_path}")
    print(f"Wrote grouped OCR visual: {visual_path}")


def find_ocr_page(
    ocr_report: dict[str, Any],
    comic_id: str,
    page_file: str,
) -> dict[str, Any] | None:
    normalized_suffix = f"/by_comic/{comic_id}/"
    for page in ocr_report.get("ocr_results") or []:
        path = str(page.get("path") or "").replace("\\", "/")
        if path.endswith(f"/{page_file}") and normalized_suffix in path:
            return page
    return None


def resolve_image_path(
    ocr_page: dict[str, Any],
    image_root: Path | None,
    comic_id: str,
    page_file: str,
    dataset_name: str,
) -> Path | None:
    original = Path(str(ocr_page.get("path") or ""))
    if original.exists():
        return original
    if image_root is None:
        return None
    root = image_root.expanduser().resolve()
    candidates = [
        root / comic_id / dataset_name / page_file,
        root / comic_id / page_file,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "comic_id": payload["comic_id"],
        "page_file": payload["page_file"],
        "ocr_block_count": payload["ocr_block_count"],
        "group_count": payload["group_count"],
        "magi_context_used": payload["magi_context_used"],
        "group_texts": [group["text"] for group in payload["groups"]],
    }


if __name__ == "__main__":
    main()
