from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from features.magi_postprocess import build_quality_report
from features.ocr_paddle import (
    PaddleOCRComplementaryExtractor,
    compare_magi_with_paddle,
)
from reports.box_visualization import (
    draw_paddle_ocr_readable_overlay_from_dict,
    visual_comic_dir,
    visual_page_name,
)
from tools.analyze_magi_results import load_magi_bundle, normalize_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR on selected pages and compare text regions with Magi."
    )
    parser.add_argument(
        "--magi-input",
        required=True,
        help="Colab Magi ZIP, output directory, or magi_results.json.",
    )
    parser.add_argument(
        "--image-root",
        required=True,
        help="Root containing by_comic/<comic>/<dataset>/<image> or the by_comic folder itself.",
    )
    parser.add_argument("--dataset-name", default="test_1_clean")
    parser.add_argument("--output", default="outputs/analysis/paddle_magi_ocr_comparison.json")
    parser.add_argument(
        "--visual-output-dir",
        default=None,
        help="Optional directory for PaddleOCR text box overlays.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum pages to OCR. Use 0 to process every selected page.",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--comic-id",
        action="append",
        default=[],
        help="Only OCR pages from this comic id. Can be repeated.",
    )
    parser.add_argument(
        "--selection",
        default="random",
        choices=["random", "first", "suspicious"],
        help="Which pages to OCR.",
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument(
        "--use-angle-cls",
        action="store_true",
        help="Enable text-line orientation classification. Slower, useful for rotated text.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.15)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write a partial JSON report every N processed pages. 0 disables checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_magi_bundle(Path(args.magi_input).expanduser())
    pages = normalize_pages(bundle)
    if args.comic_id:
        comic_ids = set(args.comic_id)
        pages = [page for page in pages if page.comic_id in comic_ids]
    selected_pages = select_pages(pages, args.selection, args.limit, args.seed)
    extractor = PaddleOCRComplementaryExtractor(
        lang=args.lang,
        use_angle_cls=args.use_angle_cls,
    )
    try:
        extractor.load()
    except Exception as exc:  # noqa: BLE001 - CLI should fail loudly here.
        raise RuntimeError(
            "PaddleOCR initialization failed before processing pages. "
            "Check that paddlepaddle and paddleocr are installed correctly."
        ) from exc
    visual_output_dir = (
        Path(args.visual_output_dir).expanduser().resolve()
        if args.visual_output_dir
        else None
    )
    if visual_output_dir:
        visual_output_dir.mkdir(parents=True, exist_ok=True)

    comparisons: list[dict[str, Any]] = []
    ocr_results: list[dict[str, Any]] = []
    missing_images: list[dict[str, str]] = []
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_start = time.perf_counter()

    print(
        f"Selected {len(selected_pages)} pages for PaddleOCR "
        f"(selection={args.selection}, limit={args.limit}, lang={args.lang})",
        flush=True,
    )

    for index, page in enumerate(selected_pages, 1):
        image_path = resolve_image_path(
            image_root=Path(args.image_root).expanduser(),
            page=page,
            dataset_name=args.dataset_name,
        )
        if image_path is None:
            missing_images.append(
                {
                    "page_id": page.page_id,
                    "comic_id": page.comic_id or "",
                    "file_name": page.file_name,
                }
            )
            continue
        print(f"[{index}/{len(selected_pages)}] OCR {page.comic_id}/{page.file_name}", flush=True)
        page_start = time.perf_counter()
        ocr_result = extractor.extract_page(image_path)
        page_elapsed = time.perf_counter() - page_start
        comparison = compare_magi_with_paddle(
            page=page,
            ocr_result=ocr_result,
            iou_threshold=args.iou_threshold,
        )
        visual_path = None
        if visual_output_dir and not ocr_result.error:
            visual_path = visual_comic_dir(visual_output_dir, page.comic_id) / visual_page_name(
                page.file_name,
                "ocr_boxes",
            )
            draw_paddle_ocr_readable_overlay_from_dict(ocr_result.to_dict(), visual_path)
        comparisons.append(
            comparison.to_dict()
            | {
                "image_path": str(image_path),
                "paddle_text_preview": ocr_result.text[:500],
                "visual_path": str(visual_path) if visual_path else None,
            }
        )
        ocr_results.append(ocr_result.to_dict())
        processed = len(comparisons)
        avg_elapsed = (time.perf_counter() - run_start) / max(1, processed)
        remaining = max(0, len(selected_pages) - index)
        eta_seconds = avg_elapsed * remaining
        print(
            "  -> "
            f"seconds={page_elapsed:.2f}, blocks={len(ocr_result.blocks)}, "
            f"error={ocr_result.error or 'none'}, eta_min={eta_seconds / 60:.1f}",
            flush=True,
        )
        if args.checkpoint_every > 0 and processed % args.checkpoint_every == 0:
            write_report(
                output.with_suffix(".partial.json"),
                args=args,
                visual_output_dir=visual_output_dir,
                comparisons=comparisons,
                missing_images=missing_images,
                ocr_results=ocr_results,
            )

    report = build_report(
        args=args,
        visual_output_dir=visual_output_dir,
        comparisons=comparisons,
        missing_images=missing_images,
        ocr_results=ocr_results,
    )
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote PaddleOCR/Magi comparison: {output}")


def build_report(
    args: argparse.Namespace,
    visual_output_dir: Path | None,
    comparisons: list[dict[str, Any]],
    missing_images: list[dict[str, str]],
    ocr_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "paddle_magi_ocr_comparison.v1",
        "magi_input": str(Path(args.magi_input).expanduser().resolve()),
        "image_root": str(Path(args.image_root).expanduser().resolve()),
        "dataset_name": args.dataset_name,
        "comic_ids": args.comic_id,
        "selection": args.selection,
        "limit": args.limit,
        "seed": args.seed,
        "lang": args.lang,
        "use_angle_cls": args.use_angle_cls,
        "visual_output_dir": str(visual_output_dir) if visual_output_dir else None,
        "summary": summarize_comparisons(comparisons, missing_images),
        "missing_images": missing_images,
        "comparisons": comparisons,
        "ocr_results": ocr_results,
    }


def write_report(
    output: Path,
    args: argparse.Namespace,
    visual_output_dir: Path | None,
    comparisons: list[dict[str, Any]],
    missing_images: list[dict[str, str]],
    ocr_results: list[dict[str, Any]],
) -> None:
    report = build_report(
        args=args,
        visual_output_dir=visual_output_dir,
        comparisons=comparisons,
        missing_images=missing_images,
        ocr_results=ocr_results,
    )
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def select_pages(
    pages: list[Any],
    selection: str,
    limit: int,
    seed: int,
) -> list[Any]:
    if limit <= 0:
        limit = len(pages)
    if selection == "first":
        return pages[:limit]
    if selection == "suspicious":
        quality = build_quality_report(pages)
        suspicious_ids = {
            item["page_id"] for item in quality["pages"] if item.get("suspicious")
        }
        selected = [page for page in pages if page.page_id in suspicious_ids]
        return selected[:limit]
    rng = random.Random(seed)
    selected = list(pages)
    rng.shuffle(selected)
    return selected[:limit]


def resolve_image_path(
    image_root: Path,
    page: Any,
    dataset_name: str,
) -> Path | None:
    image_root = image_root.resolve()
    candidates: list[Path] = []
    if page.comic_id:
        candidates.append(image_root / page.comic_id / dataset_name / page.file_name)
        candidates.append(image_root / "by_comic" / page.comic_id / dataset_name / page.file_name)
    if page.path:
        candidates.append(Path(page.path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if page.comic_id:
        matches = list(image_root.rglob(f"{page.comic_id}/**/{page.file_name}"))
    else:
        matches = list(image_root.rglob(page.file_name))
    return matches[0].resolve() if matches else None


def summarize_comparisons(
    comparisons: list[dict[str, Any]],
    missing_images: list[dict[str, str]],
) -> dict[str, Any]:
    if not comparisons:
        return {
            "page_count": 0,
            "missing_image_count": len(missing_images),
        }
    elapsed = [item["paddle_elapsed_seconds"] for item in comparisons]
    paddle_blocks = [item["paddle_text_blocks"] for item in comparisons]
    magi_regions = [item["magi_text_regions"] for item in comparisons]
    matched = [item["matched_regions"] for item in comparisons]
    return {
        "page_count": len(comparisons),
        "missing_image_count": len(missing_images),
        "avg_paddle_seconds": sum(elapsed) / len(elapsed),
        "max_paddle_seconds": max(elapsed),
        "avg_paddle_text_blocks": sum(paddle_blocks) / len(paddle_blocks),
        "avg_magi_text_regions": sum(magi_regions) / len(magi_regions),
        "avg_matched_regions": sum(matched) / len(matched),
        "pages_with_paddle_error": sum(1 for item in comparisons if item["paddle_error"]),
        "pages_where_paddle_found_more_text": sum(
            1 for item in comparisons if item["paddle_text_blocks"] > item["magi_text_regions"]
        ),
        "pages_where_magi_found_more_regions": sum(
            1 for item in comparisons if item["magi_text_regions"] > item["paddle_text_blocks"]
        ),
    }


if __name__ == "__main__":
    main()
