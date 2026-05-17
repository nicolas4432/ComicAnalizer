from __future__ import annotations

import argparse
import json
from pathlib import Path

from features.ocr_evidence import export_ocr_evidence
from tools.analyze_magi_results import load_magi_bundle, normalize_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export OCR evidence crops and metadata for calibration/training."
    )
    parser.add_argument(
        "--ocr-report",
        required=True,
        help="paddle_magi_ocr_comparison.json produced by compare_magi_paddleocr.",
    )
    parser.add_argument(
        "--magi-input",
        default=None,
        help="Optional Magi output folder/ZIP for panel/text/character context.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help=(
            "Optional local by_comic image root. Use this when the OCR report was "
            "generated in Colab and its /content image paths do not exist locally."
        ),
    )
    parser.add_argument("--dataset-name", default="test_1_clean")
    parser.add_argument(
        "--output-dir",
        default="annotations/ocr_evidence",
        help="Directory where evidence.jsonl and assets will be written.",
    )
    parser.add_argument(
        "--include-empty-blocks",
        action="store_true",
        help="Keep OCR detections with empty text. Useful for detector debugging.",
    )
    parser.add_argument(
        "--context-padding",
        type=int,
        default=56,
        help="Pixels around each OCR block for context crops.",
    )
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        help="Optional maximum OCR pages to export.",
    )
    parser.add_argument(
        "--asset-policy",
        choices=["all", "priority", "none"],
        default="priority",
        help=(
            "Which OCR blocks should get crop assets. JSONL evidence is still "
            "written for every block. 'priority' keeps review-focused crops."
        ),
    )
    parser.add_argument(
        "--max-asset-blocks",
        type=int,
        default=500,
        help="Maximum OCR blocks that receive crop assets. Use -1 for no limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ocr_report_path = Path(args.ocr_report).expanduser().resolve()
    ocr_report = json.loads(ocr_report_path.read_text(encoding="utf-8"))

    magi_pages = []
    if args.magi_input:
        bundle = load_magi_bundle(Path(args.magi_input).expanduser())
        magi_pages = normalize_pages(bundle)

    index = export_ocr_evidence(
        ocr_report=ocr_report,
        output_dir=Path(args.output_dir),
        magi_pages=magi_pages,
        image_root=Path(args.image_root) if args.image_root else None,
        dataset_name=args.dataset_name,
        include_empty_blocks=args.include_empty_blocks,
        context_padding=args.context_padding,
        limit_pages=args.limit_pages,
        asset_policy=args.asset_policy,
        max_asset_blocks=None if args.max_asset_blocks < 0 else args.max_asset_blocks,
    )
    print(json.dumps(index["stats"], indent=2, ensure_ascii=False))
    print(f"Wrote OCR evidence: {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
