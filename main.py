from __future__ import annotations

import argparse
from pathlib import Path

from analyzers.basic import BasicDatasetAnalyzer
from core.pipeline import ComicNarrativePipeline
from reports.export_ordered_pages import export_ordered_pages
from reports.json_report import JsonReportWriter
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct the narrative order of unordered comic page images."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Image file or directory containing unordered comic pages.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/narrative_order.json",
        help="Path for the structured JSON output.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Optional JSON config file merged over config/default.json.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Store embedding dimensions instead of full embedding vectors in JSON.",
    )
    parser.add_argument(
        "--export-ordered-dir",
        default=None,
        help="Optional directory where ordered page copies will be exported.",
    )
    parser.add_argument(
        "--export-order-source",
        choices=["model", "input", "detected", "pages"],
        default="model",
        help=(
            "Order source for exported pages. 'model' uses the predicted narrative "
            "order; 'input' uses raw load order. Aliases: detected=model, pages=input."
        ),
    )
    parser.add_argument(
        "--export-name-mode",
        choices=["numbered", "prefix-original"],
        default="numbered",
        help="Export page filenames as 001.jpg or 001_original.jpg.",
    )
    parser.add_argument(
        "--overwrite-export",
        action="store_true",
        help="Replace existing files in --export-ordered-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    include_embeddings = not args.no_embeddings and config.get("output", {}).get(
        "include_embeddings", True
    )
    pretty = config.get("output", {}).get("pretty", True)

    pipeline = ComicNarrativePipeline(
        config=config,
        analyzers=[BasicDatasetAnalyzer()],
    )
    result = pipeline.run(args.input)
    JsonReportWriter(
        include_embeddings=include_embeddings,
        pretty=pretty,
    ).write(result, args.output)

    output_path = Path(args.output).expanduser().resolve()
    if args.export_ordered_dir:
        exported_pages = export_ordered_pages(
            input_json=output_path,
            output_dir=Path(args.export_ordered_dir),
            overwrite=args.overwrite_export,
            order_source=args.export_order_source,
            name_mode=args.export_name_mode,
        )
        print(
            "Exported ordered comic: "
            f"{Path(args.export_ordered_dir).expanduser().resolve()} "
            f"({len(exported_pages)} pages)"
        )

    print(f"Processed {len(result.pages)} pages")
    print(f"Generated {len(result.relations)} directed relations")
    print(f"Model order: {result.order}")
    print(f"Wrote JSON report: {output_path}")


if __name__ == "__main__":
    main()
