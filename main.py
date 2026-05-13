from __future__ import annotations

import argparse
from pathlib import Path

from analyzers.basic import BasicDatasetAnalyzer
from core.pipeline import ComicNarrativePipeline
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
    print(f"Processed {len(result.pages)} pages")
    print(f"Generated {len(result.relations)} directed relations")
    print(f"Initial order: {result.order}")
    print(f"Wrote JSON report: {output_path}")


if __name__ == "__main__":
    main()

