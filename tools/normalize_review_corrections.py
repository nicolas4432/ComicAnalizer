from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "comic_review_corrections.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize corrections exported by the HTML review report."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="comic_ocr_corrections.json downloaded from the HTML report.",
    )
    parser.add_argument(
        "--output-dir",
        default="annotations/review_corrections",
        help="Destination folder for normalized JSONL and index files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    corrections = payload.get("corrections") or []
    normalized = [normalize_correction(item) for item in corrections]

    jsonl_path = output_dir / "review_corrections.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for item in normalized:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    index = build_index(input_path=input_path, jsonl_path=jsonl_path, corrections=normalized)
    index_path = output_dir / "review_corrections_index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(index["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote normalized corrections: {jsonl_path}")


def normalize_correction(item: dict[str, Any]) -> dict[str, Any]:
    fusion = item.get("fusion") or {}
    alternatives = item.get("alternatives") or []
    corrected_text = str(item.get("corrected_text") or "").strip()
    raw_text = str(item.get("raw_text") or "").strip()
    suggested_text = str(item.get("suggested_text") or "").strip()
    label = str(item.get("label") or "unreviewed")
    return {
        "schema_version": SCHEMA_VERSION,
        "id": item.get("id"),
        "comic_id": item.get("comic_id"),
        "page_file": item.get("file_name"),
        "page_id": item.get("page_id"),
        "target": {
            "id": item.get("target_id"),
            "kind": item.get("target_kind"),
            "display": item.get("display"),
            "box": item.get("box"),
        },
        "texts": {
            "raw": raw_text or None,
            "suggested": suggested_text or None,
            "corrected": corrected_text or None,
        },
        "label": label,
        "training": {
            "is_reviewed": True,
            "is_text_changed": bool(corrected_text and corrected_text != raw_text),
            "is_suggestion_accepted": bool(corrected_text and corrected_text == suggested_text),
            "is_false_positive": label == "false_positive",
            "error_type": label if label != "correct" else None,
        },
        "fusion": {
            "id": fusion.get("id"),
            "source": fusion.get("source"),
            "text": fusion.get("text"),
            "review_flags": fusion.get("review_flags") or [],
            "linked_targets": fusion.get("linked_targets") or [],
        },
        "alternatives": [
            {
                "source": option.get("source"),
                "text": option.get("text"),
                "confidence": option.get("confidence"),
                "target_id": option.get("target_id"),
            }
            for option in alternatives
        ],
        "source_payload": item.get("source_payload") or {},
        "updated_at": item.get("updated_at"),
    }


def build_index(
    input_path: Path,
    jsonl_path: Path,
    corrections: list[dict[str, Any]],
) -> dict[str, Any]:
    labels = Counter(str(item.get("label") or "unknown") for item in corrections)
    kinds = Counter(str((item.get("target") or {}).get("kind") or "unknown") for item in corrections)
    changed = sum(1 for item in corrections if (item.get("training") or {}).get("is_text_changed"))
    return {
        "schema_version": SCHEMA_VERSION,
        "input": str(input_path),
        "files": {
            "jsonl": str(jsonl_path),
        },
        "summary": {
            "correction_count": len(corrections),
            "text_changed_count": changed,
            "label_counts": dict(labels),
            "target_kind_counts": dict(kinds),
        },
    }


if __name__ == "__main__":
    main()
