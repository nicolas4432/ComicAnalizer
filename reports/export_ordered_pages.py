from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def _safe_name(name: str) -> str:
    unsafe_chars = '<>:"/\\|?*'
    cleaned = "".join("_" if char in unsafe_chars else char for char in name)
    return cleaned.strip().strip(".") or "page"


def _load_pipeline_output(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data.get("pages"), list):
        raise ValueError("Input JSON must contain a 'pages' list.")
    if not isinstance(data.get("order"), list):
        raise ValueError("Input JSON must contain an 'order' list.")
    return data


def _resolve_ordered_page_ids(data: dict[str, Any], order_source: str) -> list[str]:
    if order_source in {"model", "detected"}:
        if isinstance(data.get("ordered_pages"), list):
            return [page["id"] for page in data["ordered_pages"]]
        return data["order"]
    if order_source in {"input", "pages"}:
        return [page["id"] for page in data["pages"]]
    raise ValueError(f"Unsupported order source: {order_source}")


def export_ordered_pages(
    input_json: Path,
    output_dir: Path,
    overwrite: bool = False,
    order_source: str = "model",
    name_mode: str = "prefix-original",
) -> list[dict[str, Any]]:
    data = _load_pipeline_output(input_json)
    pages_by_id = {page["id"]: page for page in data["pages"]}
    ordered_page_ids = _resolve_ordered_page_ids(data, order_source)

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()

    manifest: list[dict[str, Any]] = []

    base_timestamp = int(time.time())
    for position, page_id in enumerate(ordered_page_ids, 1):
        page = pages_by_id.get(page_id)
        if page is None:
            raise ValueError(f"Order references an unknown page id: {page_id}")

        source_path = Path(page["path"]).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Ordered page source does not exist: {source_path}")

        if name_mode == "numbered":
            target_name = f"{position:03d}{source_path.suffix.lower()}"
        else:
            target_name = f"{position:03d}_{_safe_name(source_path.name)}"

        target_path = output_dir / target_name
        if target_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {target_path}. "
                "Use --overwrite to replace exported pages."
            )

        shutil.copyfile(source_path, target_path)
        ordered_timestamp = base_timestamp + position
        os.utime(target_path, (ordered_timestamp, ordered_timestamp))
        manifest.append(
            {
                "position": position,
                "page_id": page_id,
                "source_path": str(source_path),
                "output_file": target_name,
                "sha256": page.get("sha256"),
            }
        )

    skipped_inputs = data.get("anomalies", {}).get("skipped_inputs", [])
    manifest_data = {
        "input_json": str(input_json.resolve()),
        "output_dir": str(output_dir.resolve()),
        "order_source": order_source,
        "name_mode": name_mode,
        "page_count": len(manifest),
        "skipped_input_count": len(skipped_inputs),
        "skipped_inputs": skipped_inputs,
        "pages": manifest,
    }

    manifest_path = output_dir / "order_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}. "
            "Use --overwrite to replace it."
        )
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ordered comic pages from a pipeline JSON report."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to a pipeline JSON output containing pages and order.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where ordered pages will be copied.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if the export directory already contains results.",
    )
    parser.add_argument(
        "--order-source",
        choices=["model", "input", "detected", "pages"],
        default="model",
        help=(
            "Use model order from JSON 'ordered_pages'/'order', or input load order "
            "from JSON 'pages'. Aliases: detected=model, pages=input."
        ),
    )
    parser.add_argument(
        "--name-mode",
        choices=["prefix-original", "numbered"],
        default="prefix-original",
        help="Export as 001_original.jpg or as clean numbered files like 001.jpg.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_ordered_pages(
        input_json=Path(args.input_json),
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
        order_source=args.order_source,
        name_mode=args.name_mode,
    )
    print(f"Exported {len(manifest)} ordered pages")
    print(f"Output directory: {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
