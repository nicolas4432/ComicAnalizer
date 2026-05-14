from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from reports.box_visualization import (
    draw_magi_overlay,
    draw_paddle_ocr_readable_overlay_from_dict,
    sanitize_filename,
    visual_comic_dir,
    visual_page_name,
)
from tools.analyze_magi_results import (
    build_analysis_report,
    load_magi_bundle,
    normalize_pages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Magi/Colab outputs into the standard outputs/runs/<run_name> layout."
    )
    parser.add_argument("--magi-input", required=True, help="Magi ZIP or output directory.")
    parser.add_argument(
        "--ocr-comparison",
        default=None,
        help="Optional paddle_magi_ocr_comparison.json to copy into the run.",
    )
    parser.add_argument("--run-name", required=True, help="Stable run folder name.")
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument(
        "--copy-json-only",
        action="store_true",
        help="Keep only JSON files from Magi output. Recommended for local archives.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help=(
            "Optional local by_comic root used to regenerate Magi visual overlays "
            "when Colab paths are not available locally."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_root).expanduser().resolve() / args.run_name
    magi_dir = run_dir / "magi"
    analysis_dir = run_dir / "analysis"
    visuals_dir = run_dir / "visuals"
    magi_visuals_dir = visuals_dir / "magi_boxes"
    ocr_visuals_dir = visuals_dir / "ocr_boxes"
    magi_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_magi_bundle(Path(args.magi_input).expanduser())
    write_magi_json_bundle(bundle, magi_dir)

    pages = normalize_pages(bundle)
    analysis = build_analysis_report(
        pages=pages,
        bundle=bundle,
        include_regions=False,
        top_n=25,
    )
    analysis_path = analysis_dir / "magi_analysis_report.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ocr_path = None
    if args.ocr_comparison:
        source = Path(args.ocr_comparison).expanduser().resolve()
        if source.exists():
            ocr_path = analysis_dir / "paddle_magi_ocr_comparison.json"
            if source != ocr_path.resolve():
                shutil.copy2(source, ocr_path)
            write_ocr_visuals_from_report(source, ocr_visuals_dir)

    magi_visual_count = copy_magi_box_visuals(
        Path(args.magi_input).expanduser(),
        magi_visuals_dir,
        bundle,
    )
    if magi_visual_count == 0:
        magi_visual_count = write_magi_visuals_from_pages(
            pages,
            magi_visuals_dir,
            image_root=Path(args.image_root).expanduser().resolve() if args.image_root else None,
        )
    ocr_visual_count = count_files(ocr_visuals_dir, "*.jpg")

    manifest = {
        "schema_version": "standard_magi_run.v1",
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "source": {
            "magi_input": str(Path(args.magi_input).expanduser().resolve()),
            "source_type": bundle.get("source_type"),
            "ocr_comparison": str(Path(args.ocr_comparison).expanduser().resolve())
            if args.ocr_comparison
            else None,
        },
        "outputs": {
            "magi_results": str(magi_dir / "magi_results.json"),
            "metrics": str(magi_dir / "metrics.json"),
            "summary": str(magi_dir / "summary.json"),
            "magi_analysis_report": str(analysis_path),
            "paddle_magi_ocr_comparison": str(ocr_path) if ocr_path else None,
            "magi_box_visuals": str(magi_visuals_dir) if magi_visual_count else None,
            "ocr_box_visuals": str(ocr_visuals_dir) if ocr_visual_count else None,
        },
        "visuals": {
            "magi_box_count": magi_visual_count,
            "ocr_box_count": ocr_visual_count,
        },
        "summary": analysis["summary"],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(manifest["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote standard Magi run: {run_dir}")


def write_magi_json_bundle(bundle: dict[str, Any], magi_dir: Path) -> None:
    write_json(magi_dir / "magi_results.json", bundle.get("magi_results") or [])
    write_json(magi_dir / "metrics.json", bundle.get("metrics") or {})
    write_json(magi_dir / "summary.json", bundle.get("summary") or [])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_magi_box_visuals(source: Path, target_dir: Path, bundle: dict[str, Any]) -> int:
    source = source.expanduser().resolve()
    copied = 0
    reset_directory(target_dir)
    visual_map = build_magi_visual_map(bundle)
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for name in archive.namelist():
                if not name.endswith("_boxes.jpg"):
                    continue
                target = magi_visual_target(target_dir, Path(name).name, visual_map)
                with archive.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                copied += 1
    elif source.is_dir():
        for path in source.rglob("*_boxes.jpg"):
            target = magi_visual_target(target_dir, path.name, visual_map)
            shutil.copy2(path, target)
            copied += 1
    return copied


def write_ocr_visuals_from_report(report_path: Path, target_dir: Path) -> int:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reset_directory(target_dir)
    page_lookup = build_ocr_page_lookup(report)
    count = 0
    for result in report.get("ocr_results") or []:
        if result.get("error") or not result.get("blocks"):
            continue
        source_path = Path(result.get("path", ""))
        if not source_path.exists():
            continue
        page_info = page_lookup.get(str(source_path)) or page_lookup.get(str(source_path.resolve()))
        comic_id = page_info.get("comic_id") if page_info else source_path.parent.parent.name
        file_name = page_info.get("file_name") if page_info else source_path.name
        target = visual_comic_dir(target_dir, comic_id) / visual_page_name(file_name, "ocr_boxes")
        draw_paddle_ocr_readable_overlay_from_dict(result, target)
        count += 1
    return count


def write_magi_visuals_from_pages(
    pages: list[Any],
    target_dir: Path,
    image_root: Path | None,
) -> int:
    reset_directory(target_dir)
    count = 0
    for page in pages:
        image_path = resolve_page_image(page, image_root)
        if image_path is None:
            continue
        target = visual_comic_dir(target_dir, page.comic_id) / visual_page_name(
            page.file_name,
            "magi_boxes",
        )
        draw_magi_overlay(page=page, image_path=image_path, output_path=target)
        count += 1
    return count


def resolve_page_image(page: Any, image_root: Path | None) -> Path | None:
    raw_path = str(page.path or "")
    direct = Path(raw_path)
    if direct.exists():
        return direct.resolve()
    if image_root is None:
        return None

    normalized = raw_path.replace("\\", "/")
    if "/by_comic/" in normalized:
        suffix = normalized.split("/by_comic/", 1)[1]
        candidates = [
            image_root / Path(suffix),
            image_root / "by_comic" / Path(suffix),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

    if page.comic_id and page.file_name:
        matches = list(image_root.rglob(f"{page.comic_id}/**/{page.file_name}"))
        if matches:
            return matches[0].resolve()
    return None


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.rglob(pattern)) if path.exists() else 0


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_image_filename(name: str) -> str:
    path = Path(name)
    suffix = path.suffix or ".jpg"
    return f"{sanitize_filename(path.stem)}{suffix}"


def build_magi_visual_map(bundle: dict[str, Any]) -> dict[int, dict[str, str]]:
    metrics = bundle.get("metrics") or {}
    pages = metrics.get("pages") or metrics.get("metrics") or []
    visual_map: dict[int, dict[str, str]] = {}
    for index, page in enumerate(pages, 1):
        comic_id = page.get("comic_id") or "unknown"
        file_name = Path(page.get("path") or page.get("image_path") or f"{index:03d}.jpg").name
        visual_map[index] = {
            "comic_id": comic_id,
            "file_name": file_name,
        }
    return visual_map


def magi_visual_target(
    target_dir: Path,
    source_name: str,
    visual_map: dict[int, dict[str, str]],
) -> Path:
    page_info = visual_map.get(parse_visual_index(source_name))
    if page_info:
        return visual_comic_dir(target_dir, page_info["comic_id"]) / visual_page_name(
            page_info["file_name"],
            "magi_boxes",
        )
    nested = nested_visual_target(target_dir, source_name)
    if nested:
        return nested
    source_path = Path(source_name)
    return visual_comic_dir(target_dir, "unknown") / safe_image_filename(source_path.name)


def nested_visual_target(target_dir: Path, source_name: str) -> Path | None:
    source_path = Path(source_name)
    parts = source_path.parts
    if "magi_boxes" in parts:
        index = parts.index("magi_boxes")
        if len(parts) > index + 2:
            comic_id = parts[index + 1]
            return visual_comic_dir(target_dir, comic_id) / safe_image_filename(source_path.name)
    if len(parts) >= 2 and source_path.parent.name not in {"", "."}:
        parent = source_path.parent.name
        if parent not in {"outputs", "visuals", "magi", "analysis"}:
            return visual_comic_dir(target_dir, parent) / safe_image_filename(source_path.name)
    return None


def parse_visual_index(source_name: str) -> int:
    prefix = Path(source_name).stem.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return 0


def build_ocr_page_lookup(report: dict[str, Any]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for item in report.get("comparisons") or []:
        image_path = item.get("image_path")
        if not image_path:
            continue
        lookup[str(Path(image_path))] = {
            "comic_id": item.get("comic_id") or "unknown",
            "file_name": item.get("file_name") or Path(image_path).name,
        }
    return lookup


if __name__ == "__main__":
    main()
