from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from features.magi_extractor import MagiPageExtractor, save_magi_results
from reports.box_visualization import sanitize_filename, visual_comic_dir, visual_page_name
from utils.images import is_supported_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Magi inspection pass over selected comic pages."
    )
    parser.add_argument("--input", required=True, help="Image file, comic dir, or by_comic root.")
    parser.add_argument("--output-dir", required=True, help="Directory for debug output.")
    parser.add_argument(
        "--visual-output-dir",
        default=None,
        help="Optional directory for page-level box overlays. Defaults to --output-dir.",
    )
    parser.add_argument("--limit", type=int, default=2, help="Maximum pages for direct directory mode.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, etc.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype. CPU defaults to float32.",
    )
    parser.add_argument(
        "--task",
        default="both",
        choices=["detections", "ocr", "both"],
        help="Magi task to run. Detections excludes OCR and is faster.",
    )
    parser.add_argument("--no-ocr", action="store_true", help="Alias for --task detections.")
    parser.add_argument(
        "--selection",
        default="first",
        choices=["first", "middle", "last", "random"],
        help="How to pick pages from each dataset directory.",
    )
    parser.add_argument(
        "--sample-one-per-comic",
        action="store_true",
        help="Treat --input as datasets/by_comic and select one page per comic.",
    )
    parser.add_argument(
        "--all-pages-per-comic",
        action="store_true",
        help="Treat --input as datasets/by_comic and inspect every page in each selected comic.",
    )
    parser.add_argument(
        "--max-comics",
        type=int,
        default=0,
        help="Maximum comics to inspect in --sample-one-per-comic mode. 0 means all.",
    )
    parser.add_argument(
        "--comic-id",
        action="append",
        default=[],
        help="Only inspect this comic id. Can be repeated.",
    )
    parser.add_argument(
        "--dataset-name",
        default="test_1_clean",
        help="Dataset folder to use with --sample-one-per-comic.",
    )
    parser.add_argument(
        "--cache-dir",
        default="outputs/cache/magi",
        help="Cache directory for Magi outputs keyed by image hash and task.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing cache and recompute Magi outputs.",
    )
    parser.add_argument(
        "--no-panel-crops",
        action="store_true",
        help="Only save full-page box overlays, not panel crop folders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Seed for random page selection.",
    )
    return parser.parse_args()


def discover_images(args: argparse.Namespace) -> list[tuple[str, Path]]:
    root = Path(args.input).expanduser().resolve()
    comic_ids = set(args.comic_id or [])
    if args.all_pages_per_comic:
        return discover_pages_per_comic(
            root=root,
            dataset_name=args.dataset_name,
            max_comics=args.max_comics,
            comic_ids=comic_ids,
        )
    if args.sample_one_per_comic:
        return discover_one_per_comic(
            root=root,
            dataset_name=args.dataset_name,
            selection=args.selection,
            seed=args.seed,
            max_comics=args.max_comics,
            comic_ids=comic_ids,
        )
    if root.is_file():
        images = [root] if is_supported_image(root) else []
    elif root.is_dir():
        images = sorted(
            (path for path in root.iterdir() if is_supported_image(path)),
            key=lambda path: path.name.lower(),
        )[: max(1, args.limit)]
    else:
        raise FileNotFoundError(f"Input path does not exist: {root}")
    return [(root.parent.name, image) for image in images]


def discover_one_per_comic(
    root: Path,
    dataset_name: str,
    selection: str,
    seed: int,
    max_comics: int = 0,
    comic_ids: set[str] | None = None,
) -> list[tuple[str, Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    selected: list[tuple[str, Path]] = []
    for comic_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if comic_ids and comic_dir.name not in comic_ids:
            continue
        if max_comics > 0 and len(selected) >= max_comics:
            break
        dataset_dir = comic_dir / dataset_name
        if not dataset_dir.exists():
            continue
        images = sorted(
            (path for path in dataset_dir.iterdir() if is_supported_image(path)),
            key=lambda path: path.name.lower(),
        )
        if not images:
            continue
        selected.append((comic_dir.name, select_image(images, selection, seed, comic_dir.name)))
    return selected


def discover_pages_per_comic(
    root: Path,
    dataset_name: str,
    max_comics: int = 0,
    comic_ids: set[str] | None = None,
) -> list[tuple[str, Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    selected: list[tuple[str, Path]] = []
    copied_comics = 0
    for comic_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if comic_ids and comic_dir.name not in comic_ids:
            continue
        if max_comics > 0 and copied_comics >= max_comics:
            break
        dataset_dir = comic_dir / dataset_name
        if not dataset_dir.exists():
            continue
        images = sorted(
            (path for path in dataset_dir.iterdir() if is_supported_image(path)),
            key=lambda path: path.name.lower(),
        )
        if not images:
            continue
        selected.extend((comic_dir.name, image) for image in images)
        copied_comics += 1
    return selected


def select_image(images: list[Path], selection: str, seed: int, salt: str) -> Path:
    if selection == "first":
        return images[0]
    if selection == "last":
        return images[-1]
    if selection == "middle":
        return images[len(images) // 2]
    if selection == "random":
        import random

        rng = random.Random(f"{seed}:{salt}")
        return rng.choice(images)
    raise ValueError(f"Unsupported selection: {selection}")


def summarise_result(result: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    detections = result.get("detections", {})
    summary = {
        "comic_id": metrics["comic_id"],
        "path": result["path"],
        "image_sha256": metrics["image_sha256"],
        "task": metrics["task"],
        "cache_hit": metrics["cache_hit"],
        "elapsed_seconds": metrics["elapsed_seconds"],
        "detection_keys": sorted(detections.keys()) if isinstance(detections, dict) else [],
    }
    if isinstance(detections, dict):
        for key, value in detections.items():
            if isinstance(value, list):
                summary[f"{key}_count"] = len(value)
    ocr = result.get("ocr")
    if isinstance(ocr, dict):
        summary["ocr_text_count"] = len(ocr.get("ocr_texts", []))
    elif isinstance(ocr, list):
        summary["ocr_count"] = len(ocr)
    elif ocr is not None:
        summary["ocr_type"] = type(ocr).__name__
    return summary


def run_or_load(
    extractor: MagiPageExtractor,
    image_path: Path,
    comic_id: str,
    task: str,
    cache_dir: Path,
    force: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_sha = file_sha256(image_path)
    cache_path = cache_dir / f"{image_sha}_{task}.json"
    metrics = {
        "comic_id": comic_id,
        "image_path": str(image_path),
        "image_sha256": image_sha,
        "task": task,
        "cache_path": str(cache_path),
        "cache_hit": False,
        "elapsed_seconds": 0.0,
    }
    if cache_path.exists() and not force:
        start = time.perf_counter()
        result = json.loads(cache_path.read_text(encoding="utf-8"))
        metrics["cache_hit"] = True
        metrics["elapsed_seconds"] = time.perf_counter() - start
        return result, metrics

    start = time.perf_counter()
    if task == "detections":
        result = extractor.predict_detections([image_path])[0]
    elif task == "ocr":
        result = extractor.predict_ocr([image_path])[0]
    elif task == "both":
        result = extractor.predict([image_path], run_ocr=True)[0]
    else:
        raise ValueError(f"Unsupported task: {task}")
    elapsed = time.perf_counter() - start
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics["elapsed_seconds"] = elapsed
    return result, metrics


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.no_ocr:
        args.task = "detections"

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_output_dir = (
        Path(args.visual_output_dir).expanduser().resolve()
        if args.visual_output_dir
        else output_dir
    )
    visual_output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    selected_images = discover_images(args)
    if not selected_images:
        raise RuntimeError(f"No supported images found in {args.input}")

    extractor = MagiPageExtractor(device=args.device, dtype=args.dtype)
    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    wall_start = time.perf_counter()

    for index, (comic_id, image_path) in enumerate(selected_images, 1):
        print(f"[{index}/{len(selected_images)}] {comic_id}: {image_path.name}")
        result, item_metrics = run_or_load(
            extractor=extractor,
            image_path=image_path,
            comic_id=comic_id,
            task=args.task,
            cache_dir=cache_dir,
            force=args.force,
        )
        results.append(result)
        metrics.append(item_metrics)

        stem = f"{index:03d}_{sanitize_filename(comic_id)}_{sanitize_filename(image_path.stem)}"
        if "detections" in result:
            extractor.visualise(
                image_path=image_path,
                result=result,
                output_path=visual_comic_dir(visual_output_dir, comic_id)
                / visual_page_name(image_path.name, "magi_boxes"),
            )
            if args.no_panel_crops:
                panel_count = 0
            else:
                panel_count = extractor.crop_panels(
                    image_path=image_path,
                    result=result,
                    output_dir=output_dir / stem,
                )
        else:
            panel_count = 0
        summary = summarise_result(result, item_metrics)
        summary["panel_crops"] = panel_count
        summaries.append(summary)

    aggregate = {
        "input": str(Path(args.input).expanduser().resolve()),
        "task": args.task,
        "selection": args.selection,
        "dataset_name": args.dataset_name,
        "comic_ids": args.comic_id,
        "sample_one_per_comic": args.sample_one_per_comic,
        "all_pages_per_comic": args.all_pages_per_comic,
        "visual_output_dir": str(visual_output_dir),
        "panel_crops_enabled": not args.no_panel_crops,
        "page_count": len(results),
        "cache_hits": sum(1 for item in metrics if item["cache_hit"]),
        "cache_misses": sum(1 for item in metrics if not item["cache_hit"]),
        "wall_elapsed_seconds": time.perf_counter() - wall_start,
        "magi_elapsed_seconds": sum(item["elapsed_seconds"] for item in metrics),
        "pages": summaries,
        "metrics": metrics,
    }

    save_magi_results(results, output_dir / "magi_results.json")
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"Wrote Magi debug output: {output_dir}")


if __name__ == "__main__":
    main()
