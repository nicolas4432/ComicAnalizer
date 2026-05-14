from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from features.magi_postprocess import build_quality_report
from features.magi_schema import MagiPageAnalysis, normalize_magi_page


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Magi JSON outputs from a Colab ZIP, folder, or raw JSON files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Colab result ZIP, output directory, magi_results.json, or metrics.json.",
    )
    parser.add_argument(
        "--output",
        default="outputs/magi_analysis_report.json",
        help="Structured analysis report path.",
    )
    parser.add_argument(
        "--include-regions",
        action="store_true",
        help="Include full normalized boxes in the report. This makes JSON larger.",
    )
    parser.add_argument("--top-n", type=int, default=15, help="Top pages per ranking section.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_magi_bundle(Path(args.input).expanduser())
    pages = normalize_pages(bundle)
    report = build_analysis_report(
        pages=pages,
        bundle=bundle,
        include_regions=args.include_regions,
        top_n=args.top_n,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote Magi analysis report: {output}")


def load_magi_bundle(path: Path) -> dict[str, Any]:
    if path.is_file() and path.suffix.lower() == ".zip":
        return load_from_zip(path)
    if path.is_dir():
        return load_from_directory(path)
    if path.is_file():
        return load_from_json_file(path)
    raise FileNotFoundError(f"Input does not exist: {path}")


def load_from_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        magi_name = find_entry(names, "magi_results.json")
        metrics_name = find_entry(names, "metrics.json", required=False)
        summary_name = find_entry(names, "summary.json", required=False)
        if magi_name is None:
            raise RuntimeError(f"No magi_results.json found in {path}")
        return {
            "source_type": "zip",
            "source": str(path.resolve()),
            "magi_results_name": magi_name,
            "metrics_name": metrics_name,
            "summary_name": summary_name,
            "magi_results": json.loads(archive.read(magi_name).decode("utf-8")),
            "metrics": (
                json.loads(archive.read(metrics_name).decode("utf-8"))
                if metrics_name
                else {}
            ),
            "summary": (
                json.loads(archive.read(summary_name).decode("utf-8"))
                if summary_name
                else []
            ),
            "artifact_counts": {
                "entries": len(names),
                "box_images": sum(1 for name in names if name.endswith("_boxes.jpg")),
                "panel_crops": sum(
                    1 for name in names if "/panel_" in name and name.endswith(".jpg")
                ),
            },
        }


def load_from_directory(path: Path) -> dict[str, Any]:
    magi_path = path / "magi_results.json"
    metrics_path = path / "metrics.json"
    summary_path = path / "summary.json"
    if not magi_path.exists():
        candidates = list(path.rglob("magi_results.json"))
        if not candidates:
            raise RuntimeError(f"No magi_results.json found under {path}")
        magi_path = candidates[0]
        metrics_path = magi_path.with_name("metrics.json")
        summary_path = magi_path.with_name("summary.json")
    return {
        "source_type": "directory",
        "source": str(path.resolve()),
        "magi_results_name": str(magi_path),
        "metrics_name": str(metrics_path) if metrics_path.exists() else None,
        "summary_name": str(summary_path) if summary_path.exists() else None,
        "magi_results": json.loads(magi_path.read_text(encoding="utf-8")),
        "metrics": (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists()
            else {}
        ),
        "summary": (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else []
        ),
        "artifact_counts": {
            "entries": sum(1 for _ in path.rglob("*")),
            "box_images": sum(1 for item in path.rglob("*_boxes.jpg")),
            "panel_crops": sum(1 for item in path.rglob("panel_*.jpg")),
        },
    }


def load_from_json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if path.name == "metrics.json":
        magi_path = path.with_name("magi_results.json")
        summary_path = path.with_name("summary.json")
        return {
            "source_type": "metrics_file",
            "source": str(path.resolve()),
            "magi_results_name": str(magi_path) if magi_path.exists() else None,
            "metrics_name": str(path),
            "summary_name": str(summary_path) if summary_path.exists() else None,
            "magi_results": (
                json.loads(magi_path.read_text(encoding="utf-8"))
                if magi_path.exists()
                else []
            ),
            "metrics": payload,
            "summary": (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.exists()
                else []
            ),
            "artifact_counts": {},
        }
    return {
        "source_type": "magi_results_file",
        "source": str(path.resolve()),
        "magi_results_name": str(path),
        "metrics_name": None,
        "summary_name": None,
        "magi_results": payload,
        "metrics": {},
        "summary": [],
        "artifact_counts": {},
    }


def find_entry(names: list[str], suffix: str, required: bool = True) -> str | None:
    matches = [name for name in names if name.endswith(suffix)]
    if matches:
        return sorted(matches, key=len)[0]
    if required:
        raise RuntimeError(f"No entry ending in {suffix} found")
    return None


def normalize_pages(bundle: dict[str, Any]) -> list[MagiPageAnalysis]:
    results = bundle.get("magi_results") or []
    metric_pages = {item.get("path") or item.get("image_path"): item for item in pages_from_metrics(bundle)}
    pages: list[MagiPageAnalysis] = []
    for index, result in enumerate(results):
        metric = metric_pages.get(result.get("path"))
        if metric is None:
            metric = match_metric_by_file(result, metric_pages)
        pages.append(normalize_magi_page(result, metric=metric, index=index))
    return pages


def pages_from_metrics(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = bundle.get("metrics") or {}
    pages = metrics.get("metrics") or metrics.get("pages") or []
    return pages if isinstance(pages, list) else []


def match_metric_by_file(
    result: dict[str, Any],
    metric_pages: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    path = str(result.get("path") or "").replace("\\", "/")
    if not path:
        return None
    suffix_parts = path.split("/")[-4:]
    suffix = "/".join(suffix_parts)
    for metric_path, metric in metric_pages.items():
        normalized = str(metric_path).replace("\\", "/")
        if normalized.endswith(suffix):
            return metric
    return None


def build_analysis_report(
    pages: list[MagiPageAnalysis],
    bundle: dict[str, Any],
    include_regions: bool,
    top_n: int,
) -> dict[str, Any]:
    quality = build_quality_report(pages)
    per_comic = build_per_comic_summary(pages, quality)
    summary = build_global_summary(pages, bundle, quality)
    return {
        "schema_version": "magi_analysis.v1",
        "source": {
            "type": bundle.get("source_type"),
            "path": bundle.get("source"),
            "magi_results": bundle.get("magi_results_name"),
            "metrics": bundle.get("metrics_name"),
            "summary": bundle.get("summary_name"),
            "artifacts": bundle.get("artifact_counts") or {},
        },
        "summary": summary,
        "per_comic": per_comic,
        "quality": quality,
        "rankings": {
            "slowest_pages": ranked_pages(pages, "elapsed_seconds", top_n),
            "most_text_regions": ranked_pages(pages, "text_count", top_n),
            "most_characters": ranked_pages(pages, "character_count", top_n),
            "most_tails": ranked_pages(pages, "tail_count", top_n),
            "single_panel_dense_pages": single_panel_dense_pages(pages),
        },
        "pages": [page.to_dict(include_regions=include_regions) for page in pages],
    }


def build_global_summary(
    pages: list[MagiPageAnalysis],
    bundle: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    times = [page.elapsed_seconds for page in pages if page.elapsed_seconds is not None]
    uncached_times = [
        page.elapsed_seconds
        for page in pages
        if page.elapsed_seconds is not None and not page.cache_hit
    ]
    metrics = bundle.get("metrics") or {}
    return {
        "page_count": len(pages),
        "comic_count": len({page.comic_id for page in pages if page.comic_id}),
        "cache_hits": sum(1 for page in pages if page.cache_hit),
        "cache_misses": sum(1 for page in pages if page.cache_hit is False),
        "wall_elapsed_seconds": metrics.get("wall_elapsed_seconds"),
        "magi_elapsed_seconds": metrics.get("magi_elapsed_seconds"),
        "avg_page_seconds": mean_or_none(times),
        "avg_uncached_page_seconds": mean_or_none(uncached_times),
        "median_uncached_page_seconds": median_or_none(uncached_times),
        "total_panels": sum(page.panel_count for page in pages),
        "total_text_regions": sum(page.text_count for page in pages),
        "total_characters": sum(page.character_count for page in pages),
        "total_tails": sum(page.tail_count for page in pages),
        "suspicious_page_count": quality["suspicious_page_count"],
        "flag_counts": quality["flag_counts"],
    }


def build_per_comic_summary(
    pages: list[MagiPageAnalysis],
    quality: dict[str, Any],
) -> list[dict[str, Any]]:
    flags_by_page = {item["page_id"]: item for item in quality["pages"]}
    grouped: dict[str, list[MagiPageAnalysis]] = defaultdict(list)
    for page in pages:
        grouped[page.comic_id or "unknown"].append(page)

    summaries: list[dict[str, Any]] = []
    for comic_id, items in grouped.items():
        times = [
            page.elapsed_seconds
            for page in items
            if page.elapsed_seconds is not None and not page.cache_hit
        ]
        summaries.append(
            {
                "comic_id": comic_id,
                "page_count": len(items),
                "cache_hits": sum(1 for page in items if page.cache_hit),
                "avg_uncached_seconds": mean_or_none(times),
                "median_uncached_seconds": median_or_none(times),
                "max_seconds": max(
                    (page.elapsed_seconds or 0.0 for page in items),
                    default=0.0,
                ),
                "avg_panels": mean_or_none([page.panel_count for page in items]),
                "avg_text_regions": mean_or_none([page.text_count for page in items]),
                "avg_characters": mean_or_none([page.character_count for page in items]),
                "avg_tails": mean_or_none([page.tail_count for page in items]),
                "one_panel_pages": sum(1 for page in items if page.panel_count <= 1),
                "zero_text_pages": sum(1 for page in items if page.text_count == 0),
                "zero_character_pages": sum(1 for page in items if page.character_count == 0),
                "suspicious_pages": sum(
                    1 for page in items if flags_by_page.get(page.page_id, {}).get("suspicious")
                ),
            }
        )
    return sorted(summaries, key=lambda item: item["comic_id"])


def ranked_pages(
    pages: list[MagiPageAnalysis],
    field_name: str,
    top_n: int,
) -> list[dict[str, Any]]:
    def value(page: MagiPageAnalysis) -> float:
        raw = getattr(page, field_name)
        return float(raw or 0.0)

    return [page_brief(page) | {field_name: value(page)} for page in sorted(pages, key=value, reverse=True)[:top_n]]


def single_panel_dense_pages(pages: list[MagiPageAnalysis]) -> list[dict[str, Any]]:
    selected = [
        page for page in pages if page.panel_count <= 1 and (page.text_count >= 5 or page.character_count >= 4)
    ]
    return [page_brief(page) for page in selected]


def page_brief(page: MagiPageAnalysis) -> dict[str, Any]:
    return {
        "page_id": page.page_id,
        "comic_id": page.comic_id,
        "file_name": page.file_name,
        "path": page.path,
        "elapsed_seconds": page.elapsed_seconds,
        "cache_hit": page.cache_hit,
        "counts": {
            "panels": page.panel_count,
            "texts": page.text_count,
            "characters": page.character_count,
            "tails": page.tail_count,
        },
    }


def mean_or_none(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def median_or_none(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(statistics.median(clean)) if clean else None


if __name__ == "__main__":
    main()
