from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from features.ocr_grouping import group_ocr_blocks
from features.ocr_fusion import fuse_ocr_texts
from features.magi_schema import infer_comic_and_file
from features.page_numbers import detect_page_number_candidates
from features.page_type import classify_page_type
from reports.box_visualization import (
    draw_magi_overlay,
    draw_ocr_grouped_overlay,
    draw_paddle_ocr_readable_overlay_from_dict,
    visual_comic_dir,
    visual_page_name,
)
from tools.analyze_magi_results import load_magi_bundle, normalize_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate page-understanding JSON and a visual HTML review for a run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Standard run directory, e.g. outputs/runs/colab_full_pipeline.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path. Defaults to <run-dir>/analysis/page_understanding_report.json.",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Optional output HTML path. Defaults to <run-dir>/report/index.html.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Limit pages in HTML review. 0 includes every page.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional local by_comic root to resolve original images outside Colab.",
    )
    parser.add_argument(
        "--ocr-report",
        default=None,
        help=(
            "Optional Paddle/Magi OCR comparison JSON. Defaults to "
            "<run-dir>/analysis/paddle_magi_ocr_comparison.json."
        ),
    )
    parser.add_argument("--dataset-name", default="test_1_clean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    analysis_dir = run_dir / "analysis"
    report_dir = run_dir / "report"
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else analysis_dir / "page_understanding_report.json"
    )
    output_html = (
        Path(args.output_html).expanduser().resolve()
        if args.output_html
        else report_dir / "index.html"
    )

    pages = normalize_pages(load_magi_bundle(run_dir / "magi"))
    ocr_report_path = (
        Path(args.ocr_report).expanduser().resolve()
        if args.ocr_report
        else analysis_dir / "paddle_magi_ocr_comparison.json"
    )
    ocr_report = load_optional_json(ocr_report_path)
    comparisons = build_comparison_lookup(ocr_report)
    ocr_results = build_ocr_result_lookup(ocr_report)
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else None

    page_items = []
    for page in pages:
        key = page_key(page.comic_id, page.file_name)
        comparison = comparisons.get(key, {})
        ocr_page = ocr_results.get(key)
        original_image_path = resolve_original_image_path(
            ocr_page or {},
            image_root,
            page.comic_id,
            page.file_name,
            args.dataset_name,
        )
        ocr_groups = ensure_grouped_ocr_visual(
            run_dir=run_dir,
            page=page,
            ocr_page=ocr_page,
            comparison=comparison,
            image_path=original_image_path,
        )
        ensure_magi_visual(
            run_dir=run_dir,
            page=page,
            image_path=original_image_path,
        )
        ensure_ocr_box_visual(
            run_dir=run_dir,
            page=page,
            ocr_page=ocr_page,
            image_path=original_image_path,
        )
        comparison_for_analysis = dict(comparison)
        if ocr_groups:
            comparison_for_analysis["ocr_group_count"] = len(ocr_groups)
        fused_texts = [item.to_dict() for item in fuse_ocr_texts(page, ocr_groups)]
        candidates = [
            item.to_dict()
            for item in detect_page_number_candidates(
                ocr_page,
                image_size=image_size_for_ocr_page(
                    ocr_page=ocr_page,
                    image_path=original_image_path,
                ),
            )
        ]
        page_type = classify_page_type(
            page=page,
            comparison=comparison_for_analysis,
            page_number_candidates=candidates,
        )
        page_items.append(
            {
                "page_id": page.page_id,
                "comic_id": page.comic_id,
                "file_name": page.file_name,
                "image_abs_path": str(original_image_path) if original_image_path else None,
                "image_src": relative_path(original_image_path, output_html)
                if original_image_path
                else None,
                "path": page.path,
                "counts": page.to_dict(include_regions=False)["counts"],
                "interactive": interactive_payload(
                    page=page,
                    ocr_page=ocr_page,
                    ocr_groups=ocr_groups,
                    fused_texts=fused_texts,
                    image_path=original_image_path,
                    output_html=output_html,
                ),
                "page_type": page_type.to_dict(),
                "page_number_candidates": candidates[:5],
                "ocr": ocr_summary(comparison_for_analysis),
                "ocr_groups": compact_group_summaries(ocr_groups),
                "ocr_fusion": compact_fusion_summaries(fused_texts),
                "visuals": visual_paths(run_dir, page.comic_id, page.file_name),
            }
        )

    report = {
        "schema_version": "page_understanding.v1",
        "run_dir": str(run_dir),
        "summary": summarize(page_items),
        "pages": page_items,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    output_html.parent.mkdir(parents=True, exist_ok=True)
    write_split_html_report(
        report=report,
        output_html=output_html,
        max_pages=args.max_pages,
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote page-understanding JSON: {output_json}")
    print(f"Wrote visual HTML report: {output_html}")


def write_split_html_report(report: dict[str, Any], output_html: Path, max_pages: int) -> None:
    pages = report["pages"][: max_pages or None]
    comic_groups = group_pages_by_comic(pages)
    comics_dir = output_html.parent / "comics"
    comics_dir.mkdir(parents=True, exist_ok=True)

    comic_links: dict[str, str] = {}
    for comic_id, comic_pages in comic_groups.items():
        comic_html = comics_dir / f"{html_id(comic_id)}.html"
        comic_links[comic_id] = Path(os.path.relpath(comic_html, output_html.parent)).as_posix()
        comic_report = {
            **report,
            "summary": summarize(comic_pages),
            "pages": comic_pages,
            "report_scope": {"type": "comic", "comic_id": comic_id},
        }
        comic_html.write_text(
            render_html(
                comic_report,
                output_html=comic_html,
                max_pages=0,
                page_mode="comic",
                index_href=Path(os.path.relpath(output_html, comic_html.parent)).as_posix(),
            ),
            encoding="utf-8",
        )

    output_html.write_text(
        render_index_html(
            report=report,
            output_html=output_html,
            pages=pages,
            comic_links=comic_links,
        ),
        encoding="utf-8",
    )


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison_lookup(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in report.get("comparisons") or []:
        key = page_key(item.get("comic_id"), item.get("file_name"))
        lookup[key] = item
    return lookup


def build_ocr_result_lookup(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    comparisons_by_path = {
        str(item.get("image_path") or ""): item
        for item in report.get("comparisons") or []
        if item.get("image_path")
    }
    lookup: dict[str, dict[str, Any]] = {}
    for result in report.get("ocr_results") or []:
        path = str(result.get("path") or "")
        comparison = comparisons_by_path.get(path)
        if comparison:
            lookup[page_key(comparison.get("comic_id"), comparison.get("file_name"))] = result
            continue
        comic_id, file_name = infer_comic_and_file(path)
        lookup[page_key(comic_id, file_name)] = result
    return lookup


def page_key(comic_id: str | None, file_name: str | None) -> str:
    return f"{comic_id or 'unknown'}::{file_name or ''}"


def ocr_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "paddle_text_blocks": comparison.get("paddle_text_blocks"),
        "ocr_group_count": comparison.get("ocr_group_count"),
        "magi_text_regions": comparison.get("magi_text_regions"),
        "matched_regions": comparison.get("matched_regions"),
        "paddle_avg_confidence": comparison.get("paddle_avg_confidence"),
        "paddle_error": comparison.get("paddle_error"),
    }


def ensure_grouped_ocr_visual(
    run_dir: Path,
    page: Any,
    ocr_page: dict[str, Any] | None,
    comparison: dict[str, Any],
    image_path: Path | None,
) -> list[dict[str, Any]]:
    if not ocr_page or ocr_page.get("error") or not ocr_page.get("blocks"):
        return list(comparison.get("ocr_groups") or [])
    if image_path is None:
        return list(comparison.get("ocr_groups") or [])

    groups = group_ocr_blocks(
        ocr_page=ocr_page,
        magi_page=page,
        iou_threshold=0.15,
    )
    group_dicts = [group.to_dict() for group in groups]
    visual_path = visual_comic_dir(run_dir / "visuals" / "ocr_groups", page.comic_id) / visual_page_name(
        page.file_name,
        "ocr_groups",
    )
    draw_ocr_grouped_overlay(
        image_path=image_path,
        groups=groups,
        output_path=visual_path,
    )
    return group_dicts


def ensure_magi_visual(
    run_dir: Path,
    page: Any,
    image_path: Path | None,
) -> Path | None:
    if image_path is None:
        return None
    visual_path = visual_comic_dir(run_dir / "visuals" / "magi_boxes", page.comic_id) / visual_page_name(
        page.file_name,
        "magi_boxes",
    )
    draw_magi_overlay(
        page=page,
        image_path=image_path,
        output_path=visual_path,
    )
    return visual_path


def ensure_ocr_box_visual(
    run_dir: Path,
    page: Any,
    ocr_page: dict[str, Any] | None,
    image_path: Path | None,
) -> Path | None:
    if not ocr_page or ocr_page.get("error") or not ocr_page.get("blocks") or image_path is None:
        return None
    visual_path = visual_comic_dir(run_dir / "visuals" / "ocr_boxes", page.comic_id) / visual_page_name(
        page.file_name,
        "ocr_boxes",
    )
    remapped = dict(ocr_page)
    remapped["path"] = str(image_path)
    draw_paddle_ocr_readable_overlay_from_dict(remapped, visual_path)
    return visual_path


def compact_group_summaries(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for index, group in enumerate(groups, 1):
        compact.append(
            {
                "display_index": index,
                "group_id": group.get("group_id"),
                "source": group.get("source"),
                "text": group.get("text"),
                "confidence": group.get("confidence"),
                "block_indices": group.get("block_indices") or [],
                "box": group.get("box"),
            }
        )
    return compact


def compact_fusion_summaries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items:
        compact.append(
            {
                "display_index": item.get("display"),
                "fusion_id": item.get("id"),
                "text": item.get("text"),
                "source": item.get("source"),
                "confidence": item.get("confidence"),
                "review_flags": item.get("review_flags") or [],
                "options": [
                    {
                        "source": option.get("source"),
                        "text": option.get("text"),
                        "confidence": option.get("confidence"),
                        "target_id": option.get("target_id"),
                    }
                    for option in item.get("options") or []
                ],
            }
        )
    return compact


def interactive_payload(
    page: Any,
    ocr_page: dict[str, Any] | None,
    ocr_groups: list[dict[str, Any]],
    fused_texts: list[dict[str, Any]],
    image_path: Path | None,
    output_html: Path,
) -> dict[str, Any]:
    fusion_by_target = build_fusion_target_lookup(fused_texts)
    return {
        "page_id": page.page_id,
        "comic_id": page.comic_id,
        "file_name": page.file_name,
        "image_src": relative_path(image_path, output_html) if image_path else None,
        "magi": magi_layers(page, fusion_by_target=fusion_by_target),
        "ocr_blocks": [
            {
                "id": f"ocr:block:{block.get('index')}",
                "kind": "ocr_block",
                "display": int(block.get("index") or 0) + 1,
                "text": block.get("text"),
                "confidence": block.get("confidence"),
                "box": block.get("box"),
                "block_index": block.get("index"),
                "fusion": fusion_by_target.get(f"ocr:block:{block.get('index')}"),
            }
            for block in (ocr_page or {}).get("blocks") or []
            if block.get("box") and str(block.get("text") or "").strip()
        ],
        "ocr_groups": [
            {
                "id": f"ocr:group:{index}",
                "kind": "ocr_group",
                "display": index,
                "text": group.get("text"),
                "confidence": group.get("confidence"),
                "box": group.get("box"),
                "block_indices": group.get("block_indices") or [],
                "fusion": fusion_by_target.get(f"ocr:group:{index}"),
            }
            for index, group in enumerate(ocr_groups, 1)
            if group.get("box")
        ],
        "ocr_fusions": fused_texts,
    }


def build_fusion_target_lookup(fused_texts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in fused_texts:
        for target_id in item.get("linked_targets") or []:
            lookup[str(target_id)] = item
        for option in item.get("options") or []:
            target_id = option.get("target_id")
            if target_id:
                lookup[str(target_id)] = item
    return lookup


def magi_layers(
    page: Any,
    fusion_by_target: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    specs = {
        "panels": ("magi_panel", "paneles", page.panels),
        "texts": ("magi_text", "textos", page.texts),
        "characters": ("magi_character", "personajes", page.characters),
        "tails": ("magi_tail", "colas", page.tails),
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for key, (kind, _label, regions) in specs.items():
        result[key] = [
            {
                "id": f"{kind}:{region.index}",
                "kind": kind,
                "display": region.index + 1,
                "text": magi_region_text(kind, region),
                "confidence": None,
                "box": region.box.to_dict(),
                "attributes": region.attributes,
                "fusion": (fusion_by_target or {}).get(f"{kind}:{region.index}"),
            }
            for region in regions
        ]
    return result


def magi_region_text(kind: str, region: Any) -> str:
    if kind == "magi_text":
        ocr_text = str(region.attributes.get("ocr_text") or "").strip()
        if ocr_text:
            return ocr_text
    if kind == "magi_character" and region.attributes.get("cluster_label") is not None:
        return f"personaje cluster {region.attributes['cluster_label']}"
    if kind == "magi_text" and region.attributes.get("is_essential") is not None:
        suffix = "esencial" if region.attributes["is_essential"] else "decorativo"
        return f"sin OCR Magi ({suffix})"
    return f"{kind}:{region.index + 1}"


def image_size_for_ocr_page(
    ocr_page: dict[str, Any] | None,
    image_path: Path | None,
) -> tuple[int, int] | None:
    if not ocr_page:
        return None
    if image_path is None:
        return None
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return None


def resolve_original_image_path(
    ocr_page: dict[str, Any],
    image_root: Path | None,
    comic_id: str | None,
    file_name: str,
    dataset_name: str,
) -> Path | None:
    direct_value = str(ocr_page.get("path") or "").strip()
    if direct_value:
        direct = Path(direct_value)
        if direct.exists() and direct.is_file():
            return direct.resolve()
    if image_root is None or not comic_id:
        return None
    candidates = [
        image_root / comic_id / dataset_name / file_name,
        image_root / "by_comic" / comic_id / dataset_name / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = list(image_root.rglob(f"{comic_id}/**/{file_name}"))
    return matches[0].resolve() if matches else None


def visual_paths(run_dir: Path, comic_id: str | None, file_name: str) -> dict[str, str | None]:
    visuals_dir = run_dir / "visuals"
    specs = {
        "magi_boxes": ("magi_boxes", "magi_boxes"),
        "ocr_boxes": ("ocr_boxes", "ocr_boxes"),
        "ocr_groups": ("ocr_groups", "ocr_groups"),
    }
    result: dict[str, str | None] = {}
    for key, (folder, suffix) in specs.items():
        path = visual_comic_dir(visuals_dir / folder, comic_id) / visual_page_name(file_name, suffix)
        result[key] = str(path) if path.exists() else None
    return result


def summarize(pages: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts = Counter(item["page_type"]["page_type"] for item in pages)
    number_pages = sum(1 for item in pages if item["page_number_candidates"])
    comics = {item["comic_id"] for item in pages if item["comic_id"]}
    return {
        "page_count": len(pages),
        "comic_count": len(comics),
        "page_type_counts": dict(type_counts),
        "pages_with_number_candidates": number_pages,
    }


def render_index_html(
    report: dict[str, Any],
    output_html: Path,
    pages: list[dict[str, Any]],
    comic_links: dict[str, str],
) -> str:
    summary = report["summary"]
    comic_groups = group_pages_by_comic(pages)
    type_summary = " ".join(
        f'<span class="type-chip">{escape(page_type)}: {count}</span>'
        for page_type, count in sorted(summary.get("page_type_counts", {}).items())
    )
    comic_cards = "\n".join(
        render_comic_index_card(
            comic_id=comic_id,
            pages=comic_pages,
            href=comic_links.get(comic_id, "#"),
            output_html=output_html,
        )
        for comic_id, comic_pages in comic_groups.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ComicAnalizer - Reporte por comics</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --surface: #ffffff;
      --ink: #17202a;
      --muted: #64748b;
      --line: #d8dee6;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; background: var(--bg); color: var(--ink); }}
    header {{ padding: 30px 34px 26px; background: linear-gradient(135deg, #17202a, #263445); color: white; }}
    main {{ padding: 24px 32px 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .run-path {{ display: inline-block; max-width: 940px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.18); border-radius: 6px; padding: 8px 10px; color: #e6edf3; font-family: Consolas, monospace; font-size: 13px; overflow-wrap: anywhere; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; margin-top: 18px; }}
    .stat {{ background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.18); border-radius: 8px; padding: 12px; }}
    .stat strong {{ display: block; font-size: 24px; line-height: 1; }}
    .stat span {{ color: #d2dce7; font-size: 13px; }}
    .type-summary {{ margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }}
    .type-chip {{ background: #e8f3f1; color: #0f514b; border: 1px solid #b9ded8; border-radius: 999px; padding: 5px 9px; font-size: 13px; }}
    .intro {{ max-width: 980px; margin: 0 0 18px; color: #475569; line-height: 1.45; }}
    .comic-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 16px; }}
    .comic-card {{ display: grid; grid-template-columns: 108px 1fr; gap: 14px; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px; text-decoration: none; color: inherit; box-shadow: 0 1px 2px rgba(15,23,42,.06); }}
    .comic-card:hover {{ border-color: #7dbdb5; box-shadow: 0 8px 24px rgba(15,23,42,.10); }}
    .thumb {{ width: 108px; height: 150px; background: #e2e8f0; border: 1px solid #cbd5e1; border-radius: 6px; overflow: hidden; display: flex; align-items: center; justify-content: center; color: #64748b; font-size: 12px; }}
    .thumb img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
    .comic-title {{ font-size: 16px; font-weight: 700; margin: 2px 0 8px; overflow-wrap: anywhere; }}
    .comic-metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; margin: 0 0 10px; }}
    .metric {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 7px; }}
    .metric strong {{ display: block; font-size: 17px; }}
    .metric span {{ display: block; color: #64748b; font-size: 12px; }}
    .type-line {{ color: #475569; font-size: 13px; line-height: 1.35; }}
    .open-label {{ display: inline-block; margin-top: 10px; color: #0f766e; font-weight: 700; font-size: 13px; }}
    @media (max-width: 760px) {{
      main {{ padding: 18px; }}
      header {{ padding: 24px 20px; }}
      .summary {{ grid-template-columns: 1fr 1fr; }}
      .comic-card {{ grid-template-columns: 86px 1fr; }}
      .thumb {{ width: 86px; height: 122px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>ComicAnalizer - Reporte por comics</h1>
    <div class="run-path">{escape(report.get("run_dir", ""))}</div>
    <div class="summary">
      <div class="stat"><strong>{summary.get("page_count", 0)}</strong><span>paginas</span></div>
      <div class="stat"><strong>{summary.get("comic_count", 0)}</strong><span>comics</span></div>
      <div class="stat"><strong>{summary.get("pages_with_number_candidates", 0)}</strong><span>posibles numeros</span></div>
      <div class="stat"><strong>{len(summary.get("page_type_counts", {}))}</strong><span>tipos detectados</span></div>
    </div>
    <div class="type-summary">{type_summary}</div>
  </header>
  <main>
    <p class="intro">Selecciona un comic para abrir su reporte interactivo separado. Cada vista mantiene sus propios filtros, capas Magi/OCR, zoom y sistema de correcciones.</p>
    <div class="comic-grid">{comic_cards}</div>
  </main>
</body>
</html>
"""


def render_comic_index_card(
    comic_id: str,
    pages: list[dict[str, Any]],
    href: str,
    output_html: Path,
) -> str:
    summary = summarize(pages)
    type_counts = Counter(page["page_type"]["page_type"] for page in pages)
    type_text = ", ".join(f"{key}: {value}" for key, value in sorted(type_counts.items())) or "sin tipos"
    text_regions = sum(int((page.get("ocr") or {}).get("magi_text_regions") or 0) for page in pages)
    ocr_groups = sum(int((page.get("ocr") or {}).get("ocr_group_count") or 0) for page in pages)
    thumb_src = None
    for page in pages:
        image_abs_path = page.get("image_abs_path")
        if image_abs_path and Path(image_abs_path).exists():
            thumb_src = relative_path(Path(image_abs_path), output_html)
            break
    thumb = f'<img src="{escape(thumb_src)}" loading="lazy" alt="{escape(comic_id)}">' if thumb_src else "sin imagen"
    return f"""
<a class="comic-card" href="{escape(href)}">
  <div class="thumb">{thumb}</div>
  <div>
    <div class="comic-title">{escape(comic_id)}</div>
    <div class="comic-metrics">
      <div class="metric"><strong>{summary.get("page_count", 0)}</strong><span>paginas</span></div>
      <div class="metric"><strong>{summary.get("pages_with_number_candidates", 0)}</strong><span>numeracion</span></div>
      <div class="metric"><strong>{text_regions}</strong><span>textos Magi</span></div>
      <div class="metric"><strong>{ocr_groups}</strong><span>grupos OCR</span></div>
    </div>
    <div class="type-line">{escape(type_text)}</div>
    <span class="open-label">Abrir reporte del comic</span>
  </div>
</a>
"""


def render_html(
    report: dict[str, Any],
    output_html: Path,
    max_pages: int,
    page_mode: str = "run",
    index_href: str | None = None,
) -> str:
    pages = report["pages"][: max_pages or None]
    summary = report["summary"]
    scope = report.get("report_scope") or {}
    scope_title = (
        f"ComicAnalizer - {scope.get('comic_id')}"
        if page_mode == "comic" and scope.get("comic_id")
        else "ComicAnalizer Run Report"
    )
    back_link = (
        f'<a class="back-link" href="{escape(index_href)}">Volver al indice de comics</a>'
        if index_href
        else ""
    )
    comic_groups = group_pages_by_comic(pages)
    rows = "\n".join(
        render_comic_section(comic_id, comic_pages, output_html)
        for comic_id, comic_pages in comic_groups.items()
    )
    comic_nav = " ".join(
        f'<a class="comic-chip" href="#{escape(html_id(comic_id))}">{escape(comic_id)} <span>{len(comic_pages)}</span></a>'
        for comic_id, comic_pages in comic_groups.items()
    )
    comic_options = "\n".join(
        f'<option value="{escape(comic_id)}">{escape(comic_id)} ({len(comic_pages)})</option>'
        for comic_id, comic_pages in comic_groups.items()
    )
    type_summary = " ".join(
        f'<span class="type-chip">{escape(page_type)}: {count}</span>'
        for page_type, count in sorted(summary.get("page_type_counts", {}).items())
    )
    type_options = "\n".join(
        f'<option value="{escape(page_type)}">{escape(page_type)}</option>'
        for page_type in sorted(summary.get("page_type_counts", {}))
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(scope_title)}</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --surface: #ffffff;
      --ink: #17202a;
      --muted: #64748b;
      --line: #d8dee6;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: Arial, sans-serif; margin: 0; background: var(--bg); color: var(--ink); }}
    body.modal-open {{ overflow: hidden; }}
    header {{ padding: 28px 34px 24px; background: linear-gradient(135deg, #17202a, #263445); color: white; }}
    main {{ padding: 24px 32px; }}
    .hero {{ display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .run-path {{ display: inline-block; max-width: 920px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.18); border-radius: 6px; padding: 8px 10px; color: #e6edf3; font-family: Consolas, monospace; font-size: 13px; overflow-wrap: anywhere; }}
    .back-link {{ display: inline-block; margin: 0 0 12px; color: #dffaf6; text-decoration: none; border: 1px solid rgba(255,255,255,.24); background: rgba(255,255,255,.08); border-radius: 6px; padding: 7px 10px; font-size: 13px; }}
    .back-link:hover {{ background: rgba(255,255,255,.14); }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; margin-top: 18px; }}
    .stat {{ background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.18); border-radius: 8px; padding: 12px; }}
    .stat strong {{ display: block; font-size: 24px; line-height: 1; }}
    .stat span {{ color: #d2dce7; font-size: 13px; }}
    .type-summary {{ margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }}
    .type-chip {{ background: #e8f3f1; color: #0f514b; border: 1px solid #b9ded8; border-radius: 999px; padding: 5px 9px; font-size: 13px; }}
    .pill {{ display: inline-block; background: #eef2f7; border: 1px solid #d5dce5; border-radius: 6px; padding: 8px 10px; margin-bottom: 14px; }}
    .comic-nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 0; }}
    .comic-chip {{ color: #e6edf3; text-decoration: none; border: 1px solid rgba(255,255,255,.22); background: rgba(255,255,255,.08); border-radius: 999px; padding: 6px 10px; font-size: 13px; }}
    .comic-chip span {{ color: #b9c7d6; margin-left: 4px; }}
    .toolbar {{ display: grid; grid-template-columns: 1.5fr repeat(5, minmax(140px, 1fr)); gap: 10px; margin: 0 0 18px; }}
    .toolbar input, .toolbar select, .toolbar label {{ background: white; border: 1px solid #cfd6dd; border-radius: 5px; padding: 9px 10px; font-size: 14px; }}
    .toolbar label {{ display: flex; align-items: center; gap: 8px; }}
    .global-layer-controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: stretch; margin: 0 0 14px; padding: 10px; background: white; border: 1px solid var(--line); border-radius: 8px; }}
    .global-layer-group {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 6px 8px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; }}
    .global-layer-group strong {{ margin-right: 4px; font-size: 13px; color: #334155; }}
    .global-layer-group label {{ display: inline-flex; gap: 5px; align-items: center; font-size: 13px; color: #334155; }}
    .global-layer-controls button {{ border: 1px solid #cfd6dd; background: #f8fafc; border-radius: 5px; padding: 7px 9px; cursor: pointer; font-size: 13px; }}
    .global-layer-controls button:hover {{ background: #e8f3f1; border-color: #9bd2ca; }}
    .comic-section {{ margin: 0 0 28px; scroll-margin-top: 16px; }}
    .comic-section.hidden {{ display: none; }}
    .comic-header {{ position: sticky; top: 0; z-index: 5; display: flex; justify-content: space-between; gap: 16px; align-items: center; padding: 12px 14px; margin: 0 0 12px; background: #e8eef5; border: 1px solid #d1dbe7; border-radius: 8px; box-shadow: 0 1px 2px rgba(15,23,42,.05); }}
    .comic-header h2 {{ margin: 0; font-size: 18px; }}
    .comic-header .meta {{ color: #475569; }}
    .comic-pages {{ min-height: 12px; }}
    .card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; margin-bottom: 18px; overflow: hidden; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06); }}
    .card-header {{ display: flex; justify-content: space-between; gap: 16px; padding: 14px 16px; border-bottom: 1px solid #e2e2e2; }}
    .meta {{ color: #555; font-size: 14px; }}
    .visuals {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; padding: 16px; }}
    .interactive-review {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 16px; }}
    .interactive-viewer {{ border: 1px solid #dfe5ec; border-radius: 6px; background: #fbfcfd; overflow: hidden; }}
    .viewer-head {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 10px 12px; background: #f7f9fb; border-bottom: 1px solid #dfe5ec; }}
    .viewer-controls {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; font-size: 13px; }}
    .viewer-controls label {{ display: inline-flex; gap: 4px; align-items: center; }}
    .mode-button {{ border: 1px solid #cfd6dd; background: white; border-radius: 5px; padding: 5px 8px; cursor: pointer; }}
    .mode-button.active {{ background: #0f766e; color: white; border-color: #0f766e; }}
    .viewer-body {{ display: grid; grid-template-columns: minmax(0, 1fr) 240px; min-height: 560px; }}
    .canvas-wrap {{ position: relative; height: 560px; background: #eef2f7; overflow: hidden; cursor: zoom-in; }}
    .clean-page {{ width: 100%; height: 100%; object-fit: contain; display: block; user-select: none; -webkit-user-drag: none; }}
    .overlay-svg {{ position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: auto; cursor: crosshair; }}
    .overlay-box {{ fill: transparent; stroke-width: 4; vector-effect: non-scaling-stroke; pointer-events: none; }}
    .overlay-badge {{ pointer-events: none; }}
    .overlay-badge circle {{ stroke: #111827; stroke-width: 1.5; }}
    .overlay-badge text {{ fill: white; font-size: 13px; font-weight: 700; text-anchor: middle; dominant-baseline: central; }}
    .corrected .overlay-box {{ stroke-dasharray: 8 5; }}
    .selected-overlay .overlay-box {{ stroke-width: 7; filter: drop-shadow(0 0 5px rgba(15,23,42,.36)); }}
    .interactive-legend {{ border-left: 1px solid #dfe5ec; background: #fff; overflow: auto; max-height: 560px; padding: 10px; }}
    .legend-title {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 13px; color: #334155; }}
    .legend-list {{ display: grid; gap: 6px; }}
    .legend-item {{ width: 100%; text-align: left; border: 1px solid #dce3ea; background: #f8fafc; border-radius: 6px; padding: 7px 8px; cursor: pointer; display: grid; grid-template-columns: auto 1fr auto; gap: 7px; align-items: start; font-size: 12px; color: #17202a; }}
    .legend-item:hover {{ border-color: #94a3b8; background: #f1f5f9; }}
    .legend-item.active {{ outline: 2px solid #0f766e; border-color: #0f766e; }}
    .legend-item.corrected {{ background: #fff7ed; border-color: #fb923c; }}
    .legend-swatch {{ width: 12px; height: 12px; border-radius: 999px; margin-top: 2px; box-shadow: inset 0 0 0 1px rgba(15,23,42,.18); }}
    .legend-main {{ min-width: 0; }}
    .legend-name {{ font-weight: 700; margin-bottom: 2px; }}
    .legend-text {{ color: #475569; overflow-wrap: anywhere; line-height: 1.25; }}
    .legend-mark {{ color: #b45309; font-weight: 700; font-size: 11px; }}
    .fallback-visuals {{ margin: 0 16px 16px; border: 1px solid #dfe5ec; border-radius: 6px; background: #fbfcfd; }}
    .fallback-visuals summary {{ padding: 10px 12px; cursor: pointer; color: #334155; }}
    .visual {{ border: 1px solid #dfe5ec; border-radius: 6px; background: #fbfcfd; min-height: 120px; overflow: hidden; }}
    .visual h4 {{ margin: 0; padding: 10px 12px; font-size: 14px; border-bottom: 1px solid #dfe5ec; background: #f7f9fb; display: flex; justify-content: space-between; gap: 8px; }}
    .visual small {{ color: var(--muted); font-weight: normal; }}
    .visual-link {{ display: block; background: #f0f3f6; cursor: zoom-in; }}
    .visual img {{ width: 100%; height: 520px; object-fit: contain; display: block; }}
    .details {{ padding: 0 16px 16px; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .info-panel {{ background: #f8fafc; border: 1px solid #e3e8ef; border-radius: 6px; padding: 12px; color: #25313f; }}
    .info-panel h5 {{ margin: 0 0 8px; font-size: 14px; color: #0f172a; }}
    .info-panel dl {{ display: grid; grid-template-columns: 1fr auto; gap: 6px 12px; margin: 0; font-size: 13px; }}
    .info-panel dt {{ color: #64748b; }}
    .info-panel dd {{ margin: 0; font-weight: 700; }}
    .note-list {{ margin: 0; padding-left: 18px; font-size: 13px; }}
    .note-list li {{ margin-bottom: 5px; }}
    .group-preview {{ margin-top: 8px; padding-top: 8px; border-top: 1px solid #e3e8ef; font-size: 13px; }}
    a {{ color: #155e75; }}
    .modal {{ position: fixed; inset: 0; display: none; align-items: center; justify-content: center; background: rgba(8, 13, 20, 0.82); z-index: 20; padding: 24px; }}
    .modal.open {{ display: flex; }}
    .modal-inner {{ max-width: 96vw; max-height: 94vh; background: #111827; border: 1px solid rgba(255,255,255,0.18); border-radius: 8px; overflow: hidden; }}
    .modal-title {{ color: white; padding: 10px 14px; font-size: 14px; display: flex; justify-content: space-between; gap: 16px; }}
    .modal-title span:last-child {{ color: #cbd5e1; }}
    .modal-stage {{ width: min(96vw, 1500px); height: 84vh; overflow: hidden; display: flex; align-items: center; justify-content: center; cursor: grab; background: #0b1220; touch-action: none; overscroll-behavior: contain; }}
    .modal-stage:active {{ cursor: grabbing; }}
    .modal img {{ max-width: none; max-height: none; width: auto; height: auto; transform-origin: 0 0; user-select: none; -webkit-user-drag: none; pointer-events: none; }}
    .modal-tools {{ display: flex; gap: 6px; align-items: center; }}
    .modal-tools button, .correction-panel button {{ border: 1px solid #cbd5e1; background: white; border-radius: 5px; padding: 5px 8px; cursor: pointer; }}
    .correction-panel {{ position: fixed; right: 22px; bottom: 22px; width: 360px; max-width: calc(100vw - 44px); background: white; border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: 0 18px 38px rgba(15,23,42,0.24); z-index: 25; display: none; }}
    .correction-panel.open {{ display: block; }}
    .correction-panel header {{ background: #17202a; color: white; padding: 10px 12px; border-radius: 8px 8px 0 0; }}
    .correction-panel .body {{ padding: 12px; }}
    .correction-panel textarea, .correction-panel select, .correction-panel input {{ width: 100%; border: 1px solid #cbd5e1; border-radius: 5px; padding: 8px; margin: 6px 0 10px; }}
    .box-comparison {{ border: 1px solid #dbe4ee; border-radius: 6px; background: #f8fafc; padding: 9px; margin: 8px 0 10px; font-size: 12px; }}
    .comparison-empty {{ color: #64748b; }}
    .comparison-suggestion {{ margin-bottom: 4px; font-size: 13px; }}
    .comparison-source {{ color: #64748b; margin-bottom: 6px; }}
    .comparison-flags {{ display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 7px; }}
    .review-flag {{ background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; border-radius: 999px; padding: 2px 6px; }}
    .comparison-options {{ display: grid; gap: 5px; }}
    .comparison-option {{ display: grid; gap: 2px; border-top: 1px solid #e2e8f0; padding-top: 5px; }}
    .comparison-option span {{ overflow-wrap: anywhere; }}
    .comparison-option small {{ color: #64748b; }}
    @media (max-width: 1200px) {{
      .toolbar {{ grid-template-columns: 1fr 1fr; }}
      .interactive-review {{ grid-template-columns: 1fr; }}
      .viewer-body {{ grid-template-columns: 1fr; }}
      .interactive-legend {{ border-left: 0; border-top: 1px solid #dfe5ec; max-height: 260px; }}
      .visuals {{ grid-template-columns: 1fr; }}
      .details {{ grid-template-columns: 1fr; }}
      .summary {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="hero">
      <div>
        {back_link}
        <h1>{escape(scope_title)}</h1>
        <div class="run-path">{escape(report.get("run_dir", ""))}</div>
      </div>
    </div>
    <div class="summary">
      <div class="stat"><strong>{summary.get("page_count", 0)}</strong><span>paginas</span></div>
      <div class="stat"><strong>{summary.get("comic_count", 0)}</strong><span>comics</span></div>
      <div class="stat"><strong>{summary.get("pages_with_number_candidates", 0)}</strong><span>posibles numeros</span></div>
      <div class="stat"><strong>{len(summary.get("page_type_counts", {}))}</strong><span>tipos detectados</span></div>
    </div>
    <div class="type-summary">{type_summary}</div>
    <nav class="comic-nav" aria-label="Navegacion por comic">{comic_nav}</nav>
  </header>
  <main>
    <div class="toolbar">
      <input id="searchBox" type="search" placeholder="Buscar comic, pagina o texto OCR">
      <select id="comicFilter">
        <option value="">Todos los comics</option>
        {comic_options}
      </select>
      <select id="typeFilter">
        <option value="">Todos los tipos</option>
        {type_options}
      </select>
      <select id="sortMode">
        <option value="file">Orden por pagina</option>
        <option value="ocr-desc">Mas bloques OCR</option>
        <option value="groups-desc">Mas grupos OCR</option>
        <option value="magi-desc">Mas regiones Magi</option>
      </select>
      <label><input id="onlyNumbers" type="checkbox"> Solo con numeracion</label>
      <label><input id="onlyUnknown" type="checkbox"> Solo unknown</label>
    </div>
    <div class="global-layer-controls" aria-label="Controles globales de capas">
      <div class="global-layer-group">
        <strong>Magi</strong>
        <label><input type="checkbox" data-global-magi-layer="panels" checked> Paneles</label>
        <label><input type="checkbox" data-global-magi-layer="texts" checked> Textos</label>
        <label><input type="checkbox" data-global-magi-layer="characters" checked> Personajes</label>
        <label><input type="checkbox" data-global-magi-layer="tails" checked> Colas</label>
        <button type="button" data-global-magi-preset="all">Todo</button>
        <button type="button" data-global-magi-preset="none">Solo imagen</button>
      </div>
      <div class="global-layer-group">
        <strong>OCR</strong>
        <label><input type="checkbox" data-global-ocr-layer="blocks" checked> Bloques</label>
        <label><input type="checkbox" data-global-ocr-layer="groups"> Grupos</label>
        <button type="button" data-global-ocr-preset="all">Todo</button>
        <button type="button" data-global-ocr-preset="none">Solo imagen</button>
      </div>
    </div>
    <div id="visibleCount" class="pill"></div>
    <div id="cards">{rows}</div>
  </main>
  <div id="imageModal" class="modal" aria-hidden="true">
    <div class="modal-inner">
      <div class="modal-title">
        <strong id="modalTitle"></strong>
        <div class="modal-tools">
          <button type="button" id="zoomOut">-</button>
          <button type="button" id="zoomReset">100%</button>
          <button type="button" id="zoomIn">+</button>
          <span>click fuera o Esc para cerrar</span>
        </div>
      </div>
      <div id="modalStage" class="modal-stage"><img id="modalImage" alt=""></div>
    </div>
  </div>
  <section id="correctionPanel" class="correction-panel">
    <header><strong>Correccion de caja</strong></header>
    <div class="body">
      <div id="selectedBoxInfo" class="meta"></div>
      <div id="boxComparison" class="box-comparison"></div>
      <label>Estado</label>
      <select id="correctionLabel">
        <option value="correct">Correcta</option>
        <option value="bad_text">Texto malo</option>
        <option value="bad_box">Caja mala</option>
        <option value="false_positive">Falso positivo</option>
        <option value="missing_merge">Falta unir</option>
        <option value="wrong_group">Grupo incorrecto</option>
        <option value="missed_detection">Falta deteccion cercana</option>
      </select>
      <label>Texto corregido / nota</label>
      <textarea id="correctionText" rows="4"></textarea>
      <div class="modal-tools">
        <button type="button" id="saveCorrection">Guardar</button>
        <button type="button" id="clearCorrection">Quitar marca</button>
        <button type="button" id="closeCorrection">Cerrar</button>
      </div>
    </div>
  </section>
  <button type="button" id="downloadCorrections" style="position:fixed;left:22px;bottom:22px;z-index:18;border:1px solid #0f766e;background:#0f766e;color:white;border-radius:7px;padding:10px 12px;box-shadow:0 10px 24px rgba(15,23,42,.18);">Descargar correcciones</button>
  <script>
    const cards = Array.from(document.querySelectorAll('.card'));
    const searchBox = document.getElementById('searchBox');
    const comicFilter = document.getElementById('comicFilter');
    const typeFilter = document.getElementById('typeFilter');
    const sortMode = document.getElementById('sortMode');
    const onlyNumbers = document.getElementById('onlyNumbers');
    const onlyUnknown = document.getElementById('onlyUnknown');
    const visibleCount = document.getElementById('visibleCount');
    const cardsRoot = document.getElementById('cards');
    const sections = Array.from(document.querySelectorAll('.comic-section'));
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const modalTitle = document.getElementById('modalTitle');
    const modalStage = document.getElementById('modalStage');
    const correctionPanel = document.getElementById('correctionPanel');
    const selectedBoxInfo = document.getElementById('selectedBoxInfo');
    const correctionLabel = document.getElementById('correctionLabel');
    const correctionText = document.getElementById('correctionText');
    const corrections = loadCorrections();
    modalImage.draggable = false;
    modalImage.addEventListener('dragstart', event => event.preventDefault());
    let selectedShape = null;
    let selectedPayload = null;
    let zoom = 1;
    let panX = 0;
    let panY = 0;
    let dragging = false;
    let dragStart = null;

    function applyFilters() {{
      const query = searchBox.value.trim().toLowerCase();
      const comic = comicFilter.value;
      const type = typeFilter.value;
      const requireNumbers = onlyNumbers.checked;
      const requireUnknown = onlyUnknown.checked;
      let visible = 0;

      cards.forEach(card => {{
        const text = card.dataset.search || '';
        const matchesQuery = !query || text.includes(query);
        const matchesComic = !comic || card.dataset.comic === comic;
        const matchesType = !type || card.dataset.pageType === type;
        const matchesNumbers = !requireNumbers || card.dataset.hasNumbers === '1';
        const matchesUnknown = !requireUnknown || card.dataset.pageType === 'unknown';
        const show = matchesQuery && matchesComic && matchesType && matchesNumbers && matchesUnknown;
        card.style.display = show ? '' : 'none';
        if (show) visible += 1;
      }});
      let visibleComics = 0;
      sections.forEach(section => {{
        const shown = Array.from(section.querySelectorAll('.card')).filter(card => card.style.display !== 'none').length;
        section.classList.toggle('hidden', shown === 0);
        const count = section.querySelector('.section-visible-count');
        if (count) count.textContent = `${{shown}} / ${{section.dataset.pageCount}} paginas`;
        if (shown) visibleComics += 1;
      }});
      visibleCount.textContent = `Mostrando ${{visible}} / ${{cards.length}} paginas en ${{visibleComics}} comics`;
    }}

    function applySort() {{
      const mode = sortMode.value;
      const sorted = [...cards].sort((a, b) => {{
        if (mode === 'ocr-desc') return Number(b.dataset.ocrBlocks) - Number(a.dataset.ocrBlocks);
        if (mode === 'groups-desc') return Number(b.dataset.ocrGroups) - Number(a.dataset.ocrGroups);
        if (mode === 'magi-desc') return Number(b.dataset.magiTexts) - Number(a.dataset.magiTexts);
        return Number(a.dataset.fileIndex) - Number(b.dataset.fileIndex);
      }});
      sections.forEach(section => {{
        const container = section.querySelector('.comic-pages');
        const sectionCards = sorted.filter(card => card.closest('.comic-section') === section);
        sectionCards.forEach(card => container.appendChild(card));
      }});
      applyFilters();
    }}

    [searchBox, comicFilter, typeFilter, onlyNumbers, onlyUnknown].forEach(el => el.addEventListener('input', applyFilters));
    sortMode.addEventListener('change', applySort);
    document.querySelectorAll('.visual-link').forEach(link => {{
      link.addEventListener('click', event => {{
        event.preventDefault();
        openModal(link.href, link.dataset.title || '');
      }});
    }});
    modal.addEventListener('click', event => {{
      if (event.target === modal) {{
        closeModal();
      }}
    }});
    document.addEventListener('keydown', event => {{
      if (event.key === 'Escape') {{
        closeModal();
      }}
    }});
    document.getElementById('zoomIn').addEventListener('click', () => setZoom(zoom * 1.25));
    document.getElementById('zoomOut').addEventListener('click', () => setZoom(zoom / 1.25));
    document.getElementById('zoomReset').addEventListener('click', () => {{
      zoom = 1; panX = 0; panY = 0; applyModalTransform();
    }});
    modalStage.addEventListener('wheel', event => {{
      event.preventDefault();
      event.stopPropagation();
      setZoom(zoom * (event.deltaY < 0 ? 1.12 : 0.88));
    }}, {{ passive: false }});
    modalStage.addEventListener('pointerdown', event => {{
      if (event.button !== 0) return;
      event.preventDefault();
      dragging = true;
      dragStart = {{ x: event.clientX, y: event.clientY, panX, panY }};
      modalStage.setPointerCapture(event.pointerId);
    }});
    modalStage.addEventListener('pointermove', event => {{
      if (!dragging || !dragStart) return;
      event.preventDefault();
      panX = dragStart.panX + event.clientX - dragStart.x;
      panY = dragStart.panY + event.clientY - dragStart.y;
      applyModalTransform();
    }});
    modalStage.addEventListener('pointerup', stopModalDrag);
    modalStage.addEventListener('pointercancel', stopModalDrag);
    modalStage.addEventListener('lostpointercapture', stopModalDrag);

    document.querySelectorAll('.card').forEach(card => setupInteractiveCard(card));
    document.querySelectorAll('[data-global-magi-layer]').forEach(input => {{
      input.addEventListener('change', syncGlobalMagiLayers);
    }});
    document.querySelectorAll('[data-global-ocr-layer]').forEach(input => {{
      input.addEventListener('change', syncGlobalOcrLayers);
    }});
    document.querySelectorAll('[data-global-magi-preset]').forEach(button => {{
      button.addEventListener('click', () => applyGlobalMagiPreset(button.dataset.globalMagiPreset));
    }});
    document.querySelectorAll('[data-global-ocr-preset]').forEach(button => {{
      button.addEventListener('click', () => applyGlobalOcrPreset(button.dataset.globalOcrPreset));
    }});
    document.getElementById('saveCorrection').addEventListener('click', saveCurrentCorrection);
    document.getElementById('clearCorrection').addEventListener('click', clearCurrentCorrection);
    document.getElementById('closeCorrection').addEventListener('click', () => correctionPanel.classList.remove('open'));
    document.getElementById('downloadCorrections').addEventListener('click', downloadCorrections);
    applySort();

    function openModal(src, title) {{
      modalImage.src = src;
      modalTitle.textContent = title;
      zoom = 1; panX = 0; panY = 0; applyModalTransform();
      modal.classList.add('open');
      modal.setAttribute('aria-hidden', 'false');
      document.body.classList.add('modal-open');
    }}

    function closeModal() {{
      modal.classList.remove('open');
      modal.setAttribute('aria-hidden', 'true');
      modalImage.src = '';
      document.body.classList.remove('modal-open');
      stopModalDrag();
    }}

    function setZoom(value) {{
      zoom = Math.min(8, Math.max(0.25, value));
      applyModalTransform();
    }}

    function applyModalTransform() {{
      modalImage.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{zoom}})`;
    }}

    function stopModalDrag() {{
      dragging = false;
      dragStart = null;
    }}

    function setupInteractiveCard(card) {{
      const payloadTag = card.querySelector('.page-payload');
      if (!payloadTag) return;
      const payload = JSON.parse(payloadTag.textContent || '{{}}');
      card.querySelectorAll('.interactive-viewer').forEach(viewer => {{
        const img = viewer.querySelector('.clean-page');
        const svg = viewer.querySelector('.overlay-svg');
        if (!payload.image_src) return;
        img.src = payload.image_src;
        img.draggable = false;
        img.addEventListener('dragstart', event => event.preventDefault());
        img.addEventListener('load', () => renderViewer(viewer, payload));
        img.addEventListener('click', () => openModal(payload.image_src, viewer.dataset.pageTitle || ''));
        viewer.querySelectorAll('input[data-layer]').forEach(input => input.addEventListener('change', () => renderViewer(viewer, payload)));
        viewer.querySelectorAll('input[data-ocr-layer]').forEach(input => input.addEventListener('change', () => renderViewer(viewer, payload)));
        svg.addEventListener('click', event => handleOverlayClick(event, viewer, payload));
      }});
    }}

    function renderViewer(viewer, payload) {{
      const img = viewer.querySelector('.clean-page');
      const svg = viewer.querySelector('.overlay-svg');
      const rect = img.getBoundingClientRect();
      const naturalW = img.naturalWidth || 1;
      const naturalH = img.naturalHeight || 1;
      const rendered = containRect(rect.width, rect.height, naturalW, naturalH);
      svg.setAttribute('viewBox', `0 0 ${{rect.width}} ${{rect.height}}`);
      svg.innerHTML = '';
      let boxes = [];
      if (viewer.dataset.viewer === 'magi') {{
        const enabled = new Set(Array.from(viewer.querySelectorAll('input[data-layer]:checked')).map(input => input.dataset.layer));
        boxes = [...enabled].flatMap(layer => (payload.magi && payload.magi[layer]) || []);
      }} else {{
        const enabled = new Set(Array.from(viewer.querySelectorAll('input[data-ocr-layer]:checked')).map(input => input.dataset.ocrLayer));
        boxes = [
          ...(enabled.has('groups') ? payload.ocr_groups || [] : []),
          ...(enabled.has('blocks') ? payload.ocr_blocks || [] : []),
        ];
      }}
      const hitBoxes = [];
      boxes.forEach(box => {{
        const hit = renderedBox(box, rendered);
        if (hit) hitBoxes.push({{ box, hit }});
        drawInteractiveBox(svg, box, payload, rendered, hit);
      }});
      viewer.__hitBoxes = hitBoxes;
      renderLegend(viewer, boxes, payload);
      refreshCorrectionMarks(viewer);
    }}

    function containRect(width, height, naturalW, naturalH) {{
      const scale = Math.min(width / naturalW, height / naturalH);
      const drawW = naturalW * scale;
      const drawH = naturalH * scale;
      return {{ scale, offsetX: (width - drawW) / 2, offsetY: (height - drawH) / 2 }};
    }}

    function renderedBox(box, rendered) {{
      if (!box.box) return null;
      const b = box.box;
      const x = rendered.offsetX + Number(b.x1 || 0) * rendered.scale;
      const y = rendered.offsetY + Number(b.y1 || 0) * rendered.scale;
      const w = Math.max(3, Number(b.x2 || 0) * rendered.scale + rendered.offsetX - x);
      const h = Math.max(3, Number(b.y2 || 0) * rendered.scale + rendered.offsetY - y);
      return {{ x, y, w, h, area: w * h }};
    }}

    function drawInteractiveBox(svg, box, payload, rendered, hit = null) {{
      hit = hit || renderedBox(box, rendered);
      if (!hit) return;
      const {{ x, y, w, h }} = hit;
      const color = colorForKind(box.kind);
      const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      group.dataset.correctionId = correctionId(payload, box);
      group.dataset.boxId = box.id || '';
      group.dataset.boxKind = box.kind || '';
      group.dataset.hitArea = String(hit.area || 0);
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', x); rect.setAttribute('y', y); rect.setAttribute('width', w); rect.setAttribute('height', h);
      rect.setAttribute('class', 'overlay-box');
      rect.setAttribute('stroke', color);
      const badge = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      badge.setAttribute('class', 'overlay-badge');
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      const radius = 12;
      circle.setAttribute('cx', x + radius + 2); circle.setAttribute('cy', y + radius + 2); circle.setAttribute('r', radius);
      circle.setAttribute('fill', color);
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', x + radius + 2); text.setAttribute('y', y + radius + 2);
      text.textContent = String(box.display || '');
      badge.appendChild(circle); badge.appendChild(text);
      group.appendChild(rect); group.appendChild(badge);
      svg.appendChild(group);
      return group;
    }}

    function handleOverlayClick(event, viewer, payload) {{
      event.preventDefault();
      event.stopPropagation();
      const svg = viewer.querySelector('.overlay-svg');
      const point = svgPoint(svg, event);
      const candidates = (viewer.__hitBoxes || [])
        .filter(item => containsPoint(item.hit, point.x, point.y))
        .sort((a, b) => {{
          const priority = boxPriority(a.box) - boxPriority(b.box);
          if (priority !== 0) return priority;
          return a.hit.area - b.hit.area;
        }});
      if (!candidates.length) {{
        const src = payload.image_src;
        if (src) openModal(src, viewer.dataset.pageTitle || '');
        return;
      }}
      const selected = candidates[0].box;
      const shape = viewer.querySelector(`[data-box-id="${{cssEscape(selected.id || '')}}"]`);
      if (shape) selectBox(shape, payload, selected);
    }}

    function svgPoint(svg, event) {{
      const ctm = svg.getScreenCTM();
      if (!ctm) return {{ x: event.offsetX, y: event.offsetY }};
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      return point.matrixTransform(ctm.inverse());
    }}

    function containsPoint(hit, x, y) {{
      return x >= hit.x && x <= hit.x + hit.w && y >= hit.y && y <= hit.y + hit.h;
    }}

    function boxPriority(box) {{
      const priorities = {{
        magi_text: 0,
        ocr_block: 0,
        magi_tail: 1,
        magi_character: 2,
        ocr_group: 3,
        magi_panel: 4
      }};
      return priorities[box.kind] ?? 2;
    }}

    function renderLegend(viewer, boxes, payload) {{
      const legend = viewer.querySelector('.interactive-legend');
      if (!legend) return;
      if (!boxes.length) {{
        legend.innerHTML = '<div class="legend-title"><strong>Leyenda</strong><span>sin cajas visibles</span></div>';
        return;
      }}
      const title = viewer.dataset.viewer === 'magi' ? 'Magi' : 'OCR';
      const items = boxes.map(box => {{
        const id = correctionId(payload, box);
        const color = colorForKind(box.kind);
        const label = legendLabel(box);
        const text = (box.text || box.kind || '').toString().slice(0, 110);
        const corrected = Boolean(corrections[id]);
        return `<button type="button" class="legend-item${{corrected ? ' corrected' : ''}}" data-correction-id="${{escapeAttr(id)}}" data-legend-box-id="${{escapeAttr(box.id || '')}}">
          <span class="legend-swatch" style="background:${{color}}"></span>
          <span class="legend-main"><span class="legend-name">${{escapeHtml(label)}}</span><span class="legend-text">${{escapeHtml(text || 'sin texto')}}</span></span>
          <span class="legend-mark">${{corrected ? 'edit' : ''}}</span>
        </button>`;
      }}).join('');
      legend.innerHTML = `<div class="legend-title"><strong>Leyenda ${{title}}</strong><span>${{boxes.length}} cajas</span></div><div class="legend-list">${{items}}</div>`;
      legend.querySelectorAll('.legend-item').forEach(item => {{
        item.addEventListener('click', () => {{
          const box = boxes.find(candidate => String(candidate.id || '') === item.dataset.legendBoxId);
          if (box) focusLegendBox(viewer, payload, box);
        }});
      }});
    }}

    function legendLabel(box) {{
      const display = box.display ? `#${{box.display}}` : '';
      const labels = {{
        magi_panel: 'Panel',
        magi_text: 'Texto Magi',
        magi_character: 'Personaje',
        magi_tail: 'Cola',
        ocr_group: 'Grupo OCR',
        ocr_block: 'Bloque OCR'
      }};
      return `${{display}} ${{labels[box.kind] || box.kind || 'Caja'}}`.trim();
    }}

    function focusLegendBox(viewer, payload, box) {{
      const shape = viewer.querySelector(`[data-box-id="${{cssEscape(box.id || '')}}"]`);
      if (shape) {{
        selectBox(shape, payload, box);
        shape.classList.add('selected-overlay');
        shape.scrollIntoView({{ behavior: 'smooth', block: 'center', inline: 'center' }});
      }}
      viewer.querySelectorAll('.legend-item').forEach(item => item.classList.remove('active'));
      const legendItem = viewer.querySelector(`[data-legend-box-id="${{cssEscape(box.id || '')}}"]`);
      if (legendItem) legendItem.classList.add('active');
    }}

    function colorForKind(kind) {{
      if (kind === 'magi_panel') return '#28be5f';
      if (kind === 'magi_text') return '#1e78f0';
      if (kind === 'magi_character') return '#e64141';
      if (kind === 'magi_tail') return '#be4bd2';
      if (kind === 'ocr_group') return '#00aa5a';
      return '#ffb81c';
    }}

    function selectBox(shape, payload, box) {{
      if (selectedShape) selectedShape.classList.remove('selected-overlay');
      selectedShape = shape;
      selectedShape.classList.add('selected-overlay');
      selectedPayload = {{ payload, box, id: correctionId(payload, box) }};
      document.querySelectorAll('.legend-item.active').forEach(item => item.classList.remove('active'));
      const ownerViewer = shape.closest('.interactive-viewer');
      const activeLegend = ownerViewer?.querySelector(`[data-legend-box-id="${{cssEscape(box.id || '')}}"]`);
      if (activeLegend) activeLegend.classList.add('active');
      const existing = corrections[selectedPayload.id] || {{}};
      correctionLabel.value = existing.label || 'correct';
      correctionText.value = existing.corrected_text || existing.note || suggestedText(box) || box.text || '';
      selectedBoxInfo.textContent = `${{payload.comic_id || ''}}/${{payload.file_name || ''}} - ${{box.kind}} #${{box.display || ''}}`;
      renderBoxComparison(box);
      correctionPanel.classList.add('open');
    }}

    function suggestedText(box) {{
      return box.fusion?.text || null;
    }}

    function renderBoxComparison(box) {{
      const container = document.getElementById('boxComparison');
      if (!container) return;
      const fusion = box.fusion;
      if (!fusion) {{
        container.innerHTML = '<div class="comparison-empty">Sin comparacion fusionada para esta caja.</div>';
        return;
      }}
      const flags = (fusion.review_flags || []).map(flag => `<span class="review-flag">${{escapeHtml(flag)}}</span>`).join('');
      const options = (fusion.options || []).map(option => `
        <div class="comparison-option">
          <strong>${{escapeHtml(option.source || 'unknown')}}</strong>
          <span>${{escapeHtml(option.text || '')}}</span>
          <small>conf: ${{option.confidence == null ? '--' : Number(option.confidence).toFixed(3)}}</small>
        </div>
      `).join('');
      container.innerHTML = `
        <div class="comparison-suggestion"><strong>Sugerencia:</strong> ${{escapeHtml(fusion.text || '')}}</div>
        <div class="comparison-source">fuente: ${{escapeHtml(fusion.source || '')}}</div>
        <div class="comparison-flags">${{flags}}</div>
        <div class="comparison-options">${{options}}</div>
      `;
    }}

    function correctionId(payload, box) {{
      return [payload.comic_id, payload.file_name, box.id].join('::');
    }}

    function saveCurrentCorrection() {{
      if (!selectedPayload) return;
      const {{ payload, box, id }} = selectedPayload;
      corrections[id] = {{
        id,
        comic_id: payload.comic_id,
        file_name: payload.file_name,
        page_id: payload.page_id,
        target_id: box.id,
        target_kind: box.kind,
        display: box.display,
        raw_text: box.text || null,
        suggested_text: suggestedText(box) || null,
        fusion: box.fusion || null,
        alternatives: box.fusion?.options || [],
        confidence: box.confidence ?? null,
        box: box.box || null,
        label: correctionLabel.value,
        corrected_text: correctionText.value,
        source_payload: {{
          tool: box.kind?.startsWith('magi') ? 'magi' : 'paddle_or_fusion',
          attributes: box.attributes || null,
          block_index: box.block_index ?? null,
          block_indices: box.block_indices || null,
          linked_targets: box.linked_targets || null
        }},
        updated_at: new Date().toISOString()
      }};
      persistCorrections();
      refreshCorrectionMarks(document);
    }}

    function clearCurrentCorrection() {{
      if (!selectedPayload) return;
      delete corrections[selectedPayload.id];
      persistCorrections();
      refreshCorrectionMarks(document);
    }}

    function refreshCorrectionMarks(root) {{
      root.querySelectorAll('[data-correction-id]').forEach(shape => {{
        const hasCorrection = Boolean(corrections[shape.dataset.correctionId]);
        shape.classList.toggle('corrected', hasCorrection);
        const mark = shape.querySelector?.('.legend-mark');
        if (mark) mark.textContent = hasCorrection ? 'edit' : '';
      }});
    }}

    function syncGlobalMagiLayers() {{
      const enabled = new Set(Array.from(document.querySelectorAll('[data-global-magi-layer]:checked')).map(input => input.dataset.globalMagiLayer));
      document.querySelectorAll('.interactive-viewer[data-viewer="magi"]').forEach(viewer => {{
        viewer.querySelectorAll('input[data-layer]').forEach(input => {{
          input.checked = enabled.has(input.dataset.layer);
        }});
        const card = viewer.closest('.card');
        const payloadTag = card?.querySelector('.page-payload');
        if (payloadTag) renderViewer(viewer, JSON.parse(payloadTag.textContent || '{{}}'));
      }});
    }}

    function syncGlobalOcrLayers() {{
      const enabled = new Set(Array.from(document.querySelectorAll('[data-global-ocr-layer]:checked')).map(input => input.dataset.globalOcrLayer));
      document.querySelectorAll('.interactive-viewer[data-viewer="ocr"]').forEach(viewer => {{
        viewer.querySelectorAll('input[data-ocr-layer]').forEach(input => {{
          input.checked = enabled.has(input.dataset.ocrLayer);
        }});
        const card = viewer.closest('.card');
        const payloadTag = card?.querySelector('.page-payload');
        if (payloadTag) renderViewer(viewer, JSON.parse(payloadTag.textContent || '{{}}'));
      }});
    }}

    function applyGlobalMagiPreset(mode) {{
      document.querySelectorAll('[data-global-magi-layer]').forEach(input => {{
        input.checked = mode === 'all';
      }});
      syncGlobalMagiLayers();
    }}

    function applyGlobalOcrPreset(mode) {{
      document.querySelectorAll('[data-global-ocr-layer]').forEach(input => {{
        input.checked = mode === 'all';
      }});
      syncGlobalOcrLayers();
    }}

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, char => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }}[char]));
    }}

    function escapeAttr(value) {{
      return escapeHtml(value);
    }}

    function cssEscape(value) {{
      const text = String(value ?? '');
      if (window.CSS && CSS.escape) return CSS.escape(text);
      return text.replace(/["\\\\]/g, '\\\\$&');
    }}

    function loadCorrections() {{
      try {{ return JSON.parse(localStorage.getItem('comic_ocr_corrections') || '{{}}'); }}
      catch {{ return {{}}; }}
    }}

    function persistCorrections() {{
      localStorage.setItem('comic_ocr_corrections', JSON.stringify(corrections));
    }}

    function downloadCorrections() {{
      const payload = {{
        schema_version: 'comic_ocr_corrections.v1',
        exported_at: new Date().toISOString(),
        corrections: Object.values(corrections)
      }};
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'comic_ocr_corrections.json';
      a.click();
      URL.revokeObjectURL(url);
    }}
  </script>
</body>
</html>
"""


def group_pages_by_comic(pages: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        comic_id = str(page.get("comic_id") or "unknown")
        groups.setdefault(comic_id, []).append(page)
    return dict(sorted(groups.items(), key=lambda item: item[0].lower()))


def render_comic_section(
    comic_id: str,
    pages: list[dict[str, Any]],
    output_html: Path,
) -> str:
    page_count = len(pages)
    type_counts = Counter(page["page_type"]["page_type"] for page in pages)
    type_text = ", ".join(f"{key}: {value}" for key, value in sorted(type_counts.items()))
    rows = "\n".join(render_page_card(page, output_html) for page in pages)
    return f"""
<section class="comic-section" id="{escape(html_id(comic_id))}" data-comic="{escape(comic_id)}" data-page-count="{page_count}">
  <div class="comic-header">
    <div>
      <h2>{escape(comic_id)}</h2>
      <div class="meta">{escape(type_text)}</div>
    </div>
    <div class="meta section-visible-count">{page_count} / {page_count} paginas</div>
  </div>
  <div class="comic-pages">{rows}</div>
</section>
"""


def html_id(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "-" for char in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return f"comic-{safe or 'unknown'}"


def render_page_card(page: dict[str, Any], output_html: Path) -> str:
    title = f"{page.get('comic_id') or 'unknown'} / {page.get('file_name')}"
    comic_id = str(page.get("comic_id") or "unknown")
    page_type = page["page_type"]
    ocr = page.get("ocr", {})
    ocr_groups = page.get("ocr_groups") or []
    ocr_fusion = page.get("ocr_fusion") or []
    number_text = ", ".join(
        f"{candidate['value']} ({candidate['score']})"
        for candidate in page.get("page_number_candidates") or []
    ) or "none"
    fallback_visuals = "\n".join(
        render_visual(label, description, page["visuals"].get(key), output_html, title)
        for key, label, description in (
            ("magi_boxes", "Magi render", "respaldo imagen generada"),
            ("ocr_groups", "OCR render", "respaldo agrupado"),
        )
    )
    interactive_payload_for_html = dict(page.get("interactive") or {})
    image_abs_path = page.get("image_abs_path")
    if image_abs_path and Path(image_abs_path).exists():
        interactive_payload_for_html["image_src"] = relative_path(Path(image_abs_path), output_html)
    interactive_json = json_script_escape(json.dumps(interactive_payload_for_html, ensure_ascii=False))
    search_text = " ".join(
        [
            str(page.get("comic_id") or ""),
            str(page.get("file_name") or ""),
            str(page_type.get("page_type") or ""),
            " ".join(str(group.get("text") or "") for group in ocr_groups),
            " ".join(str(item.get("text") or "") for item in ocr_fusion),
        ]
    ).lower()
    file_index = page_type.get("signals", {}).get("file_index")
    if file_index is None:
        file_index = 0
    return f"""
<section class="card"
  data-comic="{escape(comic_id)}"
  data-page-type="{escape(page_type["page_type"])}"
  data-has-numbers="{'1' if page.get("page_number_candidates") else '0'}"
  data-ocr-blocks="{int(ocr.get("paddle_text_blocks") or 0)}"
  data-ocr-groups="{int(ocr.get("ocr_group_count") or 0)}"
    data-magi-texts="{int(ocr.get("magi_text_regions") or 0)}"
  data-ocr-fusions="{len(ocr_fusion)}"
  data-file-index="{int(file_index)}"
  data-search="{escape(search_text)}">
  <div class="card-header">
    <div>
      <strong>{escape(title)}</strong>
      <div class="meta">type: {escape(page_type["page_type"])} ({page_type["confidence"]})</div>
    </div>
    <div class="meta">page numbers: {escape(number_text)}</div>
  </div>
  <script type="application/json" class="page-payload">{interactive_json}</script>
  <div class="interactive-review">
    {render_interactive_viewer("Magi interactivo", "magi", title)}
    {render_interactive_viewer("OCR interactivo", "ocr", title)}
  </div>
  <details class="fallback-visuals">
    <summary>Ver renders estaticos</summary>
    <div class="visuals">{fallback_visuals}</div>
  </details>
  <div class="details">
    {render_counts_panel(page.get("counts", {}))}
    {render_ocr_panel(page.get("ocr", {}), ocr_fusion)}
    {render_notes_panel(page_type.get("reasons", []), ocr_groups, ocr_fusion)}
  </div>
</section>
"""


def render_counts_panel(counts: dict[str, Any]) -> str:
    rows = [
        ("Paneles Magi", counts.get("panels")),
        ("Textos Magi", counts.get("texts")),
        ("OCR Magi textos", counts.get("magi_ocr_texts")),
        ("Personajes", counts.get("characters")),
        ("Colas de globo", counts.get("tails")),
        ("Texto-personaje", counts.get("text_character_associations")),
        ("Texto-cola", counts.get("text_tail_associations")),
    ]
    return render_definition_panel("Magi", rows)


def render_interactive_viewer(title: str, mode: str, page_title: str) -> str:
    if mode == "magi":
        controls = """
          <label><input type="checkbox" data-layer="panels" checked> Paneles</label>
          <label><input type="checkbox" data-layer="texts" checked> Textos</label>
          <label><input type="checkbox" data-layer="characters" checked> Personajes</label>
          <label><input type="checkbox" data-layer="tails" checked> Colas</label>
        """
    else:
        controls = """
          <label><input type="checkbox" data-ocr-layer="blocks" checked> Bloques</label>
          <label><input type="checkbox" data-ocr-layer="groups"> Grupos</label>
        """
    return f"""
<div class="interactive-viewer" data-viewer="{escape(mode)}" data-page-title="{escape(page_title)}">
  <div class="viewer-head">
    <strong>{escape(title)}</strong>
    <div class="viewer-controls">{controls}</div>
  </div>
  <div class="viewer-body">
    <div class="canvas-wrap">
      <img class="clean-page" alt="{escape(page_title)}" draggable="false">
      <svg class="overlay-svg"></svg>
    </div>
    <aside class="interactive-legend" aria-label="Leyenda interactiva"></aside>
  </div>
</div>
"""


def render_ocr_panel(ocr: dict[str, Any], ocr_fusion: list[dict[str, Any]]) -> str:
    rows = [
        ("Bloques PaddleOCR", ocr.get("paddle_text_blocks")),
        ("Grupos OCR", ocr.get("ocr_group_count")),
        ("Fusiones OCR", len(ocr_fusion)),
        ("Regiones Magi texto", ocr.get("magi_text_regions")),
        ("Coincidencias", ocr.get("matched_regions")),
        ("Confianza media", round(float(ocr.get("paddle_avg_confidence") or 0), 3)),
        ("Error OCR", ocr.get("paddle_error") or "none"),
    ]
    return render_definition_panel("OCR", rows)


def render_definition_panel(title: str, rows: list[tuple[str, Any]]) -> str:
    content = "".join(
        f"<dt>{escape(label)}</dt><dd>{escape(value if value is not None else '--')}</dd>"
        for label, value in rows
    )
    return f'<div class="info-panel"><h5>{escape(title)}</h5><dl>{content}</dl></div>'


def render_notes_panel(
    reasons: list[str],
    groups: list[dict[str, Any]],
    fusion: list[dict[str, Any]] | None = None,
) -> str:
    reason_items = "".join(f"<li>{escape(reason)}</li>" for reason in (reasons or ["sin alertas fuertes"]))
    group_items = "".join(
        f"<li><strong>{escape(group.get('display_index'))}</strong>: {escape(str(group.get('text') or '')[:90])}</li>"
        for group in groups[:5]
    )
    if not group_items:
        group_items = "<li>sin grupos OCR disponibles</li>"
    fusion_items = "".join(
        f"<li><strong>{escape(item.get('display_index'))}</strong>: {escape(str(item.get('text') or '')[:90])}</li>"
        for item in (fusion or [])[:5]
    )
    if not fusion_items:
        fusion_items = "<li>sin fusion OCR disponible</li>"
    return (
        '<div class="info-panel"><h5>Revision</h5>'
        f'<ul class="note-list">{reason_items}</ul>'
        '<div class="group-preview"><strong>Primeros grupos OCR</strong>'
        f'<ul class="note-list">{group_items}</ul></div>'
        '<div class="group-preview"><strong>Sugerencias fusionadas</strong>'
        f'<ul class="note-list">{fusion_items}</ul></div></div>'
    )


def compact_groups_for_html(groups: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    compact = []
    for group in groups[:limit]:
        compact.append(
            {
                "id": group.get("display_index"),
                "text": group.get("text"),
                "blocks": group.get("block_indices"),
            }
        )
    return compact


def render_visual(
    label: str,
    description: str,
    path: str | None,
    output_html: Path,
    page_title: str,
) -> str:
    if not path:
        return (
            f'<div class="visual"><h4>{escape(label)} <small>{escape(description)}</small></h4>'
            "<pre>not available</pre></div>"
        )
    target = Path(path)
    src = Path(os.path.relpath(target.resolve(), output_html.parent.resolve())).as_posix()
    title = f"{page_title} - {label}"
    return (
        f'<div class="visual"><h4>{escape(label)} <small>{escape(description)}</small></h4>'
        f'<a class="visual-link" href="{escape(src)}" data-title="{escape(title)}">'
        f'<img src="{escape(src)}" loading="lazy" alt="{escape(title)}"></a></div>'
    )


def relative_path(path: Path | None, output_html: Path) -> str | None:
    if path is None:
        return None
    return Path(os.path.relpath(path.resolve(), output_html.parent.resolve())).as_posix()


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def json_script_escape(value: str) -> str:
    return (
        value.replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


if __name__ == "__main__":
    main()
