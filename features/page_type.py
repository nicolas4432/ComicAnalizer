from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from features.magi_schema import MagiPageAnalysis


CREDIT_KEYWORDS = {
    "translation",
    "translator",
    "translated",
    "typeset",
    "typesetter",
    "redraw",
    "redrawer",
    "proofread",
    "proofreader",
    "scanlation",
    "cleaner",
    "tl:",
}
TITLE_KEYWORDS = {"episode", "chapter", "title", "prologue"}
AD_KEYWORDS = {"patreon", "subscribe", "follow", "discord", "twitter", "instagram"}


@dataclass(frozen=True)
class PageTypePrediction:
    page_type: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_type": self.page_type,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "signals": self.signals,
        }


def classify_page_type(
    page: MagiPageAnalysis | None,
    comparison: dict[str, Any] | None = None,
    page_number_candidates: list[dict[str, Any]] | None = None,
) -> PageTypePrediction:
    comparison = comparison or {}
    text = collect_text(comparison)
    lower_text = text.lower()
    file_index = file_index_from_name(page.file_name if page else comparison.get("file_name", ""))
    signals = build_signals(page, comparison, lower_text, file_index, page_number_candidates or [])

    if is_likely_noise(signals):
        return prediction("noise_or_blank", 0.78, signals, ["very_low_detected_content"])
    if has_keywords(lower_text, CREDIT_KEYWORDS) and signals["panel_count"] <= 2:
        return prediction("credits", 0.82, signals, ["credit_keywords", "low_panel_count"])
    if has_keywords(lower_text, AD_KEYWORDS) and signals["character_count"] <= 2:
        return prediction("ad_or_social", 0.7, signals, ["social_or_ad_keywords"])
    if is_likely_cover(signals, lower_text):
        reasons = ["early_page", "low_panel_count"]
        if has_keywords(lower_text, TITLE_KEYWORDS):
            reasons.append("title_or_episode_keyword")
        return prediction("cover_or_title", 0.76, signals, reasons)
    if is_text_heavy(signals):
        return prediction("text_heavy", 0.68, signals, ["many_ocr_blocks", "few_characters"])
    if is_likely_interior(signals):
        return prediction("interior_story", 0.82, signals, ["story_layout_signals"])
    return prediction("unknown", 0.45, signals, ["weak_or_mixed_signals"])


def build_signals(
    page: MagiPageAnalysis | None,
    comparison: dict[str, Any],
    lower_text: str,
    file_index: int | None,
    page_number_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    panel_count = page.panel_count if page else int(comparison.get("magi_panel_count") or 0)
    text_count = page.text_count if page else int(comparison.get("magi_text_regions") or 0)
    character_count = page.character_count if page else int(comparison.get("magi_character_count") or 0)
    tail_count = page.tail_count if page else int(comparison.get("magi_tail_count") or 0)
    paddle_blocks = int(comparison.get("paddle_text_blocks") or 0)
    avg_confidence = comparison.get("paddle_avg_confidence")
    return {
        "file_index": file_index,
        "panel_count": panel_count,
        "text_count": text_count,
        "character_count": character_count,
        "tail_count": tail_count,
        "paddle_text_blocks": paddle_blocks,
        "ocr_group_count": int(comparison.get("ocr_group_count") or 0),
        "paddle_avg_confidence": avg_confidence,
        "has_page_number_candidate": bool(page_number_candidates),
        "credit_keyword_count": keyword_count(lower_text, CREDIT_KEYWORDS),
        "title_keyword_count": keyword_count(lower_text, TITLE_KEYWORDS),
        "ad_keyword_count": keyword_count(lower_text, AD_KEYWORDS),
    }


def collect_text(comparison: dict[str, Any]) -> str:
    parts = [str(comparison.get("paddle_text_preview") or "")]
    for group in comparison.get("ocr_groups") or []:
        parts.append(str(group.get("text") or ""))
    return "\n".join(part for part in parts if part)


def is_likely_noise(signals: dict[str, Any]) -> bool:
    avg_conf = signals.get("paddle_avg_confidence")
    low_conf = isinstance(avg_conf, (int, float)) and float(avg_conf) < 0.35
    return (
        signals["panel_count"] == 0
        and signals["text_count"] == 0
        and signals["character_count"] == 0
    ) or (
        signals["panel_count"] <= 1
        and signals["character_count"] == 0
        and signals["paddle_text_blocks"] <= 2
        and low_conf
    )


def is_likely_cover(signals: dict[str, Any], lower_text: str) -> bool:
    early = signals["file_index"] is not None and signals["file_index"] <= 1
    sparse_story = signals["panel_count"] <= 1 and signals["tail_count"] <= 1
    title_like = signals["title_keyword_count"] > 0 or bool(re.search(r"\bepisode\s*\d+", lower_text))
    return early and sparse_story and (signals["text_count"] <= 4 or title_like)


def is_text_heavy(signals: dict[str, Any]) -> bool:
    return (
        signals["paddle_text_blocks"] >= 35
        and signals["character_count"] <= 2
        and signals["panel_count"] <= 3
    )


def is_likely_interior(signals: dict[str, Any]) -> bool:
    return (
        signals["panel_count"] >= 2
        and (signals["character_count"] >= 2 or signals["text_count"] >= 2)
    ) or (
        signals["tail_count"] >= 2 and signals["text_count"] >= 2
    )


def prediction(
    page_type: str,
    confidence: float,
    signals: dict[str, Any],
    reasons: list[str],
) -> PageTypePrediction:
    return PageTypePrediction(
        page_type=page_type,
        confidence=confidence,
        reasons=reasons,
        signals=signals,
    )


def has_keywords(text: str, keywords: set[str]) -> bool:
    return keyword_count(text, keywords) > 0


def keyword_count(text: str, keywords: set[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def file_index_from_name(file_name: str | None) -> int | None:
    if not file_name:
        return None
    stem = Path(file_name).stem
    match = re.search(r"(\d+)$", stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None
