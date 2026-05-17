from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CORRECTION_SCHEMA_VERSION = "ocr_correction.v1"


@dataclass(frozen=True)
class OCRCorrection:
    evidence_id: str
    comic_id: str
    page_file: str
    block_index: int
    raw_text: str
    corrected_text: str
    is_correct: bool | None = None
    error_type: str | None = None
    is_false_positive: bool | None = None
    belongs_to_bubble: bool | None = None
    group_id: str | None = None
    review_priority: int | None = None
    review_flags: list[str] | None = None
    asset_hint: str | None = None
    notes: str | None = None
    reviewer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CORRECTION_SCHEMA_VERSION,
            "evidence_id": self.evidence_id,
            "comic_id": self.comic_id,
            "page_file": self.page_file,
            "block_index": self.block_index,
            "raw_text": self.raw_text,
            "corrected_text": self.corrected_text,
            "is_correct": self.is_correct,
            "error_type": self.error_type,
            "is_false_positive": self.is_false_positive,
            "belongs_to_bubble": self.belongs_to_bubble,
            "group_id": self.group_id,
            "review_priority": self.review_priority,
            "review_flags": self.review_flags or [],
            "asset_hint": self.asset_hint,
            "notes": self.notes,
            "reviewer": self.reviewer,
        }


def load_corrections(path: str | Path) -> list[OCRCorrection]:
    path = Path(path).expanduser().resolve()
    corrections: list[OCRCorrection] = []
    if not path.exists():
        return corrections
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        corrections.append(parse_correction(payload, line_number=line_number))
    return corrections


def parse_correction(payload: dict[str, Any], line_number: int = 0) -> OCRCorrection:
    required = ["evidence_id", "comic_id", "page_file", "block_index", "raw_text", "corrected_text"]
    missing = [key for key in required if key not in payload]
    if missing:
        where = f" at line {line_number}" if line_number else ""
        raise ValueError(f"Missing OCR correction fields{where}: {missing}")
    return OCRCorrection(
        evidence_id=str(payload["evidence_id"]),
        comic_id=str(payload["comic_id"]),
        page_file=str(payload["page_file"]),
        block_index=int(payload["block_index"]),
        raw_text=str(payload["raw_text"]),
        corrected_text=str(payload["corrected_text"]),
        is_correct=payload.get("is_correct"),
        error_type=payload.get("error_type"),
        is_false_positive=payload.get("is_false_positive"),
        belongs_to_bubble=payload.get("belongs_to_bubble"),
        group_id=payload.get("group_id"),
        review_priority=payload.get("review_priority"),
        review_flags=list(payload.get("review_flags") or []),
        asset_hint=payload.get("asset_hint"),
        notes=payload.get("notes"),
        reviewer=payload.get("reviewer"),
    )


def corrections_by_evidence_id(corrections: list[OCRCorrection]) -> dict[str, OCRCorrection]:
    return {item.evidence_id: item for item in corrections}
