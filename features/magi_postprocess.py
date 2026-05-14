from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from features.magi_schema import MagiPageAnalysis


@dataclass(frozen=True)
class QualityFlag:
    code: str
    severity: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass
class PageQualityReport:
    page_id: str
    comic_id: str | None
    file_name: str
    flags: list[QualityFlag] = field(default_factory=list)

    @property
    def suspicious(self) -> bool:
        return any(flag.severity in {"warning", "critical"} for flag in self.flags)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "comic_id": self.comic_id,
            "file_name": self.file_name,
            "suspicious": self.suspicious,
            "flags": [flag.to_dict() for flag in self.flags],
        }


@dataclass(frozen=True)
class MagiQualityThresholds:
    dense_text_min: int = 5
    dense_character_min: int = 4
    very_dense_character_min: int = 10
    many_text_min: int = 8
    many_tail_min: int = 4
    slow_page_seconds: float = 10.0
    weak_association_ratio: float = 0.6


def evaluate_page_quality(
    page: MagiPageAnalysis,
    thresholds: MagiQualityThresholds | None = None,
) -> PageQualityReport:
    thresholds = thresholds or MagiQualityThresholds()
    flags: list[QualityFlag] = []

    if page.panel_count == 0:
        flags.append(
            QualityFlag(
                code="no_panels",
                severity="critical",
                message="Magi did not detect any panel.",
                evidence=counts(page),
            )
        )
    elif page.panel_count == 1 and (
        page.text_count >= thresholds.dense_text_min
        or page.character_count >= thresholds.dense_character_min
    ):
        flags.append(
            QualityFlag(
                code="single_panel_dense_page",
                severity="warning",
                message="Only one panel was detected despite dense text or character content.",
                evidence=counts(page),
            )
        )

    if page.panel_count <= 2 and page.character_count >= thresholds.very_dense_character_min:
        flags.append(
            QualityFlag(
                code="dense_characters_low_panel_count",
                severity="warning",
                message="Many characters were detected but the panel count is low.",
                evidence=counts(page),
            )
        )

    if page.text_count == 0:
        flags.append(
            QualityFlag(
                code="no_text_regions",
                severity="info",
                message="Magi did not detect text regions.",
                evidence=counts(page),
            )
        )

    if page.character_count == 0:
        flags.append(
            QualityFlag(
                code="no_characters",
                severity="info",
                message="Magi did not detect characters.",
                evidence=counts(page),
            )
        )

    if len(page.character_cluster_labels) != page.character_count:
        flags.append(
            QualityFlag(
                code="character_cluster_label_mismatch",
                severity="warning",
                message="Character cluster labels do not match character detections.",
                evidence={
                    "characters": page.character_count,
                    "character_cluster_labels": len(page.character_cluster_labels),
                },
            )
        )

    if len(page.is_essential_text) != page.text_count:
        flags.append(
            QualityFlag(
                code="essential_text_label_mismatch",
                severity="warning",
                message="Essential-text labels do not match text detections.",
                evidence={
                    "texts": page.text_count,
                    "is_essential_text": len(page.is_essential_text),
                },
            )
        )

    flags.extend(association_quality_flags(page, thresholds))

    if page.elapsed_seconds is not None and page.elapsed_seconds >= thresholds.slow_page_seconds:
        flags.append(
            QualityFlag(
                code="slow_magi_page",
                severity="info",
                message="Magi inference was slow for this page.",
                evidence={"elapsed_seconds": page.elapsed_seconds},
            )
        )

    return PageQualityReport(
        page_id=page.page_id,
        comic_id=page.comic_id,
        file_name=page.file_name,
        flags=flags,
    )


def association_quality_flags(
    page: MagiPageAnalysis,
    thresholds: MagiQualityThresholds,
) -> list[QualityFlag]:
    flags: list[QualityFlag] = []
    text_tail_sources = {item.source_index for item in page.text_tail_associations}
    text_character_sources = {item.source_index for item in page.text_character_associations}
    tail_targets = {item.target_index for item in page.text_tail_associations}

    if page.text_count >= thresholds.many_text_min:
        linked_texts = text_tail_sources | text_character_sources
        unlinked_count = page.text_count - len(linked_texts)
        ratio = unlinked_count / page.text_count if page.text_count else 0.0
        if ratio >= thresholds.weak_association_ratio:
            flags.append(
                QualityFlag(
                    code="many_texts_weak_association",
                    severity="warning",
                    message="Many text regions are not associated with tails or characters.",
                    evidence={
                        "texts": page.text_count,
                        "linked_texts": len(linked_texts),
                        "unlinked_texts": unlinked_count,
                        "unlinked_ratio": ratio,
                    },
                )
            )

    if page.tail_count >= thresholds.many_tail_min:
        orphan_tail_count = page.tail_count - len(tail_targets)
        ratio = orphan_tail_count / page.tail_count if page.tail_count else 0.0
        if ratio >= thresholds.weak_association_ratio:
            flags.append(
                QualityFlag(
                    code="many_orphan_tails",
                    severity="warning",
                    message="Many tails are not associated with text regions.",
                    evidence={
                        "tails": page.tail_count,
                        "associated_tails": len(tail_targets),
                        "orphan_tails": orphan_tail_count,
                        "orphan_ratio": ratio,
                    },
                )
            )

    return flags


def build_quality_report(
    pages: list[MagiPageAnalysis],
    thresholds: MagiQualityThresholds | None = None,
) -> dict[str, Any]:
    page_reports = [evaluate_page_quality(page, thresholds) for page in pages]
    flag_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    flags_by_comic: dict[str, Counter[str]] = defaultdict(Counter)

    for report in page_reports:
        comic_id = report.comic_id or "unknown"
        for flag in report.flags:
            flag_counts[flag.code] += 1
            severity_counts[flag.severity] += 1
            flags_by_comic[comic_id][flag.code] += 1

    return {
        "page_count": len(pages),
        "suspicious_page_count": sum(1 for report in page_reports if report.suspicious),
        "flag_counts": dict(flag_counts),
        "severity_counts": dict(severity_counts),
        "flags_by_comic": {
            comic_id: dict(counter) for comic_id, counter in flags_by_comic.items()
        },
        "pages": [report.to_dict() for report in page_reports],
    }


def counts(page: MagiPageAnalysis) -> dict[str, int]:
    return {
        "panels": page.panel_count,
        "texts": page.text_count,
        "characters": page.character_count,
        "tails": page.tail_count,
        "text_character_associations": len(page.text_character_associations),
        "text_tail_associations": len(page.text_tail_associations),
    }
