from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PageInput:
    """Raw image discovered during ingestion."""

    page_id: str
    path: Path
    sha256: str


@dataclass
class ImageMetadata:
    width: int
    height: int
    channels: int
    aspect_ratio: float
    mean_color_bgr: list[float]
    std_color_bgr: list[float]
    brightness: float
    file_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "aspect_ratio": self.aspect_ratio,
            "mean_color_bgr": self.mean_color_bgr,
            "std_color_bgr": self.std_color_bgr,
            "brightness": self.brightness,
            "file_size_bytes": self.file_size_bytes,
        }


@dataclass
class TextFeatures:
    text: str = ""
    confidence: float | None = None
    text_embedding: list[float] | None = None
    blocks: list[dict[str, Any]] = field(default_factory=list)
    backend: str = "none"

    def to_dict(self, include_embeddings: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "confidence": self.confidence,
            "blocks": self.blocks,
            "backend": self.backend,
        }
        if include_embeddings:
            data["text_embedding"] = self.text_embedding
        else:
            data["text_embedding_dim"] = (
                len(self.text_embedding) if self.text_embedding is not None else 0
            )
        return data


@dataclass
class LayoutFeatures:
    panels: list[dict[str, Any]] = field(default_factory=list)
    reading_order: list[int] = field(default_factory=list)
    backend: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "panels": self.panels,
            "reading_order": self.reading_order,
            "backend": self.backend,
        }


@dataclass
class PageFeatures:
    page_id: str
    path: str
    sha256: str
    metadata: ImageMetadata
    visual_embedding: list[float]
    visual_backend: str
    text: TextFeatures = field(default_factory=TextFeatures)
    layout: LayoutFeatures = field(default_factory=LayoutFeatures)

    def to_dict(self, include_embeddings: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.page_id,
            "path": self.path,
            "sha256": self.sha256,
            "metadata": self.metadata.to_dict(),
            "visual_backend": self.visual_backend,
            "text": self.text.to_dict(include_embeddings=include_embeddings),
            "layout": self.layout.to_dict(),
        }
        if include_embeddings:
            data["visual_embedding"] = self.visual_embedding
        else:
            data["visual_embedding_dim"] = len(self.visual_embedding)
        return data


@dataclass
class Relation:
    source: str
    target: str
    score: float
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "score": self.score,
            "components": self.components,
        }


@dataclass
class ClusterInfo:
    cluster_id: str
    page_ids: list[str]
    confidence: float
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "page_ids": self.page_ids,
            "confidence": self.confidence,
            "method": self.method,
        }


@dataclass
class AnomalyReport:
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    outliers: list[dict[str, Any]] = field(default_factory=list)
    missing_pages: list[dict[str, Any]] = field(default_factory=list)
    low_confidence_edges: list[dict[str, Any]] = field(default_factory=list)
    skipped_inputs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duplicates": self.duplicates,
            "outliers": self.outliers,
            "missing_pages": self.missing_pages,
            "low_confidence_edges": self.low_confidence_edges,
            "skipped_inputs": self.skipped_inputs,
        }


@dataclass
class PipelineResult:
    pages: list[PageFeatures]
    relations: list[Relation]
    order: list[str]
    clusters: list[ClusterInfo]
    anomalies: AnomalyReport
    analyzer_results: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_embeddings: bool = True) -> dict[str, Any]:
        pages_by_id = {page.page_id: page for page in self.pages}
        ordered_pages = []
        for position, page_id in enumerate(self.order, 1):
            page = pages_by_id.get(page_id)
            if page is None:
                continue
            ordered_pages.append(
                {
                    "position": position,
                    "id": page.page_id,
                    "path": page.path,
                    "sha256": page.sha256,
                }
            )

        return {
            "pages": [
                page.to_dict(include_embeddings=include_embeddings) for page in self.pages
            ],
            "relations": [relation.to_dict() for relation in self.relations],
            "order": self.order,
            "ordered_pages": ordered_pages,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "anomalies": self.anomalies.to_dict(),
            "analysis": self.analyzer_results,
        }
