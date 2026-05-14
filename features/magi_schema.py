from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, float]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "area": self.area,
        }


@dataclass(frozen=True)
class MagiRegion:
    id: str
    kind: str
    index: int
    box: BoundingBox
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "index": self.index,
            "box": self.box.to_dict(),
            "attributes": self.attributes,
        }


@dataclass(frozen=True)
class MagiAssociation:
    kind: str
    source_id: str
    target_id: str
    source_index: int
    target_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source_index": self.source_index,
            "target_index": self.target_index,
        }


@dataclass
class MagiPageAnalysis:
    page_id: str
    path: str
    file_name: str
    comic_id: str | None
    image_sha256: str | None
    task: str
    cache_hit: bool | None
    elapsed_seconds: float | None
    panels: list[MagiRegion] = field(default_factory=list)
    texts: list[MagiRegion] = field(default_factory=list)
    characters: list[MagiRegion] = field(default_factory=list)
    tails: list[MagiRegion] = field(default_factory=list)
    character_cluster_labels: list[int] = field(default_factory=list)
    text_character_associations: list[MagiAssociation] = field(default_factory=list)
    text_tail_associations: list[MagiAssociation] = field(default_factory=list)
    is_essential_text: list[bool] = field(default_factory=list)
    raw_counts: dict[str, int] = field(default_factory=dict)

    @property
    def panel_count(self) -> int:
        return len(self.panels)

    @property
    def text_count(self) -> int:
        return len(self.texts)

    @property
    def character_count(self) -> int:
        return len(self.characters)

    @property
    def tail_count(self) -> int:
        return len(self.tails)

    def to_dict(self, include_regions: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "page_id": self.page_id,
            "path": self.path,
            "file_name": self.file_name,
            "comic_id": self.comic_id,
            "image_sha256": self.image_sha256,
            "task": self.task,
            "cache_hit": self.cache_hit,
            "elapsed_seconds": self.elapsed_seconds,
            "counts": {
                "panels": self.panel_count,
                "texts": self.text_count,
                "characters": self.character_count,
                "tails": self.tail_count,
                "text_character_associations": len(self.text_character_associations),
                "text_tail_associations": len(self.text_tail_associations),
            },
            "character_cluster_labels": self.character_cluster_labels,
            "is_essential_text": self.is_essential_text,
            "text_character_associations": [
                item.to_dict() for item in self.text_character_associations
            ],
            "text_tail_associations": [item.to_dict() for item in self.text_tail_associations],
            "raw_counts": self.raw_counts,
        }
        if include_regions:
            data.update(
                {
                    "panels": [item.to_dict() for item in self.panels],
                    "texts": [item.to_dict() for item in self.texts],
                    "characters": [item.to_dict() for item in self.characters],
                    "tails": [item.to_dict() for item in self.tails],
                }
            )
        return data


def normalize_magi_page(
    result: dict[str, Any],
    metric: dict[str, Any] | None = None,
    index: int = 0,
) -> MagiPageAnalysis:
    detections = result.get("detections") or {}
    path = str(result.get("path") or (metric or {}).get("image_path") or "")
    comic_id, file_name = infer_comic_and_file(path, metric)
    page_id = build_page_id(comic_id=comic_id, file_name=file_name, index=index)

    character_cluster_labels = [
        int(item) for item in detections.get("character_cluster_labels") or []
    ]
    is_essential_text = [bool(item) for item in detections.get("is_essential_text") or []]

    characters = build_regions(
        "character",
        detections.get("characters"),
        page_id,
        extra_by_index={
            idx: {"cluster_label": label}
            for idx, label in enumerate(character_cluster_labels)
        },
    )
    texts = build_regions(
        "text",
        detections.get("texts"),
        page_id,
        extra_by_index={
            idx: {"is_essential": is_essential}
            for idx, is_essential in enumerate(is_essential_text)
        },
    )
    panels = build_regions("panel", detections.get("panels"), page_id)
    tails = build_regions("tail", detections.get("tails"), page_id)

    return MagiPageAnalysis(
        page_id=page_id,
        path=path,
        file_name=file_name,
        comic_id=comic_id,
        image_sha256=(metric or {}).get("image_sha256"),
        task=str((metric or {}).get("task") or "unknown"),
        cache_hit=(metric or {}).get("cache_hit"),
        elapsed_seconds=(metric or {}).get("elapsed_seconds"),
        panels=panels,
        texts=texts,
        characters=characters,
        tails=tails,
        character_cluster_labels=character_cluster_labels,
        text_character_associations=build_associations(
            "text_character",
            detections.get("text_character_associations"),
            source_prefix=f"{page_id}:text",
            target_prefix=f"{page_id}:character",
        ),
        text_tail_associations=build_associations(
            "text_tail",
            detections.get("text_tail_associations"),
            source_prefix=f"{page_id}:text",
            target_prefix=f"{page_id}:tail",
        ),
        is_essential_text=is_essential_text,
        raw_counts={
            key: len(value) for key, value in detections.items() if isinstance(value, list)
        },
    )


def build_regions(
    kind: str,
    value: Any,
    page_id: str,
    extra_by_index: dict[int, dict[str, Any]] | None = None,
) -> list[MagiRegion]:
    regions: list[MagiRegion] = []
    for index, item in enumerate(value or []):
        box = coerce_box(item)
        if box is None:
            continue
        attributes = (extra_by_index or {}).get(index, {})
        regions.append(
            MagiRegion(
                id=f"{page_id}:{kind}:{index}",
                kind=kind,
                index=index,
                box=box,
                attributes=dict(attributes),
            )
        )
    return regions


def build_associations(
    kind: str,
    value: Any,
    source_prefix: str,
    target_prefix: str,
) -> list[MagiAssociation]:
    associations: list[MagiAssociation] = []
    for item in value or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            source_index = int(item[0])
            target_index = int(item[1])
        except (TypeError, ValueError):
            continue
        associations.append(
            MagiAssociation(
                kind=kind,
                source_id=f"{source_prefix}:{source_index}",
                target_id=f"{target_prefix}:{target_index}",
                source_index=source_index,
                target_index=target_index,
            )
        )
    return associations


def coerce_box(value: Any) -> BoundingBox | None:
    if isinstance(value, dict):
        for key in ("box", "bbox", "bboxes"):
            if key in value:
                return coerce_box(value[key])
        return None
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value[idx]) for idx in range(4)]
    except (TypeError, ValueError):
        return None
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return BoundingBox(x1=left, y1=top, x2=right, y2=bottom)


def infer_comic_and_file(
    path: str,
    metric: dict[str, Any] | None = None,
) -> tuple[str | None, str]:
    metric = metric or {}
    comic_id = metric.get("comic_id")
    file_name = ""
    if path:
        normalized = path.replace("\\", "/")
        parts = PurePosixPath(normalized).parts
        file_name = parts[-1] if parts else ""
        if not comic_id and "by_comic" in parts:
            by_comic_index = parts.index("by_comic")
            if by_comic_index + 1 < len(parts):
                comic_id = parts[by_comic_index + 1]
    if not file_name and metric.get("image_path"):
        file_name = PureWindowsPath(str(metric["image_path"])).name
    return str(comic_id) if comic_id else None, file_name


def build_page_id(comic_id: str | None, file_name: str, index: int) -> str:
    stem = file_name.rsplit(".", 1)[0] if file_name else f"{index:04d}"
    if comic_id:
        return f"{comic_id}:{stem}"
    return stem
