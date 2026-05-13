from __future__ import annotations

from typing import Any, Protocol

from core.data import (
    AnomalyReport,
    ClusterInfo,
    PageFeatures,
    PageInput,
    PipelineResult,
    Relation,
)


class ImageIngestor(Protocol):
    def load(self, input_path: str) -> list[PageInput]:
        ...


class FeatureExtractor(Protocol):
    def extract(self, page: PageInput) -> PageFeatures:
        ...


class RelationScorer(Protocol):
    def score(self, source: PageFeatures, target: PageFeatures) -> Relation:
        ...


class NarrativeGraphBuilder(Protocol):
    def build(self, pages: list[PageFeatures], relations: list[Relation]) -> Any:
        ...


class OrderingStrategy(Protocol):
    def order(self, pages: list[PageFeatures], relations: list[Relation]) -> list[str]:
        ...


class Validator(Protocol):
    def validate(
        self,
        pages: list[PageFeatures],
        relations: list[Relation],
        order: list[str],
    ) -> tuple[list[ClusterInfo], AnomalyReport]:
        ...


class ReportWriter(Protocol):
    def write(self, result: PipelineResult, output_path: str) -> None:
        ...

