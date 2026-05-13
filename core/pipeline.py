from __future__ import annotations

from typing import Iterable

from analyzers.base import Analyzer
from core.data import PipelineResult
from core.ingestion import DirectoryImageIngestor
from features.extractor import CompositeFeatureExtractor
from graph.narrative_graph import NetworkXNarrativeGraphBuilder, build_pairwise_relations
from graph.ordering import GreedyNarrativeOrdering
from graph.validation import NarrativeValidator
from models.relation_model import HeuristicTransitionScorer


class ComicNarrativePipeline:
    """End-to-end orchestration layer.

    The pipeline wires modules together but keeps each responsibility isolated:
    ingestion, feature extraction, transition scoring, graph construction,
    ordering, validation, and optional analysis plugins.
    """

    def __init__(
        self,
        config: dict,
        analyzers: Iterable[Analyzer] | None = None,
    ) -> None:
        self.config = config
        ingestion_config = config.get("ingestion", {})
        relation_config = config.get("relation_model", {})
        ordering_config = config.get("ordering", {})

        self.ingestor = DirectoryImageIngestor(
            recursive=ingestion_config.get("recursive", True)
        )
        self.feature_extractor = CompositeFeatureExtractor(config)
        self.scorer = HeuristicTransitionScorer(config)
        self.graph_builder = NetworkXNarrativeGraphBuilder()
        self.ordering = GreedyNarrativeOrdering(
            low_confidence_threshold=ordering_config.get(
                "low_confidence_threshold", 0.45
            )
        )
        self.validator = NarrativeValidator(config)
        self.analyzers = list(analyzers or [])
        self.top_k_edges = relation_config.get("top_k_edges", 8)

    def run(self, input_path: str) -> PipelineResult:
        raw_pages = self.ingestor.load(input_path)
        pages = [self.feature_extractor.extract(page) for page in raw_pages]
        relations = build_pairwise_relations(
            pages=pages,
            scorer=self.scorer,
            top_k_edges=self.top_k_edges,
        )
        self.graph_builder.build(pages, relations)
        order = self.ordering.order(pages, relations)
        clusters, anomalies = self.validator.validate(pages, relations, order)
        result = PipelineResult(
            pages=pages,
            relations=relations,
            order=order,
            clusters=clusters,
            anomalies=anomalies,
        )
        result.analyzer_results = self._run_analyzers(result)
        return result

    def _run_analyzers(self, result: PipelineResult) -> dict:
        outputs = {}
        for analyzer in self.analyzers:
            outputs[analyzer.name] = analyzer.run(result)
        return outputs

