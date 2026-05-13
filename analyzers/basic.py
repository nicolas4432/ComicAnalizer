from __future__ import annotations

from typing import Any

from analyzers.base import Analyzer
from core.data import PipelineResult


class BasicDatasetAnalyzer(Analyzer):
    """Small read-only analyzer used to prove the plugin boundary."""

    name = "basic_dataset"

    def run(self, data: PipelineResult) -> dict[str, Any]:
        page_count = len(data.pages)
        relation_count = len(data.relations)
        cluster_sizes = [len(cluster.page_ids) for cluster in data.clusters]
        return {
            "page_count": page_count,
            "relation_count": relation_count,
            "cluster_count": len(data.clusters),
            "cluster_sizes": cluster_sizes,
            "ordered_page_count": len(data.order),
        }

