from __future__ import annotations

from collections import defaultdict
from statistics import mean

from core.data import AnomalyReport, ClusterInfo, PageFeatures, Relation
from models.relation_model import _cosine_similarity, _probability_from_cosine


class NarrativeValidator:
    """Basic quality checks for duplicates, outliers, gaps, and clusters."""

    def __init__(self, config: dict) -> None:
        validation_config = config.get("validation", {})
        ordering_config = config.get("ordering", {})
        self.duplicate_similarity_threshold = float(
            validation_config.get("duplicate_similarity_threshold", 0.985)
        )
        self.outlier_quantile = float(validation_config.get("outlier_quantile", 0.1))
        self.cluster_similarity_threshold = float(
            validation_config.get("cluster_similarity_threshold", 0.72)
        )
        self.missing_page_threshold = float(
            validation_config.get("missing_page_threshold", 0.35)
        )
        self.low_confidence_threshold = float(
            ordering_config.get("low_confidence_threshold", 0.45)
        )

    def validate(
        self,
        pages: list[PageFeatures],
        relations: list[Relation],
        order: list[str],
    ) -> tuple[list[ClusterInfo], AnomalyReport]:
        relation_lookup = {
            (relation.source, relation.target): relation for relation in relations
        }
        anomalies = AnomalyReport(
            duplicates=self._detect_duplicates(pages),
            outliers=self._detect_outliers(pages),
            missing_pages=self._detect_missing_pages(order, relation_lookup),
            low_confidence_edges=self._detect_low_confidence_edges(
                order, relation_lookup
            ),
        )
        clusters = self._detect_clusters(pages)
        return clusters, anomalies

    def _detect_duplicates(self, pages: list[PageFeatures]) -> list[dict[str, object]]:
        duplicates: list[dict[str, object]] = []
        sha_groups: dict[str, list[str]] = defaultdict(list)
        for page in pages:
            sha_groups[page.sha256].append(page.page_id)

        for sha256, page_ids in sha_groups.items():
            if len(page_ids) > 1:
                duplicates.append(
                    {"type": "exact_hash", "page_ids": page_ids, "sha256": sha256}
                )

        for left_index, left in enumerate(pages):
            for right in pages[left_index + 1 :]:
                similarity = _probability_from_cosine(
                    _cosine_similarity(left.visual_embedding, right.visual_embedding)
                )
                if similarity >= self.duplicate_similarity_threshold:
                    duplicates.append(
                        {
                            "type": "visual_near_duplicate",
                            "page_ids": [left.page_id, right.page_id],
                            "similarity": similarity,
                        }
                    )
        return duplicates

    def _detect_outliers(self, pages: list[PageFeatures]) -> list[dict[str, object]]:
        if len(pages) < 3:
            return []
        sklearn_outliers = self._detect_outliers_with_sklearn(pages)
        if sklearn_outliers is not None:
            return sklearn_outliers

        avg_similarities: list[tuple[str, float]] = []
        for page in pages:
            similarities: list[float] = []
            for other in pages:
                if page.page_id == other.page_id:
                    continue
                similarities.append(
                    _probability_from_cosine(
                        _cosine_similarity(page.visual_embedding, other.visual_embedding)
                    )
                )
            avg_similarities.append((page.page_id, mean(similarities)))

        sorted_scores = sorted(score for _, score in avg_similarities)
        cutoff_index = max(0, min(len(sorted_scores) - 1, int(len(sorted_scores) * self.outlier_quantile)))
        cutoff = sorted_scores[cutoff_index]
        return [
            {
                "page_id": page_id,
                "reason": "low_average_visual_similarity",
                "average_similarity": score,
            }
            for page_id, score in avg_similarities
            if score <= cutoff
        ]

    def _detect_missing_pages(
        self,
        order: list[str],
        relation_lookup: dict[tuple[str, str], Relation],
    ) -> list[dict[str, object]]:
        gaps: list[dict[str, object]] = []
        for source, target in zip(order, order[1:]):
            relation = relation_lookup.get((source, target))
            if relation and relation.score < self.missing_page_threshold:
                gaps.append(
                    {
                        "between": [source, target],
                        "reason": "low_transition_probability",
                        "score": relation.score,
                    }
                )
        return gaps

    def _detect_low_confidence_edges(
        self,
        order: list[str],
        relation_lookup: dict[tuple[str, str], Relation],
    ) -> list[dict[str, object]]:
        edges: list[dict[str, object]] = []
        for source, target in zip(order, order[1:]):
            relation = relation_lookup.get((source, target))
            if relation and relation.score < self.low_confidence_threshold:
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "score": relation.score,
                    }
                )
        return edges

    def _detect_clusters(self, pages: list[PageFeatures]) -> list[ClusterInfo]:
        if not pages:
            return []
        if len(pages) == 1:
            return [
                ClusterInfo(
                    cluster_id="cluster_000",
                    page_ids=[pages[0].page_id],
                    confidence=1.0,
                    method="single_page",
                )
            ]
        sklearn_clusters = self._detect_clusters_with_sklearn(pages)
        if sklearn_clusters is not None:
            return sklearn_clusters

        graph: dict[str, set[str]] = {page.page_id: set() for page in pages}
        for left_index, left in enumerate(pages):
            for right in pages[left_index + 1 :]:
                similarity = _probability_from_cosine(
                    _cosine_similarity(left.visual_embedding, right.visual_embedding)
                )
                if similarity >= self.cluster_similarity_threshold:
                    graph[left.page_id].add(right.page_id)
                    graph[right.page_id].add(left.page_id)

        visited: set[str] = set()
        clusters: list[ClusterInfo] = []
        for page in pages:
            if page.page_id in visited:
                continue
            stack = [page.page_id]
            component: list[str] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                stack.extend(sorted(graph[current] - visited))
            clusters.append(
                ClusterInfo(
                    cluster_id=f"cluster_{len(clusters):03d}",
                    page_ids=sorted(component),
                    confidence=0.6 if len(component) == 1 and len(pages) > 1 else 0.85,
                    method="visual_similarity_connected_components",
                )
            )
        return clusters

    def _detect_outliers_with_sklearn(
        self, pages: list[PageFeatures]
    ) -> list[dict[str, object]] | None:
        if len(pages) < 4 or not self._embeddings_are_rectangular(pages):
            return None
        try:
            import numpy as np
            from sklearn.ensemble import IsolationForest
        except ImportError:
            return None

        contamination = min(0.49, max(0.01, self.outlier_quantile))
        matrix = np.array([page.visual_embedding for page in pages], dtype=np.float32)
        labels = IsolationForest(
            contamination=contamination,
            random_state=17,
        ).fit_predict(matrix)
        return [
            {
                "page_id": page.page_id,
                "reason": "isolation_forest_visual_embedding",
                "score": int(label),
            }
            for page, label in zip(pages, labels)
            if label == -1
        ]

    def _detect_clusters_with_sklearn(
        self, pages: list[PageFeatures]
    ) -> list[ClusterInfo] | None:
        if not self._embeddings_are_rectangular(pages):
            return None
        try:
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            return None

        matrix = np.array([page.visual_embedding for page in pages], dtype=np.float32)
        distance_threshold = 1.0 - self.cluster_similarity_threshold
        try:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average",
            )
        except TypeError:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                affinity="cosine",
                linkage="average",
            )
        labels = model.fit_predict(matrix)

        grouped: dict[int, list[str]] = defaultdict(list)
        for page, label in zip(pages, labels):
            grouped[int(label)].append(page.page_id)

        clusters: list[ClusterInfo] = []
        for label, page_ids in sorted(grouped.items()):
            clusters.append(
                ClusterInfo(
                    cluster_id=f"cluster_{label:03d}",
                    page_ids=sorted(page_ids),
                    confidence=0.8,
                    method="sklearn_agglomerative_cosine",
                )
            )
        return clusters

    def _embeddings_are_rectangular(self, pages: list[PageFeatures]) -> bool:
        if not pages or not pages[0].visual_embedding:
            return False
        width = len(pages[0].visual_embedding)
        return all(len(page.visual_embedding) == width for page in pages)
