from __future__ import annotations

import math

from core.data import PageFeatures, Relation


def _cosine_similarity(left: list[float] | None, right: list[float] | None) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if math.isclose(left_norm, 0.0) or math.isclose(right_norm, 0.0):
        return None
    return dot / (left_norm * right_norm)


def _probability_from_cosine(value: float | None, neutral: float = 0.5) -> float:
    if value is None:
        return neutral
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


class HeuristicTransitionScorer:
    """Initial replaceable scorer for P(A -> B).

    It is intentionally simple but directional by contract: future trained models can
    replace this class without changing graph construction or pipeline orchestration.
    """

    def __init__(self, config: dict) -> None:
        model_config = config.get("relation_model", {})
        self.visual_weight = float(model_config.get("visual_weight", 0.7))
        self.text_weight = float(model_config.get("text_weight", 0.2))
        self.metadata_weight = float(model_config.get("metadata_weight", 0.1))
        self.target_visual_delta = float(model_config.get("target_visual_delta", 0.35))
        self.duplicate_penalty_delta = float(
            model_config.get("duplicate_penalty_delta", 0.04)
        )
        self.min_edge_score = float(model_config.get("min_edge_score", 0.0))

    def score(self, source: PageFeatures, target: PageFeatures) -> Relation:
        if source.page_id == target.page_id:
            return Relation(source.page_id, target.page_id, 0.0, {"self": 1.0})

        visual_cosine = _cosine_similarity(
            source.visual_embedding, target.visual_embedding
        )
        visual_similarity = _probability_from_cosine(visual_cosine)
        visual_delta = self._visual_delta(visual_cosine)
        transition_change = self._transition_change_score(visual_delta)

        text_similarity = _probability_from_cosine(
            _cosine_similarity(
                source.text.text_embedding,
                target.text.text_embedding,
            ),
            neutral=0.5,
        )
        metadata_similarity = self._metadata_similarity(source, target)

        visual_component = (0.72 * visual_similarity) + (0.28 * transition_change)
        weighted_total = (
            self.visual_weight * visual_component
            + self.text_weight * text_similarity
            + self.metadata_weight * metadata_similarity
        )
        normalizer = self.visual_weight + self.text_weight + self.metadata_weight
        score = weighted_total / normalizer if normalizer else weighted_total

        if visual_delta < self.duplicate_penalty_delta:
            score *= 0.3

        score = max(self.min_edge_score, min(1.0, score))
        return Relation(
            source=source.page_id,
            target=target.page_id,
            score=score,
            components={
                "visual_similarity": visual_similarity,
                "visual_delta": visual_delta,
                "transition_change": transition_change,
                "text_similarity": text_similarity,
                "metadata_similarity": metadata_similarity,
            },
        )

    def _visual_delta(self, visual_cosine: float | None) -> float:
        if visual_cosine is None:
            return self.target_visual_delta
        bounded_cosine = max(-1.0, min(1.0, visual_cosine))
        return math.sqrt(max(0.0, 2.0 - (2.0 * bounded_cosine)))

    def _transition_change_score(self, visual_delta: float) -> float:
        target = max(self.target_visual_delta, 1e-6)
        score = math.exp(-abs(visual_delta - target) / target)
        if visual_delta < self.duplicate_penalty_delta:
            score *= 0.2
        return max(0.0, min(1.0, score))

    def _metadata_similarity(self, source: PageFeatures, target: PageFeatures) -> float:
        source_meta = source.metadata
        target_meta = target.metadata
        aspect_similarity = 1.0 - min(
            1.0, abs(source_meta.aspect_ratio - target_meta.aspect_ratio)
        )
        brightness_similarity = 1.0 - min(
            1.0, abs(source_meta.brightness - target_meta.brightness) / 255.0
        )
        color_distance = math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(source_meta.mean_color_bgr, target_meta.mean_color_bgr)
            )
        )
        color_similarity = 1.0 - min(1.0, color_distance / (255.0 * math.sqrt(3.0)))
        return max(
            0.0,
            min(
                1.0,
                (0.35 * aspect_similarity)
                + (0.35 * brightness_similarity)
                + (0.30 * color_similarity),
            ),
        )

