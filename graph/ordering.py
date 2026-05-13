from __future__ import annotations

from collections import defaultdict

from core.data import PageFeatures, Relation


class GreedyNarrativeOrdering:
    """Initial global ordering strategy.

    It builds a high-confidence path by repeatedly choosing the strongest outgoing
    edge to an unvisited node. This is deterministic, cycle-free, and easy to
    replace with beam search, ILP, or a learned sequence model later.
    """

    def __init__(self, low_confidence_threshold: float = 0.45) -> None:
        self.low_confidence_threshold = low_confidence_threshold

    def order(self, pages: list[PageFeatures], relations: list[Relation]) -> list[str]:
        if not pages:
            return []
        if len(pages) == 1:
            return [pages[0].page_id]

        page_ids = [page.page_id for page in pages]
        unvisited = set(page_ids)
        outgoing: dict[str, list[Relation]] = defaultdict(list)
        incoming_strength = {page_id: 0.0 for page_id in page_ids}
        outgoing_strength = {page_id: 0.0 for page_id in page_ids}

        for relation in relations:
            outgoing[relation.source].append(relation)
            incoming_strength[relation.target] = incoming_strength.get(
                relation.target, 0.0
            ) + relation.score
            outgoing_strength[relation.source] = outgoing_strength.get(
                relation.source, 0.0
            ) + relation.score

        for source in outgoing:
            outgoing[source].sort(key=lambda item: item.score, reverse=True)

        order: list[str] = []
        while unvisited:
            current = self._choose_segment_start(
                unvisited, incoming_strength, outgoing_strength
            )
            order.append(current)
            unvisited.remove(current)

            while unvisited:
                next_relation = self._best_next_relation(current, unvisited, outgoing)
                if next_relation is None:
                    break
                current = next_relation.target
                order.append(current)
                unvisited.remove(current)

        return order

    def _choose_segment_start(
        self,
        candidates: set[str],
        incoming_strength: dict[str, float],
        outgoing_strength: dict[str, float],
    ) -> str:
        return min(
            candidates,
            key=lambda page_id: (
                incoming_strength.get(page_id, 0.0)
                - outgoing_strength.get(page_id, 0.0),
                page_id,
            ),
        )

    def _best_next_relation(
        self,
        current: str,
        unvisited: set[str],
        outgoing: dict[str, list[Relation]],
    ) -> Relation | None:
        for relation in outgoing.get(current, []):
            if relation.target in unvisited:
                return relation
        return None

