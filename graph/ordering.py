from __future__ import annotations

from collections import defaultdict
import math

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


class BeamSearchNarrativeOrdering:
    """Approximate global ordering using directed transition probabilities.

    Greedy follows one local choice at a time. Beam search keeps several plausible
    partial stories alive, which is a better fit for learned pairwise scores where
    the correct page may be ranked second or third locally but still produce the
    best global sequence.
    """

    def __init__(self, beam_width: int = 128, edge_epsilon: float = 1e-8) -> None:
        self.beam_width = max(1, int(beam_width))
        self.edge_epsilon = edge_epsilon

    def order(self, pages: list[PageFeatures], relations: list[Relation]) -> list[str]:
        if not pages:
            return []
        if len(pages) == 1:
            return [pages[0].page_id]

        page_ids = [page.page_id for page in pages]
        scores = {
            (relation.source, relation.target): max(
                self.edge_epsilon,
                min(1.0 - self.edge_epsilon, relation.score),
            )
            for relation in relations
        }

        beams: list[tuple[float, tuple[str, ...], frozenset[str]]] = [
            (0.0, (page_id,), frozenset({page_id})) for page_id in page_ids
        ]

        for _ in range(len(page_ids) - 1):
            candidates: list[tuple[float, tuple[str, ...], frozenset[str]]] = []
            for score, path, visited in beams:
                current = path[-1]
                for candidate in page_ids:
                    if candidate in visited:
                        continue
                    transition = scores.get((current, candidate), self.edge_epsilon)
                    next_score = score + math.log(transition)
                    candidates.append(
                        (
                            next_score,
                            (*path, candidate),
                            frozenset((*visited, candidate)),
                        )
                    )
            if not candidates:
                break
            candidates.sort(key=lambda item: item[0], reverse=True)
            beams = candidates[: self.beam_width]

        best = max(beams, key=lambda item: item[0])
        return list(best[1])
