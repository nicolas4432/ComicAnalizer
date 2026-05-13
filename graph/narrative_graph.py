from __future__ import annotations

from core.data import PageFeatures, Relation
from core.interfaces import RelationScorer


def build_pairwise_relations(
    pages: list[PageFeatures],
    scorer: RelationScorer,
    top_k_edges: int | None = None,
) -> list[Relation]:
    """Scores every directed pair and optionally keeps only top-k outgoing edges."""

    relations: list[Relation] = []
    for source in pages:
        outgoing = [
            scorer.score(source, target)
            for target in pages
            if target.page_id != source.page_id
        ]
        outgoing.sort(key=lambda relation: relation.score, reverse=True)
        if top_k_edges is not None and top_k_edges > 0:
            outgoing = outgoing[:top_k_edges]
        relations.extend(outgoing)
    return relations


class NetworkXNarrativeGraphBuilder:
    """Builds a directed weighted narrative graph with NetworkX."""

    def build(self, pages: list[PageFeatures], relations: list[Relation]):
        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError(
                "NetworkX is required for graph construction. Install networkx."
            ) from exc

        graph = nx.DiGraph()
        for page in pages:
            graph.add_node(page.page_id, page=page)
        for relation in relations:
            graph.add_edge(
                relation.source,
                relation.target,
                weight=relation.score,
                components=relation.components,
            )
        return graph

