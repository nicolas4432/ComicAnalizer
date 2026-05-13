from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from core.data import PageFeatures, Relation
from models.feature_extractor import ClipEmbeddingExtractor, build_directional_pair_features


class PairwiseTransitionMLP(nn.Module):
    """MLP that learns directional transition probability P(A -> B)."""

    def __init__(self, embedding_dim: int, dropout: float = 0.25) -> None:
        super().__init__()
        input_dim = embedding_dim * 4
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class LearnedRelationModel:
    """Inference wrapper compatible with the existing narrative pipeline."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        clip_extractor: ClipEmbeddingExtractor | None = None,
        device: str = "auto",
    ) -> None:
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        )
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=self.device,
            weights_only=True,
        )
        self.embedding_dim = int(checkpoint["embedding_dim"])
        self.model = PairwiseTransitionMLP(
            embedding_dim=self.embedding_dim,
            dropout=float(checkpoint.get("dropout", 0.0)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.clip_extractor = clip_extractor

    def score(self, page_A: Any, page_B: Any) -> float:
        embedding_a = self._resolve_embedding(page_A)
        embedding_b = self._resolve_embedding(page_B)
        features = build_directional_pair_features(embedding_a, embedding_b)
        with torch.no_grad():
            probability = self.model(features.unsqueeze(0).to(self.device))
        return float(probability.squeeze().detach().cpu().item())

    def _resolve_embedding(self, page: Any) -> torch.Tensor:
        if isinstance(page, torch.Tensor):
            return page.float().cpu()
        if isinstance(page, (list, tuple)):
            return torch.tensor(page, dtype=torch.float32)
        if isinstance(page, (str, Path)):
            if self.clip_extractor is None:
                raise ValueError("clip_extractor is required when scoring image paths.")
            return self.clip_extractor.extract(page)
        if hasattr(page, "visual_embedding"):
            return torch.tensor(page.visual_embedding, dtype=torch.float32)
        if hasattr(page, "path"):
            if self.clip_extractor is None:
                raise ValueError("clip_extractor is required when scoring page paths.")
            return self.clip_extractor.extract(page.path)
        raise TypeError(f"Unsupported page input for scoring: {type(page)!r}")


class LearnedTransitionScorer:
    """Adapter that lets the learned model replace HeuristicTransitionScorer."""

    def __init__(self, learned_model: LearnedRelationModel) -> None:
        self.learned_model = learned_model

    def score(self, source: PageFeatures, target: PageFeatures) -> Relation:
        if source.page_id == target.page_id:
            return Relation(source.page_id, target.page_id, 0.0, {"self": 1.0})
        probability = self.learned_model.score(source, target)
        return Relation(
            source=source.page_id,
            target=target.page_id,
            score=probability,
            components={"model": "learned_pairwise_mlp"},
        )


def save_transition_checkpoint(
    path: str | Path,
    model: PairwiseTransitionMLP,
    optimizer: torch.optim.Optimizer,
    embedding_dim: int,
    metrics: dict[str, float],
    dropout: float,
) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "embedding_dim": embedding_dim,
            "metrics": metrics,
            "dropout": dropout,
        },
        target,
    )
