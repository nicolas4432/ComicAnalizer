from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from PIL import Image


class ClipEmbeddingExtractor:
    """Strict CLIP embedding extractor for training.

    This class intentionally has no OpenCV fallback. Training data must use a real
    CLIP image encoder so the learned transition model sees stable semantic
    features instead of environment-dependent handcrafted descriptors.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        transformers_model_name: str = "openai/clip-vit-base-patch32",
        backend: str = "auto",
        device: str = "auto",
        cache_dir: str | Path = ".cache/clip_embeddings",
    ) -> None:
        self.model_name = model_name
        self.transformers_model_name = transformers_model_name
        self.backend = backend.lower()
        self.device = self._select_device(device)
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_backend: str | None = None
        self._model: Any = None
        self._preprocess: Any = None
        self._processor: Any = None

    @property
    def embedding_dim(self) -> int:
        if self._loaded_backend is None:
            self._load_backend()
        if self._loaded_backend == "openai_clip":
            return int(self._model.visual.output_dim)
        if self._loaded_backend == "transformers_clip":
            return int(self._model.config.projection_dim)
        raise RuntimeError(f"Unknown CLIP backend: {self._loaded_backend}")

    @property
    def loaded_backend(self) -> str:
        return self._loaded_backend or "unloaded"

    def extract(self, image_path: str | Path) -> torch.Tensor:
        path = Path(image_path).expanduser().resolve()
        if self._loaded_backend is None:
            self._load_backend()

        cached = self._load_from_cache(path)
        if cached is not None:
            return cached

        if self._loaded_backend == "openai_clip":
            embedding = self._extract_openai_clip(path)
        elif self._loaded_backend == "transformers_clip":
            embedding = self._extract_transformers_clip(path)
        else:
            raise RuntimeError(f"Unknown CLIP backend: {self._loaded_backend}")

        embedding = F.normalize(embedding.float().cpu(), dim=0)
        self._save_to_cache(path, embedding)
        return embedding

    def extract_many(self, image_paths: Iterable[str | Path]) -> dict[str, torch.Tensor]:
        embeddings: dict[str, torch.Tensor] = {}
        for image_path in image_paths:
            resolved = str(Path(image_path).expanduser().resolve())
            embeddings[resolved] = self.extract(resolved)
        return embeddings

    def _select_device(self, requested: str) -> torch.device:
        if requested != "auto":
            return torch.device(requested)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_backend(self) -> None:
        errors: list[str] = []
        if self.backend in {"auto", "clip", "openai_clip"}:
            try:
                import clip

                model, preprocess = clip.load(self.model_name, device=str(self.device))
                model.eval()
                self._model = model
                self._preprocess = preprocess
                self._loaded_backend = "openai_clip"
                return
            except Exception as exc:  # noqa: BLE001 - backend choice should be explicit in error.
                errors.append(f"openai_clip: {exc}")
                if self.backend != "auto":
                    raise RuntimeError("; ".join(errors)) from exc

        if self.backend in {"auto", "transformers", "transformers_clip"}:
            try:
                from transformers import CLIPModel, CLIPProcessor

                model = CLIPModel.from_pretrained(self.transformers_model_name)
                processor = CLIPProcessor.from_pretrained(self.transformers_model_name)
                model.to(self.device)
                model.eval()
                self._model = model
                self._processor = processor
                self._loaded_backend = "transformers_clip"
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"transformers_clip: {exc}")
                if self.backend != "auto":
                    raise RuntimeError("; ".join(errors)) from exc

        raise RuntimeError(
            "A real CLIP backend is required for training. Install OpenAI CLIP or "
            f"transformers. Backend errors: {'; '.join(errors)}"
        )

    def _extract_openai_clip(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        with torch.no_grad():
            tensor = self._preprocess(image).unsqueeze(0).to(self.device)
            features = self._model.encode_image(tensor)
            features = F.normalize(features.float(), dim=-1)
        return features.squeeze(0)

    def _extract_transformers_clip(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        with torch.no_grad():
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self._model.get_image_features(**inputs)
            features = F.normalize(features.float(), dim=-1)
        return features.squeeze(0)

    def _cache_key(self, path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        digest.update(self.model_name.encode("utf-8"))
        digest.update(self.transformers_model_name.encode("utf-8"))
        digest.update((self._loaded_backend or self.backend).encode("utf-8"))
        return digest.hexdigest()

    def _cache_path(self, path: Path) -> Path:
        return self.cache_dir / f"{self._cache_key(path)}.pt"

    def _load_from_cache(self, path: Path) -> torch.Tensor | None:
        cache_path = self._cache_path(path)
        if not cache_path.exists():
            return None
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        embedding = payload["embedding"] if isinstance(payload, dict) else payload
        return F.normalize(embedding.float().cpu(), dim=0)

    def _save_to_cache(self, path: Path, embedding: torch.Tensor) -> None:
        torch.save(
            {
                "embedding": embedding.float().cpu(),
                "model_name": self.model_name,
                "transformers_model_name": self.transformers_model_name,
                "backend": self.loaded_backend,
                "source_path": str(path),
            },
            self._cache_path(path),
        )


def build_directional_pair_features(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
) -> torch.Tensor:
    """Create asymmetric pair features for P(A -> B)."""

    embedding_a = embedding_a.float().flatten()
    embedding_b = embedding_b.float().flatten()
    if embedding_a.shape != embedding_b.shape:
        raise ValueError(
            f"Embedding dimensions differ: {embedding_a.shape} vs {embedding_b.shape}"
        )
    return torch.cat(
        [
            embedding_a,
            embedding_b,
            embedding_b - embedding_a,
            torch.abs(embedding_b - embedding_a),
        ],
        dim=0,
    )
