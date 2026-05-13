from __future__ import annotations

from core.data import TextFeatures


class NoOpTextEmbedder:
    def embed(self, text_features: TextFeatures) -> TextFeatures:
        return text_features


class SentenceTransformerTextEmbedder:
    """Optional sentence-transformers backend for OCR text continuity."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. Disable text embeddings "
                    "or install sentence-transformers."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text_features: TextFeatures) -> TextFeatures:
        if not text_features.text.strip():
            return text_features
        model = self._load()
        embedding = model.encode(text_features.text, normalize_embeddings=True)
        text_features.text_embedding = [float(value) for value in embedding]
        return text_features

