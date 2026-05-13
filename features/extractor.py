from __future__ import annotations

from core.data import PageFeatures, PageInput
from features.clip_extractor import VisualEmbeddingExtractor
from features.layout import NoOpLayoutExtractor
from features.metadata import ImageMetadataExtractor
from features.ocr import NoOpOCRExtractor, PaddleOCRExtractor
from features.text_embeddings import NoOpTextEmbedder, SentenceTransformerTextEmbedder


class CompositeFeatureExtractor:
    """Coordinates independent feature modules without owning their internals."""

    def __init__(self, config: dict) -> None:
        feature_config = config.get("features", {})
        clip_config = feature_config.get("clip", {})
        ocr_config = feature_config.get("ocr", {})
        text_config = feature_config.get("text_embeddings", {})

        self.metadata_extractor = ImageMetadataExtractor()
        self.visual_extractor = VisualEmbeddingExtractor(
            backend=clip_config.get("backend", "auto"),
            model_name=clip_config.get("model_name", "ViT-B/32"),
            transformers_model_name=clip_config.get(
                "transformers_model_name", "openai/clip-vit-base-patch32"
            ),
            device=clip_config.get("device", "auto"),
            allow_opencv_fallback=clip_config.get("allow_opencv_fallback", True),
        )
        self.ocr_extractor = (
            PaddleOCRExtractor(lang=ocr_config.get("lang", "en"))
            if ocr_config.get("enabled", False)
            else NoOpOCRExtractor()
        )
        self.text_embedder = (
            SentenceTransformerTextEmbedder(
                model_name=text_config.get(
                    "model_name", "sentence-transformers/all-MiniLM-L6-v2"
                )
            )
            if text_config.get("enabled", False)
            else NoOpTextEmbedder()
        )
        self.layout_extractor = NoOpLayoutExtractor()

    def extract(self, page: PageInput) -> PageFeatures:
        metadata = self.metadata_extractor.extract(page.path)
        visual_embedding, visual_backend = self.visual_extractor.extract(page.path)
        text_features = self.text_embedder.embed(self.ocr_extractor.extract(page.path))
        layout_features = self.layout_extractor.extract(page.path)

        return PageFeatures(
            page_id=page.page_id,
            path=str(page.path),
            sha256=page.sha256,
            metadata=metadata,
            visual_embedding=visual_embedding,
            visual_backend=visual_backend,
            text=text_features,
            layout=layout_features,
        )
