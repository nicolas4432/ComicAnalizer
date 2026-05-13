from __future__ import annotations

import math
from pathlib import Path
from typing import Any


class VisualEmbeddingExtractor:
    """Pluggable CLIP visual embedding extractor.

    Backend selection:
    - ``clip``: OpenAI CLIP package.
    - ``transformers``: Hugging Face CLIPModel/CLIPProcessor.
    - ``auto``: try OpenAI CLIP, then transformers, then optional OpenCV fallback.
    """

    def __init__(
        self,
        backend: str = "auto",
        model_name: str = "ViT-B/32",
        transformers_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        allow_opencv_fallback: bool = True,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.transformers_model_name = transformers_model_name
        self.device = device
        self.allow_opencv_fallback = allow_opencv_fallback
        self._loaded_backend: str | None = None
        self._model: Any = None
        self._preprocess: Any = None
        self._processor: Any = None

    @property
    def loaded_backend(self) -> str:
        return self._loaded_backend or "unloaded"

    def extract(self, path: Path) -> tuple[list[float], str]:
        if self._loaded_backend is None:
            self._load_backend()

        if self._loaded_backend == "openai_clip":
            embedding = self._extract_openai_clip(path)
        elif self._loaded_backend == "transformers_clip":
            embedding = self._extract_transformers_clip(path)
        elif self._loaded_backend == "opencv_fallback":
            embedding = self._extract_opencv_fallback(path)
        else:
            raise RuntimeError(f"Unsupported visual backend: {self._loaded_backend}")

        return embedding, self._loaded_backend

    def _select_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_backend(self) -> None:
        requested = self.backend.lower()
        errors: list[str] = []

        if requested in {"auto", "clip", "openai_clip"}:
            try:
                import clip
                import torch

                device = self._select_device()
                model, preprocess = clip.load(self.model_name, device=device)
                model.eval()
                self._model = model
                self._preprocess = preprocess
                self._torch = torch
                self._device = device
                self._loaded_backend = "openai_clip"
                return
            except Exception as exc:  # noqa: BLE001 - preserve backend errors for reporting.
                errors.append(f"openai_clip: {exc}")
                if requested not in {"auto"}:
                    raise RuntimeError("; ".join(errors)) from exc

        if requested in {"auto", "transformers", "transformers_clip"}:
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                device = self._select_device()
                model = CLIPModel.from_pretrained(self.transformers_model_name)
                processor = CLIPProcessor.from_pretrained(self.transformers_model_name)
                model.to(device)
                model.eval()
                self._model = model
                self._processor = processor
                self._torch = torch
                self._device = device
                self._loaded_backend = "transformers_clip"
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"transformers_clip: {exc}")
                if requested not in {"auto"}:
                    raise RuntimeError("; ".join(errors)) from exc

        if self.allow_opencv_fallback:
            self._loaded_backend = "opencv_fallback"
            return

        message = "Could not initialize a CLIP backend. " + "; ".join(errors)
        raise RuntimeError(message)

    def _load_pil_image(self, path: Path):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for CLIP image loading.") from exc
        return Image.open(path).convert("RGB")

    def _extract_openai_clip(self, path: Path) -> list[float]:
        image = self._load_pil_image(path)
        torch = self._torch
        with torch.no_grad():
            tensor = self._preprocess(image).unsqueeze(0).to(self._device)
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return [float(value) for value in features.squeeze(0).detach().cpu().tolist()]

    def _extract_transformers_clip(self, path: Path) -> list[float]:
        image = self._load_pil_image(path)
        torch = self._torch
        with torch.no_grad():
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            features = self._model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return [float(value) for value in features.squeeze(0).detach().cpu().tolist()]

    def _extract_opencv_fallback(self, path: Path) -> list[float]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "No CLIP backend is available and OpenCV fallback cannot run. "
                "Install clip/transformers or opencv-python plus numpy."
            ) from exc

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read image: {path}")

        resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist_b = cv2.calcHist([resized], [0], None, [32], [0, 256]).flatten()
        hist_g = cv2.calcHist([resized], [1], None, [32], [0, 256]).flatten()
        hist_r = cv2.calcHist([resized], [2], None, [32], [0, 256]).flatten()
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_density = np.array([float(np.mean(edges > 0))], dtype=np.float32)
        stats = np.array(
            [
                float(np.mean(gray)) / 255.0,
                float(np.std(gray)) / 255.0,
                float(image.shape[1] / image.shape[0]) if image.shape[0] else 0.0,
            ],
            dtype=np.float32,
        )
        vector = np.concatenate([hist_b, hist_g, hist_r, hist_h, edge_density, stats])
        norm = float(np.linalg.norm(vector))
        if math.isclose(norm, 0.0):
            return [0.0 for _ in vector]
        vector = vector / norm
        return [float(value) for value in vector.tolist()]

