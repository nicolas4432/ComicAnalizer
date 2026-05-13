from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def read_magi_image(path: str | Path) -> np.ndarray:
    """Read a page in the grayscale-to-RGB format used by Magi examples."""

    with Path(path).expanduser().open("rb") as file:
        image = Image.open(file).convert("L").convert("RGB")
        return np.array(image)


class MagiPageExtractor:
    """Thin local wrapper around Magi v3 for experimental page inspection."""

    def __init__(
        self,
        model_name: str = "ragavsachdeva/magiv3",
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = self._select_device(device)
        self.dtype = self._select_dtype(dtype)
        self.model = None
        self.processor = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

    def predict(self, image_paths: list[str | Path], run_ocr: bool = True) -> list[dict[str, Any]]:
        self.load()
        images = [read_magi_image(path) for path in image_paths]
        with torch.no_grad():
            detections = self.model.predict_detections_and_associations(
                images,
                self.processor,
            )
            ocr = self.model.predict_ocr(images, self.processor) if run_ocr else None

        results: list[dict[str, Any]] = []
        for index, path in enumerate(image_paths):
            result = {
                "path": str(Path(path).expanduser().resolve()),
                "detections": _json_safe(detections[index]),
            }
            if ocr is not None:
                result["ocr"] = _json_safe(ocr[index])
            results.append(result)
        return results

    def predict_detections(self, image_paths: list[str | Path]) -> list[dict[str, Any]]:
        self.load()
        images = [read_magi_image(path) for path in image_paths]
        with torch.no_grad():
            detections = self.model.predict_detections_and_associations(
                images,
                self.processor,
            )
        return [
            {
                "path": str(Path(path).expanduser().resolve()),
                "detections": _json_safe(detections[index]),
            }
            for index, path in enumerate(image_paths)
        ]

    def predict_ocr(self, image_paths: list[str | Path]) -> list[dict[str, Any]]:
        self.load()
        images = [read_magi_image(path) for path in image_paths]
        with torch.no_grad():
            ocr = self.model.predict_ocr(images, self.processor)
        return [
            {
                "path": str(Path(path).expanduser().resolve()),
                "ocr": _json_safe(ocr[index]),
            }
            for index, path in enumerate(image_paths)
        ]

    def visualise(
        self,
        image_path: str | Path,
        result: dict[str, Any],
        output_path: str | Path,
    ) -> None:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        detections = result.get("detections", {})
        palette = {
            "panels": "lime",
            "texts": "dodgerblue",
            "characters": "red",
            "faces": "orange",
            "tails": "magenta",
        }
        for key, color in palette.items():
            for box in _extract_boxes(detections.get(key)):
                draw.rectangle(box, outline=color, width=4)
                draw.text((box[0] + 4, box[1] + 4), key, fill=color)
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target)

    def crop_panels(
        self,
        image_path: str | Path,
        result: dict[str, Any],
        output_dir: str | Path,
    ) -> int:
        image = Image.open(image_path).convert("RGB")
        target_dir = Path(output_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        boxes = _extract_boxes(result.get("detections", {}).get("panels"))
        for index, box in enumerate(boxes, 1):
            image.crop(box).save(target_dir / f"panel_{index:03d}.jpg")
        return len(boxes)

    def _select_device(self, requested: str) -> torch.device:
        if requested != "auto":
            return torch.device(requested)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _select_dtype(self, requested: str) -> torch.dtype:
        if requested == "float16":
            return torch.float16
        if requested == "bfloat16":
            return torch.bfloat16
        if requested == "float32":
            return torch.float32
        return torch.float16 if self.device.type == "cuda" else torch.float32


def save_magi_results(results: list[dict[str, Any]], output_path: str | Path) -> None:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_safe(results), indent=2, ensure_ascii=False), encoding="utf-8")


def _extract_boxes(value: Any) -> list[tuple[int, int, int, int]]:
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("boxes", "bboxes", "bbox"):
            if key in value:
                return _extract_boxes(value[key])
        return []
    if isinstance(value, np.ndarray):
        return _extract_boxes(value.tolist())
    if isinstance(value, torch.Tensor):
        return _extract_boxes(value.detach().cpu().tolist())
    if isinstance(value, (list, tuple)):
        boxes: list[tuple[int, int, int, int]] = []
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            x1, y1, x2, y2 = value
            return [(int(x1), int(y1), int(x2), int(y2))]
        for item in value:
            boxes.extend(_extract_boxes(item))
        return boxes
    return []
