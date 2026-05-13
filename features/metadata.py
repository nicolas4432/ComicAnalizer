from __future__ import annotations

from pathlib import Path

from core.data import ImageMetadata


class ImageMetadataExtractor:
    """Extracts cheap deterministic image descriptors with OpenCV."""

    def extract(self, path: Path) -> ImageMetadata:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV and NumPy are required for metadata extraction. "
                "Install dependencies from requirements.txt."
            ) from exc

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"OpenCV could not read image: {path}")

        if image.ndim == 2:
            height, width = image.shape
            channels = 1
            color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            height, width, channels = image.shape
            color_image = image[:, :, :3]

        mean_color = np.mean(color_image.reshape(-1, 3), axis=0)
        std_color = np.std(color_image.reshape(-1, 3), axis=0)
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        return ImageMetadata(
            width=int(width),
            height=int(height),
            channels=int(channels),
            aspect_ratio=float(width / height) if height else 0.0,
            mean_color_bgr=[float(value) for value in mean_color],
            std_color_bgr=[float(value) for value in std_color],
            brightness=float(np.mean(gray)),
            file_size_bytes=path.stat().st_size,
        )

