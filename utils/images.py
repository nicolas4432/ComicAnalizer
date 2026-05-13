from __future__ import annotations

import hashlib
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def can_read_image(path: Path) -> bool:
    """Return whether OpenCV can decode the file as an image."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to validate image inputs. "
            "Install dependencies from requirements.txt."
        ) from exc

    try:
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is not None
    except Exception:
        return False


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_page_id(path: Path, index: int) -> str:
    stem = path.stem.lower().replace(" ", "_")
    return f"page_{index:04d}_{stem}"
