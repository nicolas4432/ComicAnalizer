from __future__ import annotations

from pathlib import Path

from core.data import PageInput
from utils.images import file_sha256, is_supported_image, stable_page_id


class DirectoryImageIngestor:
    """Discovers comic page images from a file or directory."""

    def __init__(self, recursive: bool = True) -> None:
        self.recursive = recursive

    def load(self, input_path: str) -> list[PageInput]:
        root = Path(input_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Input path does not exist: {root}")

        if root.is_file():
            candidates = [root] if is_supported_image(root) else []
        else:
            iterator = root.rglob("*") if self.recursive else root.glob("*")
            candidates = [path for path in iterator if is_supported_image(path)]

        pages: list[PageInput] = []
        for index, path in enumerate(sorted(candidates, key=lambda item: str(item).lower()), 1):
            pages.append(
                PageInput(
                    page_id=stable_page_id(path, index),
                    path=path,
                    sha256=file_sha256(path),
                )
            )
        return pages

