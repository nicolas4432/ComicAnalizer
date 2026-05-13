from __future__ import annotations

from pathlib import Path

from core.data import PageInput
from utils.images import can_read_image, file_sha256, is_supported_image, stable_page_id


class DirectoryImageIngestor:
    """Discovers comic page images from a file or directory."""

    def __init__(self, recursive: bool = True, validate_readable: bool = True) -> None:
        self.recursive = recursive
        self.validate_readable = validate_readable
        self.skipped_inputs: list[dict[str, str]] = []

    def load(self, input_path: str) -> list[PageInput]:
        self.skipped_inputs = []
        root = Path(input_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Input path does not exist: {root}")

        if root.is_file():
            if is_supported_image(root):
                candidates = [root]
            else:
                self.skipped_inputs.append(
                    {"path": str(root), "reason": "unsupported_extension"}
                )
                candidates = []
        else:
            iterator = root.rglob("*") if self.recursive else root.glob("*")
            candidates = [path for path in iterator if is_supported_image(path)]

        pages: list[PageInput] = []
        for path in sorted(candidates, key=lambda item: str(item).lower()):
            if self.validate_readable and not can_read_image(path):
                self.skipped_inputs.append(
                    {"path": str(path), "reason": "unreadable_image"}
                )
                continue

            index = len(pages) + 1
            pages.append(
                PageInput(
                    page_id=stable_page_id(path, index),
                    path=path,
                    sha256=file_sha256(path),
                )
            )
        return pages
