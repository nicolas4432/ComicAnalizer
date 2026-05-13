from __future__ import annotations

from pathlib import Path

from core.data import LayoutFeatures


class NoOpLayoutExtractor:
    """Placeholder for future LayoutParser/panel structure extraction."""

    def extract(self, path: Path) -> LayoutFeatures:
        return LayoutFeatures(backend="none")

