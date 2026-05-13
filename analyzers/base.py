from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.data import PipelineResult


class Analyzer(ABC):
    """Extensible analysis plugin contract."""

    name: str = "analyzer"

    @abstractmethod
    def run(self, data: PipelineResult) -> dict[str, Any]:
        return {}

