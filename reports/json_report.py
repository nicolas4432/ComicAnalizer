from __future__ import annotations

import json
from pathlib import Path

from core.data import PipelineResult


class JsonReportWriter:
    def __init__(self, include_embeddings: bool = True, pretty: bool = True) -> None:
        self.include_embeddings = include_embeddings
        self.pretty = pretty

    def write(self, result: PipelineResult, output_path: str) -> None:
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as output_file:
            json.dump(
                result.to_dict(include_embeddings=self.include_embeddings),
                output_file,
                ensure_ascii=True,
                indent=2 if self.pretty else None,
            )

