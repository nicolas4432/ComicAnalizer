from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.json"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | None = None) -> dict[str, Any]:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as default_file:
        config = json.load(default_file)

    if config_path is None:
        return config

    user_path = Path(config_path)
    with user_path.open("r", encoding="utf-8") as user_file:
        user_config = json.load(user_file)
    return deep_merge(config, user_config)

