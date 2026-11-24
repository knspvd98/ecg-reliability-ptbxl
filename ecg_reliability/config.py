from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _expand_env(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(item) for item in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base_path = Path("configs/base.yaml")
    if Path(path) != base_path and base_path.exists():
        with base_path.open("r", encoding="utf-8") as handle:
            base_config = yaml.safe_load(handle) or {}
        config = deep_merge(base_config, config)
    return _expand_env(config)


def _parse_override_value(value: str) -> Any:
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed = value
    return parsed


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for override in overrides:
        key_path, raw_value = override.split("=", maxsplit=1)
        cursor = updated
        parts = key_path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _parse_override_value(raw_value)
    return updated


def get_config_value(config: dict[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor
