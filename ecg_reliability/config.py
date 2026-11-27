from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

VALID_MODEL_NAMES = {"xgboost", "simple_cnn", "resnet1d", "inceptiontime", "tcn", "transformer"}
VALID_CALIBRATION_METHODS = {"raw", "scalar_temperature", "vector_temperature", "platt", "isotonic"}


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


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate structural and scientific invariants of a resolved configuration.

    Raises ``ValueError`` for fatal misconfigurations and returns non-fatal warnings.
    """
    warnings: list[str] = []
    data = config.get("data", {})
    training = config.get("training", {})
    model = config.get("model", {})
    calibration = config.get("calibration", {})
    decision = config.get("decision", {})

    num_leads = int(data.get("num_leads", 12))
    signal_length = int(data.get("signal_length", 1000))
    sampling_rate = int(data.get("sampling_rate", 100))
    if num_leads <= 0 or signal_length <= 0 or sampling_rate <= 0:
        raise ValueError(
            f"Positive data dimensions required: num_leads={num_leads}, signal_length={signal_length}, sampling_rate={sampling_rate}"
        )

    train_folds = {int(f) for f in data.get("train_folds", [1, 2, 3, 4, 5, 6, 7, 8])}
    val_fold = int(data.get("val_fold", 9))
    test_fold = int(data.get("test_fold", 10))
    if val_fold in train_folds or test_fold in train_folds or val_fold == test_fold:
        raise ValueError(
            f"Overlapping PTB-XL folds detected: train={sorted(train_folds)}, val={val_fold}, test={test_fold}"
        )

    model_name = str(model.get("name", ""))
    if model_name and model_name not in VALID_MODEL_NAMES:
        raise ValueError(f"Unsupported model.name={model_name!r}; expected one of {sorted(VALID_MODEL_NAMES)}")

    target_labels = list(data.get("target_labels", []))
    if target_labels and "num_classes" in model and int(model["num_classes"]) != len(target_labels):
        raise ValueError(
            f"model.num_classes ({model['num_classes']}) does not match len(data.target_labels) ({len(target_labels)})"
        )

    if int(training.get("batch_size", 1)) <= 0 or int(training.get("eval_batch_size", 1)) <= 0:
        raise ValueError("training.batch_size and training.eval_batch_size must be positive integers.")
    if int(training.get("epochs", 1)) <= 0:
        raise ValueError("training.epochs must be a positive integer.")

    for method in calibration.get("methods", []):
        if method not in VALID_CALIBRATION_METHODS:
            raise ValueError(f"Unsupported calibration method: {method!r}")
    if int(calibration.get("num_bins", 15)) < 2:
        raise ValueError("calibration.num_bins must be >= 2.")

    for target in decision.get("coverage_targets", []):
        val = float(target)
        if not (0.0 < val <= 1.0):
            raise ValueError(f"decision.coverage_targets values must be in (0, 1], got {val}")

    if model_name == "transformer" and int(model.get("hidden_dim", 128)) % int(model.get("num_heads", 4)) != 0:
        raise ValueError("Transformer hidden_dim must be divisible by num_heads.")

    if sampling_rate == 100:
        warnings.append("sampling_rate=100 Hz has a 50 Hz Nyquist limit; 50/60 Hz powerline stress tests alias.")
    return warnings


def load_config(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base_path = Path("configs/base.yaml")
    if Path(path) != base_path and base_path.exists():
        with base_path.open("r", encoding="utf-8") as handle:
            base_config = yaml.safe_load(handle) or {}
        config = deep_merge(base_config, config)
    resolved = _expand_env(config)
    if validate:
        validate_config(resolved)
    return resolved


def _parse_override_value(value: str) -> Any:
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed = value
    return parsed


def apply_overrides(config: dict[str, Any], overrides: list[str], *, validate: bool = True) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format (expected key=value): {override!r}")
        key_path, raw_value = override.split("=", maxsplit=1)
        cursor = updated
        parts = key_path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _parse_override_value(raw_value)
    if validate:
        validate_config(updated)
    return updated


def get_config_value(config: dict[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor
