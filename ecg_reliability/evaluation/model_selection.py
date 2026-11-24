from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_DEEP_MODEL_EXCLUSIONS = {"xgboost", "simple_cnn"}


def normalize_deep_model_candidates(candidates: Sequence[str] | None) -> list[str]:
    if candidates is None:
        return []
    normalized: list[str] = []
    for candidate in candidates:
        if candidate in DEFAULT_DEEP_MODEL_EXCLUSIONS:
            continue
        if candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _load_metrics(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "macro_auroc": float(payload.get("macro_auroc", float("nan"))),
        "macro_auprc": float(payload.get("macro_auprc", float("nan"))),
    }


def collect_model_metrics(prediction_root: str | Path, candidates: Sequence[str]) -> dict[str, dict[str, dict[str, float]]]:
    root = Path(prediction_root)
    available: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in candidates:
        model_metrics: dict[str, dict[str, float]] = {}
        for split in ("val", "test"):
            metrics = _load_metrics(root / model_name / "raw" / f"{split}_metrics_raw.json")
            if metrics is not None:
                model_metrics[split] = metrics
        if model_metrics:
            available[model_name] = model_metrics
    return available


def select_best_deep_model(
    prediction_root: str | Path,
    candidates: Sequence[str] | None,
) -> dict[str, Any]:
    normalized_candidates = normalize_deep_model_candidates(candidates)
    available_metrics = collect_model_metrics(prediction_root, normalized_candidates)
    warnings: list[str] = []
    if not available_metrics:
        raise FileNotFoundError("No raw metrics were found for the requested deep-model candidates.")

    selection_rows: list[tuple[float, float, str, str]] = []
    for model_name, payload in available_metrics.items():
        if "val" in payload:
            primary = float(payload["val"]["macro_auroc"])
            secondary = float(payload["val"]["macro_auprc"])
            selection_rows.append(
                (
                    primary if not math.isnan(primary) else float("-inf"),
                    secondary if not math.isnan(secondary) else float("-inf"),
                    model_name,
                    "validation",
                )
            )
        elif "test" in payload:
            warnings.append(f"{model_name}: validation metrics unavailable, used test metrics as fallback")
            primary = float(payload["test"]["macro_auroc"])
            secondary = float(payload["test"]["macro_auprc"])
            selection_rows.append(
                (
                    primary if not math.isnan(primary) else float("-inf"),
                    secondary if not math.isnan(secondary) else float("-inf"),
                    model_name,
                    "test_fallback",
                )
            )

    if not selection_rows:
        raise FileNotFoundError("No usable validation or test metrics were found for deep-model selection.")

    selection_rows.sort(key=lambda row: (-row[0], -row[1], row[2]))
    selected = selection_rows[0]
    return {
        "candidates": normalized_candidates,
        "available_metrics": available_metrics,
        "selected_deep_model": selected[2],
        "selection_rule": "primary=validation_macro_auroc secondary=validation_macro_auprc fallback=test_macro_auroc_if_validation_missing",
        "selection_source": selected[3],
        "warnings": warnings,
        "timestamp": datetime.now().isoformat(),
    }


def load_selection_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_selected_model(
    explicit_model: str | None,
    selection_manifest: str | Path | None = None,
    *,
    synthetic_smoke: bool = False,
    synthetic_fallback_model: str | None = None,
) -> str:
    if explicit_model:
        return explicit_model
    if selection_manifest is not None and Path(selection_manifest).exists():
        payload = load_selection_manifest(selection_manifest)
        selected = payload.get("selected_deep_model")
        if selected:
            return str(selected)
    if synthetic_smoke and synthetic_fallback_model:
        return synthetic_fallback_model
    raise FileNotFoundError(
        "Claim-bearing analysis requires --model-name or a selection manifest with selected_deep_model."
    )


def resolve_calibration_method(
    model_name: str,
    explicit_method: str | None = None,
) -> str | None:
    if explicit_method and explicit_method != "best":
        return explicit_method
    reports_path = Path(f"outputs/calibration/{model_name}/reports.json")
    if reports_path.exists():
        with reports_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        best = payload.get("best_method")
        if best:
            return str(best)
    return None
