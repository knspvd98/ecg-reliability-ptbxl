from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ecg_reliability.calibration.isotonic import IsotonicScaler
from ecg_reliability.calibration.platt import PlattScaler
from ecg_reliability.calibration.temperature import TemperatureScaler
from ecg_reliability.evaluation.calibration_metrics import compute_calibration_report
from ecg_reliability.evaluation.thresholding import sigmoid


def fit_calibrator(method: str, logits: np.ndarray, labels: np.ndarray):
    if method == "raw":
        return None
    if method == "scalar_temperature":
        return TemperatureScaler(vector=False).fit(logits, labels)
    if method == "vector_temperature":
        return TemperatureScaler(vector=True).fit(logits, labels)
    if method == "platt":
        return PlattScaler().fit(logits, labels)
    if method == "isotonic":
        return IsotonicScaler().fit(logits, labels)
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(calibrator, logits: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return sigmoid(logits)
    return calibrator.predict_proba(logits)


def evaluate_calibration_methods(methods: list[str], logits: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for method in methods:
        calibrator = fit_calibrator(method, logits, labels)
        probs = apply_calibrator(calibrator, logits)
        report = compute_calibration_report(labels, probs, num_bins=num_bins)
        results[method] = {"calibrator": calibrator, "report": report}
    return results


def select_best_method(results: dict[str, Any]) -> str:
    scored = []
    for method, payload in results.items():
        report = payload["report"]
        scored.append((report["macro_ece"], report["macro_brier"], method))
    scored.sort(key=lambda row: (row[0], row[1], row[2]))
    return scored[0][2]


def save_calibrator(path: str | Path, method: str, calibrator: object, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump({"method": method, "calibrator": calibrator, "report": report}, handle)


def load_calibrator(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
