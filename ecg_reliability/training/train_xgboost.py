from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from ecg_reliability.data.features import compute_engineered_features
from ecg_reliability.io import read_table
from ecg_reliability.models.xgboost_baseline import MultiLabelXGBoost
from ecg_reliability.performance.runtime_profile import write_runtime_report


def _load_features(manifest_path: str | Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = read_table(manifest_path)
    subset = frame.loc[frame["split"] == split].reset_index(drop=True)
    X = np.stack([compute_engineered_features(np.load(path)) for path in subset["cache_path"]], axis=0)
    y = subset[["NORM", "MI", "STTC", "CD", "HYP"]].to_numpy(dtype=np.int64)
    ecg_ids = subset["ecg_id"].to_numpy(dtype=np.int64)
    return X, y, ecg_ids


def fit_xgboost_model(config: dict, manifest_path: str | Path, output_path: str | Path) -> dict[str, str | dict]:
    feature_start = perf_counter()
    X_train, y_train, _ = _load_features(manifest_path, "train")
    feature_elapsed = perf_counter() - feature_start
    model = MultiLabelXGBoost(**config["xgboost"])
    fit_start = perf_counter()
    model.fit(X_train, y_train)
    fit_elapsed = perf_counter() - fit_start
    model.save(output_path)
    runtime_report = {
        "model_name": "xgboost",
        "device": "cpu",
        "n_jobs": int(config["xgboost"]["n_jobs"]),
        "tree_method": str(config["xgboost"].get("tree_method", "hist")),
        "feature_extraction_seconds": float(feature_elapsed),
        "fit_seconds": float(fit_elapsed),
        "train_records": int(len(X_train)),
        "feature_records_per_second": float(len(X_train) / max(feature_elapsed, 1e-9)),
        "fit_records_per_second": float(len(X_train) / max(fit_elapsed, 1e-9)),
    }
    write_runtime_report(Path(output_path).parent / "training_runtime_report.json", runtime_report)
    return {"model_path": str(output_path), "runtime_report": runtime_report}


def predict_xgboost_model(model_path: str | Path, manifest_path: str | Path, split: str) -> dict[str, np.ndarray]:
    model = MultiLabelXGBoost.load(model_path)
    X, y, ecg_ids = _load_features(manifest_path, split)
    probs = model.predict_proba(X)
    logits = model.predict_logits(X)
    return {"ecg_ids": ecg_ids, "labels": y, "probabilities": probs, "logits": logits}
