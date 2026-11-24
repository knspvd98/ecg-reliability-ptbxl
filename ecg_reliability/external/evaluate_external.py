from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ecg_reliability.calibration.apply import apply_calibrator, load_calibrator
from ecg_reliability.data.features import compute_engineered_features
from ecg_reliability.data.labels import TARGET_LABELS
from ecg_reliability.external.label_mapping import optional_snomed_overlap_label_vector
from ecg_reliability.external.reporting import (
    compute_binary_metrics,
    compute_sourcewise_binary_metrics,
    load_snomed_overlap_mapping,
)
from ecg_reliability.models.xgboost_baseline import MultiLabelXGBoost
from ecg_reliability.performance.device import (
    configure_torch_runtime,
    cuda_memory_snapshot,
    move_batch_to_device,
    resolve_device,
)
from ecg_reliability.runtime import checkpoint_dir, load_torch_model

ABNORMAL_INDICES = [TARGET_LABELS.index(label) for label in ["MI", "STTC", "CD", "HYP"]]


class ExternalCachedDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.frame.iloc[index]
        signal = np.load(row["cache_path"]).astype(np.float32)
        label = np.asarray([row["binary_abnormal_label"]], dtype=np.float32)
        metadata = {
            "external_record_id": str(row["external_record_id"]),
            "source": str(row["source"]),
            "dx_codes": row["dx_codes"],
        }
        return torch.from_numpy(signal), torch.from_numpy(label), metadata


def external_binary_scores(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    abnormal_probs = probabilities[:, ABNORMAL_INDICES]
    return {
        "p_abnormal_max": abnormal_probs.max(axis=1),
        "p_abnormal_not_norm": 1.0 - probabilities[:, TARGET_LABELS.index("NORM")],
        "p_abnormal_mean_abnormal": abnormal_probs.mean(axis=1),
    }


def _predict_torch_logits(frame: pd.DataFrame, model_name: str, config: dict[str, Any]) -> tuple[np.ndarray, torch.device]:
    device = resolve_device(config["training"]["device"])
    configure_torch_runtime(config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, _ = load_torch_model(checkpoint_dir(model_name) / "best.pt", device=device)
    dataloader = DataLoader(
        ExternalCachedDataset(frame),
        batch_size=int(config["training"]["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    logits_all: list[np.ndarray] = []
    non_blocking = bool(config["training"].get("non_blocking_transfer", False))
    with torch.no_grad():
        for signals, _, _ in dataloader:
            signals = move_batch_to_device(signals, device, non_blocking=non_blocking)
            logits_all.append(model(signals).detach().cpu().numpy())
    return np.concatenate(logits_all, axis=0), device


def _predict_xgboost_logits(frame: pd.DataFrame) -> tuple[np.ndarray, torch.device]:
    model = MultiLabelXGBoost.load(checkpoint_dir("xgboost") / "model.pkl")
    features = np.stack([compute_engineered_features(np.load(path)) for path in frame["cache_path"]], axis=0)
    return model.predict_logits(features), torch.device("cpu")


def _load_external_frame(config: dict[str, Any]) -> pd.DataFrame:
    manifest_path = Path(config["external"]["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"External manifest not found: {manifest_path}")
    frame = pd.read_parquet(manifest_path) if manifest_path.suffix == ".parquet" else pd.read_csv(manifest_path)
    valid = frame.loc[(frame["status"] == "cached") & frame["binary_abnormal_label"].notna()].reset_index(drop=True)
    if valid.empty:
        raise ValueError("No cached external records with valid binary abnormal labels were found.")
    return valid


def evaluate_external_model(
    *,
    model_name: str,
    config: dict[str, Any],
    calibration_method: str | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any] | None, dict[str, Any]]:
    frame = _load_external_frame(config)
    start = perf_counter()
    if model_name == "xgboost":
        logits, device = _predict_xgboost_logits(frame)
    else:
        logits, device = _predict_torch_logits(frame, model_name, config)

    raw_only = True
    used_method = calibration_method
    if calibration_method is not None:
        calibrator_path = Path(f"outputs/calibration/{model_name}/{calibration_method}.pkl")
        if calibrator_path.exists():
            calibrator = load_calibrator(calibrator_path)["calibrator"]
            probabilities = apply_calibrator(calibrator, logits)
            raw_only = False
        else:
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            used_method = None
    else:
        probabilities = 1.0 / (1.0 + np.exp(-logits))

    binary_scores = external_binary_scores(probabilities)
    frame = frame.copy()
    for label_idx, label_name in enumerate(TARGET_LABELS):
        frame[f"prob_{label_name}"] = probabilities[:, label_idx]
    for score_name, values in binary_scores.items():
        frame[score_name] = values.astype(np.float32)

    label_column = "binary_abnormal_label"
    all_score_metrics = {
        score_name: compute_binary_metrics(frame[label_column].to_numpy(dtype=np.float32), values)
        for score_name, values in binary_scores.items()
    }
    primary_score_name = "p_abnormal_max"
    sourcewise = compute_sourcewise_binary_metrics(frame, label_column, primary_score_name)
    elapsed = perf_counter() - start

    overlap_results = None
    overlap_warnings: list[str] = []
    external_cfg = config["external"]
    if bool(external_cfg.get("snomed_overlap_enabled", False)):
        mapping = load_snomed_overlap_mapping(external_cfg["snomed_overlap_mapping_path"])
        if mapping is None:
            overlap_warnings.append("snomed_overlap_mapping_missing_or_empty")
        else:
            labels_payload = mapping.get("labels", mapping)
            overlap_summary: dict[str, Any] = {"labels": {}, "warnings": []}
            for label_name, payload in labels_payload.items():
                score_source = payload.get("score_source")
                if score_source is None:
                    overlap_summary["warnings"].append(f"{label_name}: missing score_source")
                    continue
                if score_source in TARGET_LABELS:
                    score = frame[f"prob_{score_source}"].to_numpy(dtype=np.float32)
                elif score_source in binary_scores:
                    score = frame[score_source].to_numpy(dtype=np.float32)
                else:
                    overlap_summary["warnings"].append(f"{label_name}: unsupported score_source={score_source}")
                    continue
                targets = np.asarray(
                    [
                        optional_snomed_overlap_label_vector(dx_codes, {"labels": {label_name: payload}})[label_name]
                        for dx_codes in frame["dx_codes"]
                    ],
                    dtype=np.float32,
                )
                overlap_summary["labels"][label_name] = compute_binary_metrics(targets, score)
            overlap_results = overlap_summary

    result_json = {
        "validation_task": "external_binary_abnormal_vs_normal",
        "warning": "External binary abnormal-vs-normal sanity check, not PTB-XL five-superclass external validation.",
        "external_five_superclass_validation": False,
        "primary_external_score": primary_score_name,
        "raw_only": raw_only,
        "calibration_method_used": used_method,
        "n_records": int(len(frame)),
        "sources": sorted(frame["source"].astype(str).unique().tolist()),
        "scores": all_score_metrics,
        "sourcewise_metrics_path": f"outputs/external/{model_name}/external_sourcewise_results.csv",
        "overlap_warnings": overlap_warnings,
    }
    runtime_report = {
        "model_name": model_name,
        "number_of_records": int(len(frame)),
        "device": str(device),
        "elapsed_seconds": float(elapsed),
        "records_per_second": float(len(frame) / max(elapsed, 1e-9)),
        "peak_cuda_memory": cuda_memory_snapshot()["peak_allocated"],
        "raw_only": raw_only,
        "calibration_method_used": used_method,
    }
    return result_json, frame, sourcewise, overlap_results, runtime_report
