from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_reliability.calibration.apply import apply_calibrator, load_calibrator
from ecg_reliability.data.dataset import CachedECGDataset
from ecg_reliability.data.features import compute_engineered_features
from ecg_reliability.decision.harm import weight_arrays
from ecg_reliability.evaluation.model_selection import resolve_calibration_method, resolve_selected_model
from ecg_reliability.evaluation.reports import save_json
from ecg_reliability.evaluation.thresholding import sigmoid
from ecg_reliability.models.xgboost_baseline import MultiLabelXGBoost
from ecg_reliability.performance.device import configure_torch_runtime, cuda_memory_snapshot, resolve_device
from ecg_reliability.performance.runtime_profile import StageTimer, write_runtime_report
from ecg_reliability.robustness.evaluate_robustness import (
    decision_layer_unavailable_payload,
    evaluate_decision_layer_under_perturbations,
    evaluate_under_perturbations,
)
from ecg_reliability.robustness.perturbations import (
    MIT_BIH_NSTDB_REFERENCE,
    build_severity_perturbation_suite,
    load_nstdb_noise_reference,
)
from ecg_reliability.runtime import (
    calibration_dir,
    checkpoint_dir,
    decision_dir,
    load_runtime_config,
    load_torch_model,
    model_name_from_config,
    robustness_dir,
)


def build_signal_batch(manifest_path: str, split: str, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    dataset = CachedECGDataset(manifest_path, split=split, limit=limit)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    signals_all, labels_all = [], []
    for signals, labels, _ in loader:
        signals_all.append(signals.numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(signals_all, axis=0), np.concatenate(labels_all, axis=0)


def resolve_thresholds(model_name: str, method: str | None) -> dict[str, float] | None:
    if method is None:
        return None
    path = decision_dir(model_name) / method / "thresholds.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): float(value) for key, value in payload.get("thresholds", {}).items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--calibration-method", default=None)
    parser.add_argument("--selection-manifest", default="outputs/summary/model_selection.json")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--severity-suite", action="store_true", help="Evaluate full multi-severity ECG perturbation suite.")
    parser.add_argument("--nstdb-root", default=None, help="Optional local path to MIT-BIH Noise Stress Test Database.")
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    configured_model = model_name_from_config(config, args.model_name)
    model_name = resolve_selected_model(
        configured_model if args.model_name is not None else None,
        args.selection_manifest,
        synthetic_smoke=args.synthetic_smoke,
        synthetic_fallback_model=configured_model,
    )
    device = resolve_device(config["training"]["device"])
    configure_torch_runtime(config)
    if model_name == "xgboost":
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    limit = config["data"].get("debug_max_records")
    signals, labels = build_signal_batch(
        config["data"]["cache_manifest_path"],
        "test",
        limit=min(limit, 64) if isinstance(limit, int) else 128,
    )

    if model_name == "xgboost":
        xgb_model = MultiLabelXGBoost.load(checkpoint_dir("xgboost") / "model.pkl")

        def predict_logits_fn(batch: np.ndarray) -> np.ndarray:
            features = np.stack([compute_engineered_features(signal) for signal in batch], axis=0)
            return xgb_model.predict_logits(features)
    else:
        model, _ = load_torch_model(checkpoint_dir(model_name) / "best.pt", device=device)

        def predict_logits_fn(batch: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                logits = model(torch.tensor(batch, dtype=torch.float32, device=device)).cpu().numpy()
            return logits

    def predict_raw_probabilities_fn(batch: np.ndarray) -> np.ndarray:
        return sigmoid(predict_logits_fn(batch))

    if args.severity_suite:
        perturbations = build_severity_perturbation_suite(
            sampling_rate=int(config["data"]["sampling_rate"]),
            seed=int(config["project"]["seed"]),
        )
    else:
        perturbations = {}
        rob_cfg = config["robustness"]["perturbations"]
        if rob_cfg["lead_dropout_single"]["enabled"]:
            perturbations["lead_dropout_single"] = {"lead_index": 0}
        if rob_cfg["lead_dropout_group"]["enabled"]:
            for group in rob_cfg["lead_dropout_group"]["groups"]:
                perturbations[f"lead_dropout_group_{group}"] = {"group": group}
        if rob_cfg["gaussian_noise"]["enabled"]:
            perturbations["gaussian_noise"] = {
                "std": rob_cfg["gaussian_noise"]["std"],
                "seed": int(config["project"]["seed"]),
            }
        if rob_cfg["baseline_wander"]["enabled"]:
            perturbations["baseline_wander"] = {
                "amplitude": rob_cfg["baseline_wander"]["amplitude"],
                "frequency_hz": rob_cfg["baseline_wander"]["frequency_hz"],
                "sampling_rate": config["data"]["sampling_rate"],
            }
        if rob_cfg["amplitude_scaling"]["enabled"]:
            perturbations["amplitude_scaling"] = {"scale": rob_cfg["amplitude_scaling"]["scale_min"]}
        if rob_cfg.get("powerline_interference", {}).get("enabled", False):
            perturbations["powerline_interference"] = {
                "amplitude": rob_cfg["powerline_interference"]["amplitude"],
                "frequency_hz": rob_cfg["powerline_interference"]["frequency_hz"],
                "sampling_rate": config["data"]["sampling_rate"],
            }
        if rob_cfg.get("muscle_noise", {}).get("enabled", False):
            perturbations["muscle_noise"] = {
                "snr_db": rob_cfg["muscle_noise"].get("snr_db", 12.0),
                "sampling_rate": config["data"]["sampling_rate"],
                "seed": int(config["project"]["seed"]),
            }

    stage_timings: dict[str, float] = {}
    with StageTimer("robustness_eval", stage_timings):
        results = evaluate_under_perturbations(signals, labels, predict_raw_probabilities_fn, perturbations)

    method = resolve_calibration_method(model_name, args.calibration_method)
    warnings: list[str] = []
    if method is None:
        warnings.append("calibration artifacts unavailable; decision-layer robustness skipped")
        decision_results = decision_layer_unavailable_payload("calibration artifacts unavailable")
    else:
        calibrator_path = calibration_dir(model_name) / f"{method}.pkl"
        thresholds = resolve_thresholds(model_name, method)
        if not calibrator_path.exists() or thresholds is None:
            warnings.append("decision artifacts unavailable; decision-layer robustness skipped")
            decision_results = decision_layer_unavailable_payload("decision artifacts unavailable")
        else:
            calibrator = load_calibrator(calibrator_path)["calibrator"]
            fn_weights, fp_weights = weight_arrays(config["decision"])

            def predict_calibrated_probabilities_fn(batch: np.ndarray) -> np.ndarray:
                return apply_calibrator(calibrator, predict_logits_fn(batch))

            with StageTimer("robustness_decision_eval", stage_timings):
                decision_results = evaluate_decision_layer_under_perturbations(
                    signals,
                    labels,
                    predict_calibrated_probabilities_fn,
                    perturbations,
                    fn_weights=fn_weights,
                    fp_weights=fp_weights,
                    thresholds=thresholds,
                    abstain_cost=float(config["decision"]["abstain_cost"]),
                )

    results["decision_layer_available"] = bool(decision_results["decision_layer_available"])
    results["decision_layer"] = decision_results
    results["warnings"] = warnings
    results["scientific_scope"] = "synthetic_stress_test_not_clinical_claim"
    results["nstdb_reference"] = load_nstdb_noise_reference(args.nstdb_root) if args.nstdb_root else MIT_BIH_NSTDB_REFERENCE

    out_dir = robustness_dir(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "results.json", results)
    pd.DataFrame(results.get("delta_summary", [])).to_csv(out_dir / "delta_summary.csv", index=False)
    pd.DataFrame(decision_results.get("decision_delta_summary", [])).to_csv(out_dir / "decision_delta_summary.csv", index=False)
    runtime_report = {
        "number_of_samples": int(len(signals)),
        "device": str(device),
        "elapsed_seconds": float(stage_timings["robustness_eval"]),
        "samples_per_second": float(len(signals) / max(stage_timings["robustness_eval"], 1e-9)),
        "peak_cuda_memory": cuda_memory_snapshot()["peak_allocated"],
        "decision_layer_available": bool(decision_results["decision_layer_available"]),
        "calibration_method": method,
        "warnings": warnings,
    }
    write_runtime_report(out_dir / "runtime_report.json", runtime_report)
    print(f"Robustness results saved to {out_dir}")


if __name__ == "__main__":
    main()
