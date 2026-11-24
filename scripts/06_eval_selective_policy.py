from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_reliability.decision.harm import weight_arrays
from ecg_reliability.decision.selective import evaluate_selective_policy, fit_coverage_thresholds
from ecg_reliability.evaluation.reports import save_json
from ecg_reliability.runtime import (
    calibration_dir,
    decision_dir,
    load_runtime_config,
    model_name_from_config,
)


def resolve_method(model_name: str, requested: str) -> str:
    if requested != "best":
        return requested
    report_path = calibration_dir(model_name) / "reports.json"
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["best_method"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--calibration-method", default="best")
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    model_name = model_name_from_config(config, args.model_name)
    method = resolve_method(model_name, args.calibration_method)
    val_dir = Path(f"outputs/predictions/{model_name}/calibrated/{method}/val")
    test_dir = Path(f"outputs/predictions/{model_name}/calibrated/{method}/test")
    import numpy as np

    val_probs = np.load(val_dir / "probabilities.npy")
    val_labels = np.load(val_dir / "labels.npy")
    test_probs = np.load(test_dir / "probabilities.npy")
    test_labels = np.load(test_dir / "labels.npy")
    fn_weights, fp_weights = weight_arrays(config["decision"])
    thresholds_payload = fit_coverage_thresholds(
        val_probs,
        val_labels,
        fn_weights=fn_weights,
        fp_weights=fp_weights,
        coverage_targets=list(config["decision"]["coverage_targets"]),
    )
    results = evaluate_selective_policy(
        test_probs,
        test_labels,
        fn_weights=fn_weights,
        fp_weights=fp_weights,
        thresholds=thresholds_payload["thresholds"],
        abstain_cost=float(config["decision"]["abstain_cost"]),
    )
    out_dir = decision_dir(model_name) / method
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "thresholds.json", thresholds_payload)
    save_json(out_dir / "results.json", results)
    save_json(out_dir / "norm_conflict_report.json", results["norm_conflict_report"])
    save_json(out_dir / "selective_policy_checks.json", results["checks"])
    leakage_report = {
        "threshold_fit_split": "val",
        "policy_eval_split": "test",
        "test_used_for_threshold_selection": False,
        "val_n": int(len(val_labels)),
        "test_n": int(len(test_labels)),
        "coverage_targets": [float(target) for target in config["decision"]["coverage_targets"]],
    }
    save_json(out_dir / "leakage_report.json", leakage_report)
    pd.DataFrame(results["curves"]).to_csv(out_dir / "results.csv", index=False)
    print(f"Selective evaluation saved to {out_dir}")


if __name__ == "__main__":
    main()
