from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ecg_reliability.calibration.apply import (
    apply_calibrator,
    evaluate_calibration_methods,
    save_calibrator,
    select_best_method,
)
from ecg_reliability.evaluation.calibration_metrics import compute_calibration_report
from ecg_reliability.evaluation.reports import save_json
from ecg_reliability.runtime import (
    calibration_dir,
    load_prediction_arrays,
    load_runtime_config,
    model_name_from_config,
    prediction_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    model_name = model_name_from_config(config, args.model_name)
    val = load_prediction_arrays(model_name, "val", namespace="raw")
    test = load_prediction_arrays(model_name, "test", namespace="raw")
    methods = list(config["calibration"]["methods"])
    num_bins = int(config["calibration"]["num_bins"])
    results = evaluate_calibration_methods(methods, val["logits"], val["labels"], num_bins=num_bins)
    best_method = select_best_method(results)
    cal_dir = calibration_dir(model_name)
    cal_dir.mkdir(parents=True, exist_ok=True)
    reports = {method: payload["report"] for method, payload in results.items()}
    reports["best_method"] = best_method
    save_json(cal_dir / "reports.json", reports)

    # Also compute held-out test calibration reports (evaluation only — never used for fitting or selection)
    test_reports: dict[str, object] = {
        "raw": compute_calibration_report(test["labels"], test["probabilities"], num_bins=num_bins),
        "selected_method_from_val": best_method,
    }

    leakage_report = {
        "calibration_fit_split": "val",
        "calibration_selection_split": "val",
        "calibration_applied_splits": ["val", "test"],
        "test_used_for_fitting": False,
        "test_used_for_method_selection": False,
        "val_n": int(len(val["labels"])),
        "test_n": int(len(test["labels"])),
        "methods": methods,
        "selected_method": best_method,
        "warning_if_small_validation": bool(len(val["labels"]) < 500),
        "debug_not_scientific": bool(len(val["labels"]) < 500),
    }
    save_json(cal_dir / "leakage_report.json", leakage_report)

    pred_root = prediction_dir(model_name, "calibrated")
    for method, payload in results.items():
        save_calibrator(cal_dir / f"{method}.pkl", method, payload["calibrator"], payload["report"])
        method_dir = pred_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for split_name, bundle in {"val": val, "test": test}.items():
            probs = apply_calibrator(payload["calibrator"], bundle["logits"])
            if split_name == "test":
                test_reports[method] = compute_calibration_report(bundle["labels"], probs, num_bins=num_bins)
            split_dir = method_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            np.save(split_dir / "probabilities.npy", probs.astype(np.float32))
            np.save(split_dir / "labels.npy", bundle["labels"].astype(np.float32))
            np.save(split_dir / "logits.npy", bundle["logits"].astype(np.float32))
            np.save(split_dir / "ecg_ids.npy", bundle["ecg_ids"].astype(np.int64))
    save_json(cal_dir / "test_reports.json", test_reports)
    print(f"Best calibration method for {model_name}: {best_method}")


if __name__ == "__main__":
    main()
