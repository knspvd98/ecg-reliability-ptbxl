from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_reliability.evaluation.model_selection import resolve_calibration_method, resolve_selected_model
from ecg_reliability.evaluation.reports import save_json
from ecg_reliability.runtime import load_runtime_config, model_name_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--calibration-method", default=None)
    parser.add_argument("--selection-manifest", default="outputs/summary/model_selection.json")
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    configured_model = model_name_from_config(config, args.model_name)
    selected_model = resolve_selected_model(
        configured_model if args.model_name is not None else None,
        args.selection_manifest,
        synthetic_smoke=args.synthetic_smoke,
        synthetic_fallback_model=configured_model,
    )
    selected_method = resolve_calibration_method(selected_model, args.calibration_method)
    if selected_method is None and not args.synthetic_smoke:
        raise FileNotFoundError(f"Calibration method could not be resolved for model={selected_model}.")

    summary = {
        "selected_deep_model": selected_model,
        "selected_calibration_method": selected_method,
        "models": {},
        "calibration": {},
        "decision": {},
        "robustness": {},
        "explanations": {},
        "external": {},
    }

    for metrics_path in Path("outputs/predictions").glob("*/raw/test_metrics_raw.json"):
        model_name = metrics_path.parts[-3]
        with metrics_path.open("r", encoding="utf-8") as handle:
            summary["models"][model_name] = json.load(handle)

    for cal_path in Path("outputs/calibration").glob("*/reports.json"):
        model_name = cal_path.parts[-2]
        with cal_path.open("r", encoding="utf-8") as handle:
            summary["calibration"][model_name] = json.load(handle)

    for decision_path in Path("outputs/decision").glob("*/*/results.json"):
        model_name = decision_path.parts[-3]
        method = decision_path.parts[-2]
        with decision_path.open("r", encoding="utf-8") as handle:
            summary["decision"].setdefault(model_name, {})[method] = json.load(handle)

    for robustness_path in Path("outputs/robustness").glob("*/results.json"):
        model_name = robustness_path.parts[-2]
        with robustness_path.open("r", encoding="utf-8") as handle:
            summary["robustness"][model_name] = json.load(handle)

    for xai_path in Path("outputs/explanations").glob("*/stability.csv"):
        model_name = xai_path.parts[-2]
        summary["explanations"][model_name] = pd.read_csv(xai_path).mean(numeric_only=True).to_dict()

    for external_path in Path("outputs/external").glob("*/external_binary_results.json"):
        model_name = external_path.parts[-2]
        with external_path.open("r", encoding="utf-8") as handle:
            summary["external"][model_name] = json.load(handle)

    summary["selected_model_artifacts"] = {
        "raw_metrics": summary["models"].get(selected_model),
        "calibration_report": summary["calibration"].get(selected_model),
        "decision_report": (summary["decision"].get(selected_model, {}) or {}).get(selected_method) if selected_method else None,
        "robustness_report": summary["robustness"].get(selected_model),
        "explanation_summary": summary["explanations"].get(selected_model),
        "external_binary_report": summary["external"].get(selected_model),
    }
    summary["xgboost_baseline"] = {
        "raw_metrics": summary["models"].get("xgboost"),
        "calibration_report": summary["calibration"].get("xgboost"),
        "decision_report": summary["decision"].get("xgboost"),
    }

    summary_dir = Path("outputs/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary_dir / "results_summary.json", summary)

    rows = []
    for model_name, metrics in summary["models"].items():
        rows.append(
            {
                "model": model_name,
                "macro_auroc": metrics.get("macro_auroc"),
                "macro_auprc": metrics.get("macro_auprc"),
                "macro_f1": metrics.get("macro_f1"),
                "selected_deep_model": model_name == selected_model,
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(summary_dir / "manuscript_ready_tables.csv", index=False)
    print(f"Summary written to {summary_dir}")


if __name__ == "__main__":
    main()
