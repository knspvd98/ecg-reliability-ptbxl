from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from ecg_reliability.evaluation.model_selection import resolve_selected_model
from ecg_reliability.io import read_table
from ecg_reliability.runtime import load_runtime_config, model_name_from_config


def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--selection-manifest", default="outputs/summary/model_selection.json")
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    configured_model = model_name_from_config(config, args.model_name)
    selected_model = resolve_selected_model(
        configured_model if args.model_name is not None else None,
        args.selection_manifest,
        synthetic_smoke=False,
        synthetic_fallback_model=None,
    )
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    index_df = read_table(config["data"]["index_path"])
    label_rows = []
    for label in config["data"]["target_labels"]:
        count = int(index_df[label].sum())
        prevalence = float(index_df[label].mean())
        label_rows.append({"label": label, "count": count, "prevalence": prevalence})
    pd.DataFrame(label_rows).to_csv(tables_dir / "table_1_label_distribution.csv", index=False)

    raw_rows = []
    for metrics_path in sorted(Path("outputs/predictions").glob("*/raw/test_metrics_raw.json")):
        model_name = metrics_path.parts[-3]
        metrics = _read_json(metrics_path)
        raw_rows.append(
            {
                "model": model_name,
                "macro_auroc": metrics.get("macro_auroc"),
                "macro_auprc": metrics.get("macro_auprc"),
                "abnormal_macro_auroc": metrics.get("abnormal_macro_auroc"),
                "abnormal_macro_auprc": metrics.get("abnormal_macro_auprc"),
                "macro_f1": metrics.get("macro_f1"),
                "micro_f1": metrics.get("micro_f1"),
                "hamming_loss": metrics.get("hamming_loss"),
            }
        )
    pd.DataFrame(raw_rows).to_csv(tables_dir / "table_2_model_raw_metrics.csv", index=False)

    calibration_rows = []
    for reports_path in sorted(Path("outputs/calibration").glob("*/reports.json")):
        model_name = reports_path.parts[-2]
        reports = _read_json(reports_path)
        best = reports.get("best_method")
        for method, payload in reports.items():
            if method == "best_method":
                continue
            calibration_rows.append(
                {
                    "model": model_name,
                    "method": method,
                    "selected_best_method": method == best,
                    "macro_ece": payload.get("macro_ece"),
                    "macro_adaptive_ece": payload.get("macro_adaptive_ece"),
                    "macro_brier": payload.get("macro_brier"),
                }
            )
    pd.DataFrame(calibration_rows).to_csv(tables_dir / "table_3_calibration_comparison.csv", index=False)

    decision_rows = []
    for results_path in sorted(Path("outputs/decision").glob("*/*/results.json")):
        model_name = results_path.parts[-3]
        method = results_path.parts[-2]
        payload = _read_json(results_path)
        for curve in payload.get("curves", []):
            decision_rows.append(
                {
                    "model": model_name,
                    "method": method,
                    **curve,
                }
            )
    pd.DataFrame(decision_rows).to_csv(tables_dir / "table_4_selective_policy.csv", index=False)

    robustness_delta_path = Path(f"outputs/robustness/{selected_model}/delta_summary.csv")
    if robustness_delta_path.exists():
        pd.read_csv(robustness_delta_path).to_csv(tables_dir / "table_5_robustness_deltas.csv", index=False)
    else:
        pd.DataFrame(columns=["perturbation"]).to_csv(tables_dir / "table_5_robustness_deltas.csv", index=False)

    xai_path = Path(f"outputs/explanations/{selected_model}/stability.csv")
    if xai_path.exists():
        pd.read_csv(xai_path).to_csv(tables_dir / "table_6_xai_stability.csv", index=False)
    else:
        pd.DataFrame(columns=["ecg_id"]).to_csv(tables_dir / "table_6_xai_stability.csv", index=False)

    external_path = Path(f"outputs/external/{selected_model}/external_binary_results.csv")
    if external_path.exists():
        external_frame = pd.read_csv(external_path)
        external_frame.insert(0, "table_title", "External binary abnormal-vs-normal validation")
        external_frame.to_csv(tables_dir / "table_7_external_binary_validation.csv", index=False)

    print(tables_dir)


if __name__ == "__main__":
    main()
