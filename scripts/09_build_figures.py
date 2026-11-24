from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_reliability.evaluation.calibration_metrics import reliability_bins
from ecg_reliability.evaluation.model_selection import resolve_calibration_method, resolve_selected_model
from ecg_reliability.io import read_table
from ecg_reliability.plotting.plot_calibration import plot_reliability_comparison, plot_risk_coverage
from ecg_reliability.plotting.plot_dataset import plot_label_distribution
from ecg_reliability.plotting.plot_model_comparison import plot_model_comparison
from ecg_reliability.plotting.plot_selective import plot_expected_harm_vs_coverage
from ecg_reliability.runtime import load_runtime_config, model_name_from_config


def build_pipeline_figure(output_path: str) -> None:
    plt.figure(figsize=(14, 3), dpi=200)
    steps = [
        "PTB-XL",
        "Preprocess\n+ Cache",
        "Model\nComparison",
        "Calibration",
        "Selective\nDecision",
        "Robustness\n+ XAI",
    ]
    for idx, step in enumerate(steps):
        plt.text(idx, 0.5, step, ha="center", va="center", bbox={"boxstyle": "round,pad=0.4", "facecolor": "#D9EAF7"})
        if idx < len(steps) - 1:
            plt.arrow(idx + 0.18, 0.5, 0.55, 0.0, head_width=0.04, head_length=0.07, length_includes_head=True)
    plt.axis("off")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


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
    method = resolve_calibration_method(selected_model, args.calibration_method)
    if method is None and not args.synthetic_smoke:
        raise FileNotFoundError(f"Calibration method could not be resolved for model={selected_model}.")

    figure_dir = Path("outputs/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    build_pipeline_figure(figure_dir / "figure_1_pipeline.png")

    index_df = read_table(config["data"]["index_path"])
    plot_label_distribution(index_df, figure_dir / "figure_2_dataset_labels.png")

    metric_rows = []
    for metrics_path in Path("outputs/predictions").glob("*/raw/test_metrics_raw.csv"):
        model_name = metrics_path.parts[-3]
        frame = pd.read_csv(metrics_path)
        subset = frame.loc[(frame["scope"] == "aggregate") & (frame["metric"].isin(["macro_auroc", "macro_auprc"]))]
        subset = subset.assign(model=model_name)
        metric_rows.append(subset[["model", "metric", "value"]])
    if metric_rows:
        plot_model_comparison(pd.concat(metric_rows, axis=0), figure_dir / "figure_3_model_comparison.png")

    if method is not None:
        raw_dir = Path(f"outputs/predictions/{selected_model}/raw/val")
        cal_dir = Path(f"outputs/predictions/{selected_model}/calibrated/{method}/val")
        if raw_dir.exists() and cal_dir.exists():
            import numpy as np

            labels = np.load(raw_dir / "labels.npy")
            raw_probs = np.load(raw_dir / "probabilities.npy")
            cal_probs = np.load(cal_dir / "probabilities.npy")
            raw_bins = reliability_bins(labels, raw_probs, class_index=1)
            cal_bins = reliability_bins(labels, cal_probs, class_index=1)
            plot_reliability_comparison(raw_bins, cal_bins, figure_dir / "figure_4_calibration.png")

        decision_path = Path(f"outputs/decision/{selected_model}/{method}/results.json")
        if decision_path.exists():
            with decision_path.open("r", encoding="utf-8") as handle:
                selective = json.load(handle)
            plot_risk_coverage(selective["curves"], figure_dir / "figure_4_risk_coverage.png")
            plot_expected_harm_vs_coverage(selective["curves"], figure_dir / "figure_4_expected_harm.png")

    xai_image = Path(f"outputs/explanations/{selected_model}/example_xai.png")
    if xai_image.exists():
        from shutil import copyfile

        copyfile(xai_image, figure_dir / "figure_5_xai.png")
    print(f"Figures written to {figure_dir}")


if __name__ == "__main__":
    main()
