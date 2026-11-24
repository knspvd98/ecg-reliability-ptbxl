from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ecg_reliability.evaluation.bootstrap import bootstrap_metric
from ecg_reliability.evaluation.metrics import compute_classification_metrics
from ecg_reliability.evaluation.reports import save_json
from ecg_reliability.runtime import load_runtime_config

DEFAULT_MODELS = ["xgboost", "resnet1d", "inceptiontime", "tcn", "transformer"]


def _classification_stat(metric_name: str):
    def _fn(y_true: np.ndarray, y_score: np.ndarray) -> float:
        return float(compute_classification_metrics(y_true, y_score).get(metric_name, float("nan")))

    return _fn


def _load_prediction_bundle(model_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    base = Path(f"outputs/predictions/{model_name}/raw/test")
    if not (base / "labels.npy").exists() or not (base / "probabilities.npy").exists():
        return None
    return np.load(base / "labels.npy"), np.load(base / "probabilities.npy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--n-bootstrap", type=int, default=500)
    args = parser.parse_args()

    load_runtime_config(args.config, args.override)
    rows = []
    metrics_to_bootstrap = [
        "macro_auroc",
        "macro_auprc",
        "abnormal_macro_auroc",
        "abnormal_macro_auprc",
    ]
    for model_name in DEFAULT_MODELS:
        bundle = _load_prediction_bundle(model_name)
        if bundle is None:
            continue
        labels, probabilities = bundle
        for metric_name in metrics_to_bootstrap:
            result = bootstrap_metric(
                labels,
                probabilities,
                _classification_stat(metric_name),
                n_bootstrap=int(args.n_bootstrap),
            )
            rows.append({"model": model_name, "metric": metric_name, **result})

    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables_dir / "table_bootstrap_classification_ci.csv", index=False)

    external_rows = []
    for model_name in DEFAULT_MODELS:
        predictions_path = Path(f"outputs/external/{model_name}/external_binary_predictions.csv")
        if not predictions_path.exists():
            continue
        frame = pd.read_csv(predictions_path)
        result = bootstrap_metric(
            frame["binary_abnormal_label"].to_numpy(dtype=np.float32),
            frame["p_abnormal_max"].to_numpy(dtype=np.float32),
            lambda y_true, y_score: float(roc_auc_score(y_true, y_score)),
            n_bootstrap=int(args.n_bootstrap),
        )
        external_rows.append({"model": model_name, "metric": "external_binary_auroc", **result})
    if external_rows:
        pd.DataFrame(external_rows).to_csv(tables_dir / "table_external_binary_bootstrap_ci.csv", index=False)

    save_json(
        "outputs/summary/bootstrap_summary.json",
        {"n_bootstrap": int(args.n_bootstrap), "models": [row["model"] for row in rows]},
    )
    print(tables_dir / "table_bootstrap_classification_ci.csv")


if __name__ == "__main__":
    main()
