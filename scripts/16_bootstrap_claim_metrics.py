from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ecg_reliability.evaluation.bootstrap import bootstrap_metric, paired_bootstrap_comparison
from ecg_reliability.evaluation.calibration_metrics import compute_calibration_report
from ecg_reliability.evaluation.metrics import compute_classification_metrics
from ecg_reliability.evaluation.reports import save_json
from ecg_reliability.io import read_table
from ecg_reliability.runtime import load_runtime_config

DEFAULT_MODELS = ["xgboost", "simple_cnn", "resnet1d", "inceptiontime", "tcn", "transformer"]


def _classification_stat(metric_name: str):
    def _fn(y_true: np.ndarray, y_score: np.ndarray) -> float:
        return float(compute_classification_metrics(y_true, y_score).get(metric_name, float("nan")))

    return _fn


def _calibration_stat(metric_name: str):
    def _fn(y_true: np.ndarray, y_score: np.ndarray) -> float:
        return float(compute_calibration_report(y_true, y_score).get(metric_name, float("nan")))

    return _fn


def _load_prediction_bundle(model_name: str) -> dict[str, np.ndarray] | None:
    base = Path(f"outputs/predictions/{model_name}/raw/test")
    if not (base / "labels.npy").exists() or not (base / "probabilities.npy").exists():
        return None
    bundle = {
        "labels": np.load(base / "labels.npy"),
        "probabilities": np.load(base / "probabilities.npy"),
    }
    if (base / "ecg_ids.npy").exists():
        bundle["ecg_ids"] = np.load(base / "ecg_ids.npy")
    return bundle


def _resolve_patient_clusters(config: dict, ecg_ids: np.ndarray | None) -> np.ndarray | None:
    if ecg_ids is None:
        return None
    for path_key in ("cache_manifest_path", "index_path"):
        candidate = config.get("data", {}).get(path_key)
        if not candidate or not Path(candidate).exists():
            continue
        frame = read_table(candidate)
        if "ecg_id" in frame.columns and "patient_id" in frame.columns and frame["patient_id"].notna().any():
            lookup = frame.drop_duplicates(subset=["ecg_id"]).set_index("ecg_id")["patient_id"]
            mapped = pd.Series(ecg_ids).map(lookup)
            if mapped.notna().all():
                return mapped.to_numpy()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    rows = []
    paired_rows = []
    metrics_to_bootstrap = [
        ("macro_auroc", _classification_stat("macro_auroc")),
        ("macro_auprc", _classification_stat("macro_auprc")),
        ("abnormal_macro_auroc", _classification_stat("abnormal_macro_auroc")),
        ("abnormal_macro_auprc", _classification_stat("abnormal_macro_auprc")),
        ("macro_ece", _calibration_stat("macro_ece")),
        ("macro_brier", _calibration_stat("macro_brier")),
    ]

    bundles: dict[str, dict[str, np.ndarray]] = {}
    for model_name in DEFAULT_MODELS:
        bundle = _load_prediction_bundle(model_name)
        if bundle is None:
            continue
        bundles[model_name] = bundle
        labels = bundle["labels"]
        probabilities = bundle["probabilities"]
        cluster_ids = _resolve_patient_clusters(config, bundle.get("ecg_ids"))
        for metric_name, stat_fn in metrics_to_bootstrap:
            result = bootstrap_metric(
                labels,
                probabilities,
                stat_fn,
                n_bootstrap=int(args.n_bootstrap),
                seed=int(args.seed),
                cluster_ids=cluster_ids,
                ci_level=float(args.ci_level),
            )
            rows.append({"model": model_name, "metric": metric_name, **result})

    # Paired bootstrap comparisons across all models sharing aligned test ecg_ids
    for model_a, model_b in combinations(sorted(bundles.keys()), 2):
        ba, bb = bundles[model_a], bundles[model_b]
        if len(ba["labels"]) != len(bb["labels"]):
            continue
        if "ecg_ids" in ba and "ecg_ids" in bb and not np.array_equal(ba["ecg_ids"], bb["ecg_ids"]):
            continue
        cluster_ids = _resolve_patient_clusters(config, ba.get("ecg_ids"))
        for metric_name in ("macro_auroc", "macro_auprc", "macro_brier"):
            stat_fn = _calibration_stat(metric_name) if metric_name == "macro_brier" else _classification_stat(metric_name)
            comp = paired_bootstrap_comparison(
                ba["labels"],
                ba["probabilities"],
                bb["probabilities"],
                stat_fn,
                n_bootstrap=int(args.n_bootstrap),
                seed=int(args.seed),
                cluster_ids=cluster_ids,
                ci_level=float(args.ci_level),
            )
            paired_rows.append({"model_a": model_a, "model_b": model_b, "metric": metric_name, **comp})

    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables_dir / "table_bootstrap_classification_ci.csv", index=False)
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(tables_dir / "table_paired_model_comparisons.csv", index=False)

    external_rows = []
    for model_name in DEFAULT_MODELS:
        predictions_path = Path(f"outputs/external/{model_name}/external_binary_predictions.csv")
        if not predictions_path.exists():
            continue
        frame = pd.read_csv(predictions_path)
        ext_clusters = (
            frame["patient_id"].to_numpy()
            if "patient_id" in frame.columns and frame["patient_id"].notna().all()
            else None
        )
        result = bootstrap_metric(
            frame["binary_abnormal_label"].to_numpy(dtype=np.float32),
            frame["p_abnormal_max"].to_numpy(dtype=np.float32),
            lambda y_true, y_score: float(roc_auc_score(y_true, y_score)),
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed),
            cluster_ids=ext_clusters,
            ci_level=float(args.ci_level),
        )
        external_rows.append({"model": model_name, "metric": "external_binary_auroc", **result})
    if external_rows:
        pd.DataFrame(external_rows).to_csv(tables_dir / "table_external_binary_bootstrap_ci.csv", index=False)

    save_json(
        "outputs/summary/bootstrap_summary.json",
        {
            "n_bootstrap": int(args.n_bootstrap),
            "ci_level": float(args.ci_level),
            "models": sorted(bundles.keys()),
            "n_paired_comparisons": int(len(paired_rows)),
            "methodology_note": (
                "Uses patient-cluster percentile bootstrap when patient_id is available in the manifest, "
                "and record-level bootstrap otherwise. Published study figures remain distinct from these "
                "research-extension paired and cluster-adjusted intervals."
            ),
        },
    )
    print(tables_dir / "table_bootstrap_classification_ci.csv")


if __name__ == "__main__":
    main()
