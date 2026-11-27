from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ecg_reliability.calibration.apply import apply_calibrator, load_calibrator
from ecg_reliability.data.features import compute_engineered_features, compute_signal_quality_indicators
from ecg_reliability.data.labels import TARGET_LABELS
from ecg_reliability.data.preprocessing import DISPLAY_12_LEAD_NAMES
from ecg_reliability.decision.consistency import apply_norm_consistency_rule
from ecg_reliability.decision.harm import weight_arrays
from ecg_reliability.decision.selective import uncertainty_scores
from ecg_reliability.evaluation.calibration_metrics import reliability_bins
from ecg_reliability.evaluation.thresholding import sigmoid
from ecg_reliability.experiments.registry import DEFAULT_RUNS_ROOT, list_runs, load_run_manifest
from ecg_reliability.explainability.gradient_input import compute_gradient_x_input
from ecg_reliability.explainability.integrated_gradients import compute_integrated_gradients
from ecg_reliability.io import read_table
from ecg_reliability.models.xgboost_baseline import MultiLabelXGBoost
from ecg_reliability.robustness.perturbations import MIT_BIH_NSTDB_REFERENCE, apply_perturbation
from ecg_reliability.runtime import load_torch_model


def sanitize_json(obj: Any) -> Any:
    """Recursively convert NaN/Inf floats and NumPy types to RFC-8259-compliant JSON values."""
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _read_json_optional(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


class WorkbenchDataAccess:
    """Read real experimental artifacts and run live perturbation/model/XAI probes."""

    def __init__(
        self,
        runs_root: str | Path = DEFAULT_RUNS_ROOT,
        outputs_root: str | Path = "outputs",
    ) -> None:
        self.runs_root = Path(runs_root)
        self.outputs_root = Path(outputs_root)
        self._model_cache: dict[str, tuple[str, Any]] = {}
        self._calibrator_cache: dict[str, Any] = {}

    def list_all_runs(self) -> list[dict[str, Any]]:
        runs = list_runs(self.runs_root)
        summaries = []
        for run in runs:
            metrics = run.get("metrics") or {}
            raw_test = metrics.get("raw_test") or {}
            cal_test = metrics.get("calibrated_test") or {}
            sel_test = metrics.get("selective_test") or {}
            summaries.append(
                {
                    "run_id": run.get("run_id"),
                    "status": run.get("status"),
                    "execution_tier": run.get("execution_tier"),
                    "scientific_claim_eligible": bool(run.get("scientific_claim_eligible", False)),
                    "suite_name": run.get("suite_name"),
                    "model_name": (run.get("model_identity") or {}).get("model_name"),
                    "total_parameters": (run.get("model_identity") or {}).get("total_parameters"),
                    "created_at_utc": run.get("created_at_utc"),
                    "updated_at_utc": run.get("updated_at_utc"),
                    "git_commit": (run.get("code_version") or {}).get("git_commit"),
                    "preprocessing_fingerprint": run.get("preprocessing_fingerprint"),
                    "selected_calibration_method": metrics.get("selected_calibration_method"),
                    "macro_auroc": raw_test.get("macro_auroc"),
                    "macro_auprc": raw_test.get("macro_auprc"),
                    "macro_f1": raw_test.get("macro_f1"),
                    "raw_macro_ece": raw_test.get("macro_ece"),
                    "cal_macro_ece": cal_test.get("macro_ece"),
                    "cal_macro_adaptive_ece": cal_test.get("macro_adaptive_ece"),
                    "cal_macro_brier": cal_test.get("macro_brier"),
                    "selective_monotonic": sel_test.get("coverage_monotonic"),
                    "failure_info": run.get("failure_info"),
                }
            )
        return sanitize_json(summaries)

    def get_run_detail(self, run_id: str) -> dict[str, Any]:
        manifest = load_run_manifest(run_id, self.runs_root)
        run_dir = self.runs_root / run_id
        files: list[dict[str, Any]] = []
        if run_dir.exists():
            for path in sorted(run_dir.rglob("*")):
                if path.is_file():
                    rel = path.relative_to(run_dir).as_posix()
                    files.append(
                        {
                            "path": rel,
                            "size_bytes": int(path.stat().st_size),
                            "suffix": path.suffix.lower(),
                        }
                    )
        training_report = _read_json_optional(run_dir / "checkpoints" / "training_runtime_report.json")
        return sanitize_json(
            {
                "manifest": manifest,
                "training_runtime_report": training_report,
                "files": files,
            }
        )

    def read_run_artifact(self, run_id: str, rel_path: str, max_bytes: int = 120_000) -> dict[str, Any]:
        run_dir = (self.runs_root / run_id).resolve()
        target = (run_dir / rel_path).resolve()
        if not str(target).startswith(str(run_dir)) or not target.exists() or not target.is_file():
            raise FileNotFoundError(f"Artifact not found: {rel_path}")
        suffix = target.suffix.lower()
        if suffix in {".json", ".yaml", ".yml", ".csv", ".txt", ".log"}:
            raw = target.read_text(encoding="utf-8", errors="replace")
            truncated = len(raw) > max_bytes
            return {
                "path": rel_path,
                "kind": suffix.lstrip("."),
                "truncated": truncated,
                "content": raw[:max_bytes],
            }
        if suffix == ".npy":
            arr = np.load(target)
            return sanitize_json(
                {
                    "path": rel_path,
                    "kind": "npy",
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "min": float(np.nanmin(arr)) if arr.size else None,
                    "max": float(np.nanmax(arr)) if arr.size else None,
                    "mean": float(np.nanmean(arr)) if arr.size else None,
                }
            )
        return {
            "path": rel_path,
            "kind": "binary",
            "size_bytes": int(target.stat().st_size),
            "note": "Binary checkpoint or serialized artifact; metadata inspected via manifest.",
        }

    def _resolve_manifest_df(self, run_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        manifest = load_run_manifest(run_id, self.runs_root)
        manifest_path = (manifest.get("dataset_provenance") or {}).get("manifest_path") or (
            manifest.get("config_snapshot") or {}
        ).get("data", {}).get("cache_manifest_path")
        if not manifest_path or not Path(manifest_path).exists():
            raise FileNotFoundError(f"Dataset cache manifest unavailable for run {run_id}: {manifest_path}")
        return read_table(manifest_path), manifest

    def _load_test_prediction_map(self, run_id: str, manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
        run_dir = self.runs_root / run_id
        best_cal = (manifest.get("metrics") or {}).get("selected_calibration_method")
        raw_dir = run_dir / "predictions" / "raw" / "test"
        cal_dir = (run_dir / "predictions" / "calibrated" / str(best_cal) / "test") if best_cal else None
        if not raw_dir.exists() or not (raw_dir / "ecg_ids.npy").exists():
            return {}
        ecg_ids = np.load(raw_dir / "ecg_ids.npy")
        raw_probs = np.load(raw_dir / "probabilities.npy")
        labels = np.load(raw_dir / "labels.npy")
        cal_probs = (
            np.load(cal_dir / "probabilities.npy")
            if (cal_dir is not None and (cal_dir / "probabilities.npy").exists())
            else raw_probs
        )
        mapping: dict[int, dict[str, Any]] = {}
        for idx, eid in enumerate(ecg_ids):
            mapping[int(eid)] = {
                "raw_probabilities": {TARGET_LABELS[k]: float(raw_probs[idx, k]) for k in range(len(TARGET_LABELS))},
                "calibrated_probabilities": {TARGET_LABELS[k]: float(cal_probs[idx, k]) for k in range(len(TARGET_LABELS))},
                "true_labels": [TARGET_LABELS[k] for k in range(len(TARGET_LABELS)) if int(labels[idx, k]) == 1],
                "calibration_method": best_cal,
            }
        return mapping

    def list_signals(self, run_id: str, *, split: str = "test", limit: int = 100) -> dict[str, Any]:
        frame, manifest = self._resolve_manifest_df(run_id)
        if split != "all" and "split" in frame.columns:
            frame = frame.loc[frame["split"] == split]
        pred_map = self._load_test_prediction_map(run_id, manifest)
        records = []
        for _, row in frame.head(int(limit)).iterrows():
            eid = int(row["ecg_id"])
            true_labels = [lbl for lbl in TARGET_LABELS if lbl in row.index and int(row[lbl]) == 1]
            pred_info = pred_map.get(eid)
            records.append(
                {
                    "ecg_id": eid,
                    "patient_id": int(row["patient_id"]) if "patient_id" in row.index and pd.notna(row["patient_id"]) else None,
                    "split": str(row["split"]) if "split" in row.index else "unknown",
                    "strat_fold": int(row["strat_fold"]) if "strat_fold" in row.index and pd.notna(row["strat_fold"]) else None,
                    "age": int(row["age"]) if "age" in row.index and pd.notna(row["age"]) else None,
                    "sex": int(row["sex"]) if "sex" in row.index and pd.notna(row["sex"]) else None,
                    "true_labels": true_labels,
                    "amplitude_unit": str(row["amplitude_unit"]) if "amplitude_unit" in row.index else "normalized_sd",
                    "predictions": pred_info,
                }
            )
        return sanitize_json(
            {
                "run_id": run_id,
                "split": split,
                "total_in_split": int(len(frame)),
                "returned": len(records),
                "execution_tier": manifest.get("execution_tier"),
                "records": records,
            }
        )

    def _load_signal_array(self, run_id: str, ecg_id: int) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        frame, manifest = self._resolve_manifest_df(run_id)
        matches = frame.loc[frame["ecg_id"] == int(ecg_id)]
        if matches.empty:
            raise FileNotFoundError(f"ECG {ecg_id} not found in manifest for run {run_id}")
        row = matches.iloc[0]
        cache_path = Path(str(row["cache_path"]))
        if not cache_path.exists():
            raise FileNotFoundError(f"Signal file missing: {cache_path}")
        signal = np.load(cache_path).astype(np.float32)
        sampling_rate = int((manifest.get("config_snapshot") or {}).get("data", {}).get("sampling_rate", 100))
        normalize_flag = bool((manifest.get("config_snapshot") or {}).get("data", {}).get("normalize_per_lead", True))
        meta = {
            "ecg_id": int(row["ecg_id"]),
            "patient_id": int(row["patient_id"]) if "patient_id" in row.index and pd.notna(row["patient_id"]) else None,
            "split": str(row["split"]) if "split" in row.index else "unknown",
            "strat_fold": int(row["strat_fold"]) if "strat_fold" in row.index and pd.notna(row["strat_fold"]) else None,
            "age": int(row["age"]) if "age" in row.index and pd.notna(row["age"]) else None,
            "sex": int(row["sex"]) if "sex" in row.index and pd.notna(row["sex"]) else None,
            "true_labels": [lbl for lbl in TARGET_LABELS if lbl in row.index and int(row[lbl]) == 1],
            "sampling_rate_hz": sampling_rate,
            "duration_seconds": float(signal.shape[1] / float(sampling_rate)),
            "num_leads": int(signal.shape[0]),
            "num_samples": int(signal.shape[1]),
            "lead_names": DISPLAY_12_LEAD_NAMES[: signal.shape[0]],
            "amplitude_unit": str(row["amplitude_unit"]) if "amplitude_unit" in row.index else ("normalized_sd" if normalize_flag else "mV"),
            "normalize_per_lead": normalize_flag,
            "execution_tier": manifest.get("execution_tier"),
            "preprocessing_fingerprint": manifest.get("preprocessing_fingerprint"),
        }
        return signal, meta, manifest

    def get_signal_detail(self, run_id: str, ecg_id: int) -> dict[str, Any]:
        signal, meta, manifest = self._load_signal_array(run_id, ecg_id)
        sqi = compute_signal_quality_indicators(signal, sampling_rate=meta["sampling_rate_hz"], lead_names=meta["lead_names"])
        pred_map = self._load_test_prediction_map(run_id, manifest)
        return sanitize_json(
            {
                "metadata": meta,
                "signal": signal.tolist(),
                "sqi": sqi,
                "predictions": pred_map.get(int(ecg_id)),
            }
        )

    def _get_loaded_model(self, run_id: str, manifest: dict[str, Any]) -> tuple[str, Any]:
        if run_id in self._model_cache:
            return self._model_cache[run_id]
        run_dir = self.runs_root / run_id
        model_name = str((manifest.get("model_identity") or {}).get("model_name", ""))
        if model_name == "xgboost":
            model = MultiLabelXGBoost.load(run_dir / "checkpoints" / "model.pkl")
            self._model_cache[run_id] = ("xgboost", model)
            return "xgboost", model
        model, _ = load_torch_model(run_dir / "checkpoints" / "best.pt", device="cpu")
        self._model_cache[run_id] = ("torch", model)
        return "torch", model

    def _get_loaded_calibrator(self, run_id: str, manifest: dict[str, Any]) -> tuple[str | None, Any | None]:
        best_method = (manifest.get("metrics") or {}).get("selected_calibration_method")
        if not best_method:
            return None, None
        cache_key = f"{run_id}:{best_method}"
        if cache_key in self._calibrator_cache:
            return best_method, self._calibrator_cache[cache_key]
        cal_path = self.runs_root / run_id / "calibration" / f"{best_method}.pkl"
        if not cal_path.exists():
            return best_method, None
        calibrator = load_calibrator(cal_path)["calibrator"]
        self._calibrator_cache[cache_key] = calibrator
        return best_method, calibrator

    def _predict_single(self, run_id: str, manifest: dict[str, Any], signal: np.ndarray) -> dict[str, Any]:
        kind, model = self._get_loaded_model(run_id, manifest)
        if kind == "xgboost":
            feats = compute_engineered_features(signal)[None, :]
            logits = model.predict_logits(feats)
        else:
            with torch.no_grad():
                logits = model(torch.tensor(signal[None, ...], dtype=torch.float32)).cpu().numpy()
        raw_probs = sigmoid(logits)
        cal_method, calibrator = self._get_loaded_calibrator(run_id, manifest)
        cal_probs = apply_calibrator(calibrator, logits) if calibrator is not None else raw_probs

        decision_cfg = (manifest.get("config_snapshot") or {}).get("decision") or {}
        selective_info: dict[str, Any] | None = None
        thresholds_payload = _read_json_optional(self.runs_root / run_id / "decision" / "thresholds.json")
        if decision_cfg and thresholds_payload and thresholds_payload.get("thresholds"):
            fn_w, fp_w = weight_arrays(decision_cfg)
            raw_dec, uncertainty = uncertainty_scores(cal_probs, fn_w, fp_w)
            consistent_dec = apply_norm_consistency_rule(raw_dec)
            thresholds = thresholds_payload["thresholds"]
            pref_key = min(thresholds.keys(), key=lambda k: abs(float(k) - 0.95))
            thresh_val = float(thresholds[pref_key])
            accepted = bool(float(uncertainty[0]) <= thresh_val)
            selective_info = {
                "target_coverage": float(pref_key),
                "threshold": thresh_val,
                "uncertainty": float(uncertainty[0]),
                "action": "ACCEPT" if accepted else "ABSTAIN",
                "accepted_labels": [TARGET_LABELS[k] for k in range(len(TARGET_LABELS)) if int(consistent_dec[0, k]) == 1],
            }

        return {
            "raw_probabilities": {TARGET_LABELS[k]: float(raw_probs[0, k]) for k in range(len(TARGET_LABELS))},
            "calibrated_probabilities": {TARGET_LABELS[k]: float(cal_probs[0, k]) for k in range(len(TARGET_LABELS))},
            "calibration_method": cal_method,
            "selective_decision": selective_info,
        }

    def apply_live_perturbation(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        ecg_id = int(payload["ecg_id"])
        perturbation_name = str(payload["perturbation"])
        params = dict(payload.get("params") or {})
        signal, meta, manifest = self._load_signal_array(run_id, ecg_id)
        params.setdefault("sampling_rate", meta["sampling_rate_hz"])
        params.setdefault("seed", 42)
        perturbed = apply_perturbation(signal, perturbation_name, **params)

        clean_sqi = compute_signal_quality_indicators(signal, sampling_rate=meta["sampling_rate_hz"], lead_names=meta["lead_names"])
        pert_sqi = compute_signal_quality_indicators(perturbed, sampling_rate=meta["sampling_rate_hz"], lead_names=meta["lead_names"])

        clean_pred = None
        pert_pred = None
        pred_error = None
        try:
            clean_pred = self._predict_single(run_id, manifest, signal)
            pert_pred = self._predict_single(run_id, manifest, perturbed)
        except Exception as exc:
            pred_error = f"{exc.__class__.__name__}: {exc}"

        snr_db_actual = None
        diff = (perturbed - signal).astype(np.float64)
        noise_power = float(np.mean(diff**2))
        sig_power = float(np.mean(signal.astype(np.float64) ** 2))
        if noise_power > 1e-12 and sig_power > 1e-12:
            snr_db_actual = float(10.0 * np.log10(sig_power / noise_power))

        return sanitize_json(
            {
                "ecg_id": ecg_id,
                "perturbation": perturbation_name,
                "params": params,
                "amplitude_unit": meta["amplitude_unit"],
                "empirical_snr_db": snr_db_actual,
                "scientific_caveat": MIT_BIH_NSTDB_REFERENCE["scientific_caveat"],
                "perturbed_signal": perturbed.tolist(),
                "clean_sqi": clean_sqi,
                "perturbed_sqi": pert_sqi,
                "clean_prediction": clean_pred,
                "perturbed_prediction": pert_pred,
                "prediction_error": pred_error,
            }
        )

    def get_reliability_analysis(self, run_id: str) -> dict[str, Any]:
        manifest = load_run_manifest(run_id, self.runs_root)
        run_dir = self.runs_root / run_id
        cal_val = _read_json_optional(run_dir / "calibration" / "reports.json")
        cal_test = _read_json_optional(run_dir / "calibration" / "test_reports.json")
        rel_bins = _read_json_optional(run_dir / "calibration" / "reliability_bins.json")
        leakage = _read_json_optional(run_dir / "calibration" / "leakage_report.json")
        selective = _read_json_optional(run_dir / "decision" / "results.json")
        thresholds = _read_json_optional(run_dir / "decision" / "thresholds.json")
        bootstrap = _read_json_optional(run_dir / "analysis" / "bootstrap_ci.json")
        raw_test_metrics = _read_json_optional(run_dir / "predictions" / "raw" / "test_metrics_raw.json")

        # Compute confidence distributions per class (positives vs negatives) from real test predictions
        best_cal = (manifest.get("metrics") or {}).get("selected_calibration_method")
        cal_dir = (run_dir / "predictions" / "calibrated" / str(best_cal) / "test") if best_cal else None
        raw_dir = run_dir / "predictions" / "raw" / "test"
        prob_src = cal_dir if (cal_dir is not None and (cal_dir / "probabilities.npy").exists()) else raw_dir
        confidence_distributions: dict[str, Any] = {}
        adaptive_bins: dict[str, Any] = {}
        if prob_src.exists() and (prob_src / "probabilities.npy").exists():
            probs = np.load(prob_src / "probabilities.npy")
            labels = np.load(prob_src / "labels.npy")
            raw_probs = np.load(raw_dir / "probabilities.npy") if (raw_dir / "probabilities.npy").exists() else probs
            edges = np.linspace(0.0, 1.0, 11)
            for idx, label in enumerate(TARGET_LABELS):
                pos_hist, _ = np.histogram(probs[labels[:, idx] == 1, idx], bins=edges)
                neg_hist, _ = np.histogram(probs[labels[:, idx] == 0, idx], bins=edges)
                confidence_distributions[label] = {
                    "bin_edges": edges.tolist(),
                    "positive_counts": pos_hist.tolist(),
                    "negative_counts": neg_hist.tolist(),
                }
                if rel_bins is None:
                    adaptive_bins[label] = {
                        "raw": reliability_bins(labels, raw_probs, class_index=idx, num_bins=10),
                        "calibrated": reliability_bins(labels, probs, class_index=idx, num_bins=10),
                    }
            if rel_bins is None:
                rel_bins = {"split": "test", "best_method": best_cal, "num_bins": 10, "per_class": adaptive_bins}

        return sanitize_json(
            {
                "run_id": run_id,
                "model_name": (manifest.get("model_identity") or {}).get("model_name"),
                "execution_tier": manifest.get("execution_tier"),
                "raw_test_metrics": raw_test_metrics,
                "calibration_val_reports": cal_val,
                "calibration_test_reports": cal_test,
                "calibration_leakage_report": leakage,
                "reliability_bins": rel_bins,
                "confidence_distributions": confidence_distributions,
                "selective_results": selective,
                "selective_thresholds": thresholds,
                "bootstrap_ci": bootstrap,
            }
        )

    def get_robustness_analysis(self, run_id: str) -> dict[str, Any]:
        manifest = load_run_manifest(run_id, self.runs_root)
        run_dir = self.runs_root / run_id
        results = _read_json_optional(run_dir / "robustness" / "results.json")
        return sanitize_json(
            {
                "run_id": run_id,
                "model_name": (manifest.get("model_identity") or {}).get("model_name"),
                "execution_tier": manifest.get("execution_tier"),
                "available": results is not None,
                "results": results,
                "nstdb_reference": MIT_BIH_NSTDB_REFERENCE,
            }
        )

    def get_explainability_analysis(
        self,
        run_id: str,
        *,
        ecg_id: int | None = None,
        target_label: str | None = None,
        method: str = "ig",
    ) -> dict[str, Any]:
        manifest = load_run_manifest(run_id, self.runs_root)
        run_dir = self.runs_root / run_id
        xai_dir = run_dir / "explanations"
        stability_path = xai_dir / "stability.csv"
        stability_rows = pd.read_csv(stability_path).to_dict(orient="records") if stability_path.exists() else []
        sanity = _read_json_optional(xai_dir / "sanity_checks.json")
        failures = _read_json_optional(run_dir / "analysis" / "failure_analysis.json")
        disagreement = _read_json_optional(self.runs_root / "comparisons" / "model_disagreement.json")

        saved_items: list[dict[str, Any]] = []
        if xai_dir.exists():
            for npy in sorted(xai_dir.glob("ecg_*_ig.npy")):
                parts = npy.stem.split("_")
                if len(parts) >= 4:
                    saved_items.append({"ecg_id": int(parts[1]), "target_label": parts[2].upper(), "stem": "_".join(parts[:3])})

        selected_ecg = ecg_id
        selected_label = target_label.upper() if target_label else None
        if selected_ecg is None and saved_items:
            selected_ecg = saved_items[0]["ecg_id"]
            selected_label = selected_label or saved_items[0]["target_label"]

        attribution_payload: dict[str, Any] | None = None
        if selected_ecg is not None:
            selected_label = selected_label or "MI"
            stem = f"ecg_{selected_ecg}_{selected_label.lower()}"
            signal, meta, _ = self._load_signal_array(run_id, selected_ecg)
            attr_path = xai_dir / f"{stem}_{method}.npy"
            attr_arr: np.ndarray | None = None
            source = "saved_artifact"
            if attr_path.exists():
                attr_arr = np.load(attr_path).astype(np.float32)
            elif (manifest.get("model_identity") or {}).get("model_name") != "xgboost":
                kind, model = self._get_loaded_model(run_id, manifest)
                if kind == "torch":
                    sig_t = torch.tensor(signal[None, ...], dtype=torch.float32)
                    t_cls = TARGET_LABELS.index(selected_label) if selected_label in TARGET_LABELS else 0
                    if method == "gx":
                        attr_arr = compute_gradient_x_input(model, sig_t, target_class=t_cls).cpu().numpy()[0]
                    else:
                        attr_arr = compute_integrated_gradients(model, sig_t, target_class=t_cls, steps=16).cpu().numpy()[0]
                    source = "computed_on_demand_from_checkpoint"

            if attr_arr is not None:
                lead_imp = np.abs(attr_arr).sum(axis=1)
                total_imp = float(lead_imp.sum()) or 1.0
                attribution_payload = {
                    "ecg_id": int(selected_ecg),
                    "target_label": selected_label,
                    "method": method,
                    "source": source,
                    "metadata": meta,
                    "signal": signal.tolist(),
                    "attribution": attr_arr.tolist(),
                    "lead_importance": [
                        {"lead": meta["lead_names"][i], "share": float(lead_imp[i] / total_imp), "abs_sum": float(lead_imp[i])}
                        for i in range(len(meta["lead_names"]))
                    ],
                }

        return sanitize_json(
            {
                "run_id": run_id,
                "model_name": (manifest.get("model_identity") or {}).get("model_name"),
                "execution_tier": manifest.get("execution_tier"),
                "saved_attributions": saved_items,
                "selected_attribution": attribution_payload,
                "stability_rows": stability_rows,
                "sanity_checks": sanity,
                "failure_analysis": failures,
                "model_disagreement": disagreement,
                "method_guides": [
                    {
                        "method": "Integrated Gradients (IG)",
                        "measures": "Path-integrated input sensitivity from a zero baseline to the observed 12-lead ECG, satisfying completeness sum(IG) ≈ f(x) - f(0).",
                        "limitations": "Depends on the zero baseline assumption; highlights model-sensitive morphology rather than causal clinical mechanisms.",
                    },
                    {
                        "method": "Gradient × Input (G×I)",
                        "measures": "First-order local linear sensitivity weighted by signal amplitude.",
                        "limitations": "Sensitive to local gradient noise and saturation; may miss distributed morphological interactions.",
                    },
                    {
                        "method": "Sliding-Window Occlusion",
                        "measures": "Direct change in target logit when zeroing out local temporal windows lead-by-lead.",
                        "limitations": "Computationally heavier and zeroing segments can create out-of-distribution signal artifacts.",
                    },
                    {
                        "method": "Sanity Checks (Randomization & Deletion)",
                        "measures": "Parameter randomization tests whether attributions depend on learned weights; deletion AUC vs random control tests faithfulness.",
                        "limitations": "Passing sanity checks is necessary but not sufficient for clinical interpretability.",
                    },
                ],
            }
        )

    def get_comparisons(self) -> dict[str, Any]:
        paired = _read_json_optional(self.runs_root / "comparisons" / "paired_comparisons.json")
        disagreement = _read_json_optional(self.runs_root / "comparisons" / "model_disagreement.json")
        return sanitize_json(
            {
                "runs": self.list_all_runs(),
                "paired_comparisons": paired,
                "model_disagreement": disagreement,
            }
        )
