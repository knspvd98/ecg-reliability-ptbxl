from __future__ import annotations

import copy
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ecg_reliability.calibration.apply import (
    apply_calibrator,
    evaluate_calibration_methods,
    save_calibrator,
    select_best_method,
)
from ecg_reliability.data.cache import (
    CACHE_SCHEMA_VERSION,
    atomic_save_npy,
    atomic_write_json,
    atomic_write_table,
    compute_preprocessing_fingerprint,
)
from ecg_reliability.data.dataset import CachedECGDataset
from ecg_reliability.data.features import compute_engineered_features
from ecg_reliability.data.labels import TARGET_LABELS
from ecg_reliability.data.preprocessing import STANDARD_12_LEAD_ORDER, normalize_per_lead
from ecg_reliability.data.ptbxl_index import compute_patient_overlap_report
from ecg_reliability.decision.harm import weight_arrays
from ecg_reliability.decision.selective import evaluate_selective_policy, fit_coverage_thresholds
from ecg_reliability.evaluation.bootstrap import bootstrap_metric, paired_bootstrap_comparison
from ecg_reliability.evaluation.calibration_metrics import compute_calibration_report, reliability_bins
from ecg_reliability.evaluation.failure_analysis import analyze_model_disagreement, analyze_model_failures
from ecg_reliability.evaluation.metrics import compute_classification_metrics
from ecg_reliability.evaluation.reports import save_json, save_metrics_csv, save_prediction_bundle
from ecg_reliability.evaluation.thresholding import sigmoid
from ecg_reliability.experiments.registry import (
    DEFAULT_RUNS_ROOT,
    create_run,
    file_sha256,
    mark_run_completed,
    mark_run_failed,
    mark_stage_status,
    prepare_resumable_run,
    save_run_manifest,
)
from ecg_reliability.explainability.gradient_input import compute_gradient_x_input
from ecg_reliability.explainability.integrated_gradients import compute_integrated_gradients
from ecg_reliability.explainability.occlusion import compute_occlusion
from ecg_reliability.explainability.sanity_checks import run_attribution_sanity_suite
from ecg_reliability.explainability.selection import build_xai_selection_manifests
from ecg_reliability.explainability.stability import (
    cosine_similarity,
    lead_attribution_entropy,
    spearman_similarity,
    topk_overlap,
)
from ecg_reliability.io import read_table
from ecg_reliability.models.xgboost_baseline import MultiLabelXGBoost
from ecg_reliability.performance.device import configure_torch_runtime, resolve_device
from ecg_reliability.robustness.evaluate_robustness import (
    evaluate_decision_layer_under_perturbations,
    evaluate_under_perturbations,
)
from ecg_reliability.robustness.perturbations import (
    MIT_BIH_NSTDB_REFERENCE,
    build_severity_perturbation_suite,
    gaussian_noise,
    lead_dropout_single,
)
from ecg_reliability.runtime import load_torch_model
from ecg_reliability.seed import seed_everything
from ecg_reliability.training.train_torch import build_dataloader, fit_torch_model, predict_torch_model
from ecg_reliability.training.train_xgboost import fit_xgboost_model, predict_xgboost_model

ALL_SUITES = (
    "discrimination",
    "calibration",
    "selective",
    "robustness",
    "explainability",
    "bootstrap",
    "failure_analysis",
)

# Lead scaling matrix for synthetic PQRST morphology (I, II, III, aVR, aVL, aVF, V1..V6)
_LEAD_R_SCALE = np.array([0.75, 1.15, 0.45, -0.95, 0.35, 0.80, -0.35, 0.25, 0.65, 1.10, 1.25, 1.00], dtype=np.float32)
_LEAD_S_SCALE = np.array([-0.15, -0.20, -0.10, 0.15, -0.10, -0.15, -0.95, -0.80, -0.45, -0.20, -0.10, -0.05], dtype=np.float32)
_LEAD_T_SCALE = np.array([0.22, 0.32, 0.12, -0.28, 0.12, 0.22, -0.05, 0.25, 0.35, 0.38, 0.32, 0.24], dtype=np.float32)
_LEAD_P_SCALE = np.array([0.10, 0.16, 0.06, -0.14, 0.05, 0.11, 0.04, 0.08, 0.09, 0.10, 0.09, 0.08], dtype=np.float32)


def synthesize_ecg_waveform(
    labels: np.ndarray,
    *,
    sampling_rate: int = 100,
    signal_length: int = 1000,
    normalize: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthesize a 12-lead ECG-like waveform for software smoke testing and workbench demos.

    Explicitly NOT real clinical data. Produces P-Q-R-S-T complexes across 12 standard leads
    with distinct synthetic morphological features per diagnostic superclass so models can
    learn non-trivial patterns during fast CPU smoke tests.
    """
    gen = rng if rng is not None else np.random.default_rng(42)
    t = np.arange(signal_length, dtype=np.float32) / float(sampling_rate)
    duration = float(signal_length) / float(sampling_rate)
    heart_rate_bpm = float(gen.uniform(58.0, 92.0))
    rr = 60.0 / heart_rate_bpm
    beat_phase = float(gen.uniform(0.15, rr * 0.7))

    is_mi = bool(labels[TARGET_LABELS.index("MI")])
    is_sttc = bool(labels[TARGET_LABELS.index("STTC")])
    is_cd = bool(labels[TARGET_LABELS.index("CD")])
    is_hyp = bool(labels[TARGET_LABELS.index("HYP")])

    qrs_width = 0.036 if is_cd else 0.017
    signal = np.zeros((12, signal_length), dtype=np.float32)
    beat_time = beat_phase
    while beat_time < duration + 0.3:
        jitter = float(gen.normal(0.0, 0.012))
        tc = beat_time + jitter
        p_wave = np.exp(-0.5 * ((t - (tc - 0.16)) / 0.028) ** 2)
        q_wave = -0.12 * np.exp(-0.5 * ((t - (tc - 0.030)) / 0.010) ** 2)
        r_wave = np.exp(-0.5 * ((t - tc) / qrs_width) ** 2)
        s_wave = np.exp(-0.5 * ((t - (tc + 0.032 + (0.018 if is_cd else 0.0))) / (qrs_width * 1.15)) ** 2)
        t_wave = np.exp(-0.5 * ((t - (tc + 0.26)) / 0.055) ** 2)

        for lead_idx in range(12):
            r_amp = float(_LEAD_R_SCALE[lead_idx]) * float(gen.uniform(0.92, 1.08))
            s_amp = float(_LEAD_S_SCALE[lead_idx])
            t_amp = float(_LEAD_T_SCALE[lead_idx])
            p_amp = float(_LEAD_P_SCALE[lead_idx])

            if is_hyp:
                if lead_idx in (6, 7):
                    s_amp *= 1.85
                elif lead_idx in (9, 10, 11):
                    r_amp *= 1.75
            if is_sttc and lead_idx in (0, 4, 9, 10, 11):
                t_amp = -abs(t_amp) * 1.45

            beat = p_amp * p_wave + q_wave + r_amp * r_wave + s_amp * s_wave + t_amp * t_wave

            if is_mi and lead_idx in (1, 2, 5, 7, 8, 9):
                # Synthetic ST elevation + deeper Q wave
                st_elev = 0.32 * np.exp(-0.5 * ((t - (tc + 0.10)) / 0.045) ** 2)
                path_q = -0.45 * np.exp(-0.5 * ((t - (tc - 0.032)) / 0.014) ** 2)
                beat = beat + st_elev + path_q
            if is_sttc and lead_idx in (0, 4, 9, 10, 11):
                st_dep = -0.24 * np.exp(-0.5 * ((t - (tc + 0.10)) / 0.045) ** 2)
                beat = beat + st_dep
            if is_cd and lead_idx in (6, 7, 10, 11):
                # Synthetic notched R' deflection
                notch = 0.55 * np.exp(-0.5 * ((t - (tc + 0.048)) / 0.018) ** 2)
                beat = beat + notch

            signal[lead_idx] += beat.astype(np.float32)
        beat_time += rr

    # Mild physiological baseline drift + sensor floor noise
    drift_freq = float(gen.uniform(0.15, 0.30))
    drift = 0.03 * np.sin(2.0 * np.pi * drift_freq * t + float(gen.uniform(0, 6.28)))
    signal += drift[None, :].astype(np.float32)
    signal += gen.normal(0.0, 0.025, size=signal.shape).astype(np.float32)
    if normalize:
        signal = normalize_per_lead(signal)
    return signal.astype(np.float32)


def generate_synthetic_ecg_cohort(
    output_root: str | Path,
    *,
    n_train_patients: int = 36,
    n_val_patients: int = 12,
    n_test_patients: int = 16,
    records_per_patient: int = 2,
    sampling_rate: int = 100,
    signal_length: int = 1000,
    normalize: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a patient-clustered synthetic 12-lead ECG cohort for reproducible CPU testing."""
    root = Path(output_root)
    signals_dir = root / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    ecg_id = 10001
    patient_id = 1001
    split_specs = [
        ("train", n_train_patients, [1, 2, 3, 4, 5, 6, 7, 8]),
        ("val", n_val_patients, [9]),
        ("test", n_test_patients, [10]),
    ]
    for split_name, n_patients, folds in split_specs:
        for p_offset in range(n_patients):
            fold = folds[p_offset % len(folds)]
            # Assign patient-level primary condition profile with multi-label co-occurrence
            if p_offset % 5 == 0:
                base_labels = np.array([1, 0, 0, 0, 0], dtype=int)
            else:
                base_labels = np.zeros(5, dtype=int)
                primary = 1 + (p_offset % 4)
                base_labels[primary] = 1
                if rng.random() < 0.30:
                    secondary = int(rng.choice([1, 2, 3, 4]))
                    base_labels[secondary] = 1
            age = int(rng.integers(32, 84))
            sex = int(rng.integers(0, 2))
            for _ in range(records_per_patient):
                labels = base_labels.copy()
                waveform = synthesize_ecg_waveform(
                    labels,
                    sampling_rate=sampling_rate,
                    signal_length=signal_length,
                    normalize=normalize,
                    rng=rng,
                )
                cache_path = signals_dir / f"{ecg_id}.npy"
                atomic_save_npy(cache_path, waveform)
                rows.append(
                    {
                        "ecg_id": ecg_id,
                        "patient_id": patient_id,
                        "strat_fold": fold,
                        "split": split_name,
                        "age": age,
                        "sex": sex,
                        "cache_path": str(cache_path),
                        "amplitude_unit": "normalized_sd" if normalize else "mV",
                        "dataset_tier": "synthetic_smoke",
                        **{label: int(labels[idx]) for idx, label in enumerate(TARGET_LABELS)},
                    }
                )
                ecg_id += 1
            patient_id += 1

    manifest_df = pd.DataFrame(rows)
    manifest_path = root / "synthetic_manifest.csv"
    atomic_write_table(manifest_df, manifest_path)
    overlap = compute_patient_overlap_report(manifest_df)
    summary = {
        "status": "synthetic_smoke_cohort_prepared",
        "dataset_identity": "synthetic_pqrst_smoke_v1",
        "scientific_claim_eligible": False,
        "warning": "Synthetic ECG-like benchmark for engineering verification and workbench inspection only. Not clinical data.",
        "manifest_path": str(manifest_path),
        "signals_dir": str(signals_dir),
        "n_records": int(len(manifest_df)),
        "split_counts": {s: int((manifest_df["split"] == s).sum()) for s in ("train", "val", "test")},
        "sampling_rate": int(sampling_rate),
        "signal_length": int(signal_length),
        "num_leads": 12,
        "canonical_lead_order": STANDARD_12_LEAD_ORDER,
        "normalize_per_lead": bool(normalize),
        "amplitude_unit": "normalized_sd" if normalize else "mV",
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "patient_overlap_report": overlap,
    }
    atomic_write_json(root / "synthetic_cohort_summary.json", summary)
    return summary


def _build_signal_batch(manifest_path: str | Path, split: str, limit: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = CachedECGDataset(manifest_path, split=split, limit=limit)
    signals, labels, ids = [], [], []
    for idx in range(len(dataset)):
        sig, lab, meta = dataset[idx]
        signals.append(sig.numpy())
        labels.append(lab.numpy())
        ids.append(int(meta["ecg_id"]))
    return np.stack(signals, axis=0), np.stack(labels, axis=0), np.asarray(ids, dtype=np.int64)


def _extract_patient_clusters(manifest_path: str | Path, ecg_ids: np.ndarray) -> np.ndarray | None:
    frame = read_table(manifest_path)
    if "patient_id" not in frame.columns or frame["patient_id"].isna().all():
        return None
    lookup = frame.drop_duplicates(subset=["ecg_id"]).set_index("ecg_id")["patient_id"]
    mapped = pd.Series(ecg_ids).map(lookup)
    return mapped.to_numpy() if mapped.notna().all() else None


def run_model_experiment(
    *,
    config: dict[str, Any],
    model_name: str,
    manifest_path: str | Path,
    execution_tier: str = "synthetic_smoke",
    suites: tuple[str, ...] = ALL_SUITES,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    run_id: str | None = None,
    resume_run_id: str | None = None,
    dataset_provenance: dict[str, Any] | None = None,
    n_bootstrap: int = 100,
    robustness_limit: int = 32,
    xai_samples: int = 4,
) -> dict[str, Any]:
    """Run an end-to-end, provenance-tracked research experiment for a single model."""
    run_config = copy.deepcopy(config)
    run_config["model"]["name"] = model_name
    run_config["data"]["cache_manifest_path"] = str(manifest_path)
    seed_everything(int(run_config["project"]["seed"]))

    manifest_df = read_table(manifest_path)
    overlap = compute_patient_overlap_report(manifest_df)
    if overlap.get("has_patient_leakage"):
        raise ValueError(f"Refusing to run experiment on manifest with patient leakage: {overlap}")

    ds_prov = {
        "manifest_path": str(manifest_path),
        "n_records": int(len(manifest_df)),
        "split_counts": {s: int((manifest_df["split"] == s).sum()) for s in ("train", "val", "test") if "split" in manifest_df.columns},
        "patient_overlap_report": overlap,
        "preprocessing_fingerprint": compute_preprocessing_fingerprint(run_config),
        **(dataset_provenance or {}),
    }

    if resume_run_id is not None:
        run_manifest, run_dir = prepare_resumable_run(resume_run_id, runs_root=runs_root)
    else:
        run_manifest, run_dir = create_run(
            config=run_config,
            model_name=model_name,
            execution_tier=execution_tier,
            suite_name="_".join(suites[:2]) if len(suites) <= 2 else "full_reliability",
            dataset_provenance=ds_prov,
            model_identity={
                "model_name": model_name,
                "hyperparameters": copy.deepcopy(run_config["xgboost"] if model_name == "xgboost" else run_config["model"]),
            },
            run_id=run_id,
            runs_root=runs_root,
        )

    current_stage = "init"
    try:
        ckpt_dir = run_dir / "checkpoints"
        pred_raw_dir = run_dir / "predictions" / "raw"
        pred_cal_dir = run_dir / "predictions" / "calibrated"
        cal_dir = run_dir / "calibration"
        dec_dir = run_dir / "decision"
        rob_dir = run_dir / "robustness"
        xai_dir = run_dir / "explanations"
        analysis_dir = run_dir / "analysis"

        # 1. Training stage (recoverable via last.pt for torch models)
        current_stage = "training"
        stage_done = run_manifest.get("stages", {}).get("training", {}).get("status") == "completed"
        if not stage_done:
            if model_name == "xgboost":
                model_path = ckpt_dir / "model.pkl"
                train_res = fit_xgboost_model(run_config, manifest_path, model_path)
                run_manifest["checkpoints"]["model"] = {
                    "path": str(model_path),
                    "sha256": file_sha256(model_path),
                }
                run_manifest["artifacts"]["training_runtime_report"] = str(ckpt_dir / "training_runtime_report.json")
                run_manifest["model_identity"]["n_estimators"] = int(run_config["xgboost"]["n_estimators"])
                mark_stage_status(run_manifest, "training", "completed", details={"runtime_report": train_res["runtime_report"]}, runs_root=runs_root)
            else:
                resume_ckpt = ckpt_dir / "last.pt" if (resume_run_id is not None and (ckpt_dir / "last.pt").exists()) else None
                limit = run_config["data"].get("debug_max_records")
                train_res = fit_torch_model(
                    config=run_config,
                    manifest_path=manifest_path,
                    output_dir=ckpt_dir,
                    limit_train=limit,
                    limit_val=min(limit, 128) if isinstance(limit, int) else None,
                    resume_from=resume_ckpt,
                )
                run_manifest["checkpoints"]["best"] = {
                    "path": str(ckpt_dir / "best.pt"),
                    "sha256": file_sha256(ckpt_dir / "best.pt"),
                }
                run_manifest["checkpoints"]["last"] = {
                    "path": str(ckpt_dir / "last.pt"),
                    "sha256": file_sha256(ckpt_dir / "last.pt"),
                }
                run_manifest["model_identity"]["total_parameters"] = int(train_res["total_parameters"])
                run_manifest["artifacts"]["training_runtime_report"] = str(train_res["runtime_report_path"])
                run_manifest["metrics"]["training_history"] = train_res["history"]
                mark_stage_status(
                    run_manifest,
                    "training",
                    "completed",
                    details={"best_epoch": train_res["best_epoch"], "epochs_ran": len(train_res["history"])},
                    runs_root=runs_root,
                )

        # 2. Discrimination evaluation (val + test)
        current_stage = "discrimination"
        device = resolve_device(run_config["training"]["device"])
        configure_torch_runtime(run_config)
        torch_model = None
        xgb_model = None
        if model_name == "xgboost":
            xgb_model = MultiLabelXGBoost.load(ckpt_dir / "model.pkl")
        else:
            torch_model, _ = load_torch_model(ckpt_dir / "best.pt", device=device)

        split_outputs: dict[str, dict[str, np.ndarray]] = {}
        for split in ("val", "test"):
            if model_name == "xgboost":
                outputs = predict_xgboost_model(ckpt_dir / "model.pkl", manifest_path, split=split)
            else:
                limit = run_config["data"].get("debug_max_records")
                loader = build_dataloader(
                    manifest_path=manifest_path,
                    split=split,
                    config=run_config,
                    limit=min(limit, 128) if isinstance(limit, int) else None,
                    shuffle=False,
                )
                outputs = predict_torch_model(torch_model, loader, device=device, config=run_config)
            split_outputs[split] = outputs
            metrics = compute_classification_metrics(outputs["labels"], outputs["probabilities"])
            cal_raw = compute_calibration_report(outputs["labels"], outputs["probabilities"], num_bins=int(run_config["calibration"]["num_bins"]))
            save_prediction_bundle(pred_raw_dir, split, outputs["ecg_ids"], outputs["logits"], outputs["probabilities"], outputs["labels"])
            save_json(pred_raw_dir / f"{split}_metrics_raw.json", metrics)
            save_metrics_csv(pred_raw_dir / f"{split}_metrics_raw.csv", metrics)
            run_manifest["metrics"][f"raw_{split}"] = {**metrics, **cal_raw}
        run_manifest["artifacts"]["raw_test_metrics"] = str(pred_raw_dir / "test_metrics_raw.json")
        mark_stage_status(run_manifest, "discrimination", "completed", runs_root=runs_root)

        val_out = split_outputs["val"]
        test_out = split_outputs["test"]

        # 3. Validation-only Calibration
        best_cal_method = "raw"
        best_calibrator = None
        test_cal_probs = test_out["probabilities"]
        val_cal_probs = val_out["probabilities"]
        if "calibration" in suites:
            current_stage = "calibration"
            methods = list(run_config["calibration"]["methods"])
            num_bins = int(run_config["calibration"]["num_bins"])
            cal_results = evaluate_calibration_methods(methods, val_out["logits"], val_out["labels"], num_bins=num_bins)
            best_cal_method = select_best_method(cal_results)
            best_calibrator = cal_results[best_cal_method]["calibrator"]
            val_reports = {m: payload["report"] for m, payload in cal_results.items()}
            val_reports["best_method"] = best_cal_method
            save_json(cal_dir / "reports.json", val_reports)

            test_reports: dict[str, Any] = {
                "raw": compute_calibration_report(test_out["labels"], test_out["probabilities"], num_bins=num_bins),
                "selected_method_from_val": best_cal_method,
            }
            for method, payload in cal_results.items():
                save_calibrator(cal_dir / f"{method}.pkl", method, payload["calibrator"], payload["report"])
                for split_name, bundle in split_outputs.items():
                    cprobs = apply_calibrator(payload["calibrator"], bundle["logits"])
                    if split_name == "test":
                        test_reports[method] = compute_calibration_report(bundle["labels"], cprobs, num_bins=num_bins)
                    if method == best_cal_method:
                        if split_name == "val":
                            val_cal_probs = cprobs
                        else:
                            test_cal_probs = cprobs
                    save_prediction_bundle(pred_cal_dir / method, split_name, bundle["ecg_ids"], bundle["logits"], cprobs, bundle["labels"])

            # Precompute reliability diagram bins on held-out test split (raw vs best calibrated)
            rel_payload: dict[str, Any] = {"split": "test", "best_method": best_cal_method, "num_bins": num_bins, "per_class": {}}
            for cls_idx, label in enumerate(TARGET_LABELS):
                rel_payload["per_class"][label] = {
                    "raw": reliability_bins(test_out["labels"], test_out["probabilities"], class_index=cls_idx, num_bins=num_bins),
                    "calibrated": reliability_bins(test_out["labels"], test_cal_probs, class_index=cls_idx, num_bins=num_bins),
                }
            save_json(cal_dir / "test_reports.json", test_reports)
            save_json(cal_dir / "reliability_bins.json", rel_payload)
            save_json(
                cal_dir / "leakage_report.json",
                {
                    "calibration_fit_split": "val",
                    "calibration_selection_split": "val",
                    "test_used_for_fitting": False,
                    "val_n": int(len(val_out["labels"])),
                    "test_n": int(len(test_out["labels"])),
                    "selected_method": best_cal_method,
                },
            )
            run_manifest["metrics"]["calibration_val"] = val_reports
            run_manifest["metrics"]["calibrated_test"] = test_reports[best_cal_method]
            run_manifest["metrics"]["selected_calibration_method"] = best_cal_method
            run_manifest["artifacts"]["calibration_reports"] = str(cal_dir / "reports.json")
            run_manifest["artifacts"]["calibration_test_reports"] = str(cal_dir / "test_reports.json")
            run_manifest["artifacts"]["reliability_bins"] = str(cal_dir / "reliability_bins.json")
            mark_stage_status(run_manifest, "calibration", "completed", details={"best_method": best_cal_method}, runs_root=runs_root)

        # 4. Validation-only Selective Prediction
        thresholds_dict: dict[str, float] | None = None
        fn_weights, fp_weights = weight_arrays(run_config["decision"])
        if "selective" in suites:
            current_stage = "selective"
            thresholds_payload = fit_coverage_thresholds(
                val_cal_probs,
                val_out["labels"],
                fn_weights=fn_weights,
                fp_weights=fp_weights,
                coverage_targets=list(run_config["decision"]["coverage_targets"]),
            )
            thresholds_dict = {str(k): float(v) for k, v in thresholds_payload["thresholds"].items()}
            selective_results = evaluate_selective_policy(
                test_cal_probs,
                test_out["labels"],
                fn_weights=fn_weights,
                fp_weights=fp_weights,
                thresholds=thresholds_dict,
                abstain_cost=float(run_config["decision"]["abstain_cost"]),
            )
            save_json(dec_dir / "thresholds.json", thresholds_payload)
            save_json(dec_dir / "results.json", selective_results)
            pd.DataFrame(selective_results["curves"]).to_csv(dec_dir / "results.csv", index=False)
            run_manifest["metrics"]["selective_test"] = selective_results
            run_manifest["artifacts"]["selective_results"] = str(dec_dir / "results.json")
            mark_stage_status(run_manifest, "selective", "completed", runs_root=runs_root)

        # 5. Robustness Stress-Test Suite
        if "robustness" in suites:
            current_stage = "robustness"
            signals, labels, _ = _build_signal_batch(manifest_path, "test", limit=robustness_limit)

            def predict_logits_fn(batch: np.ndarray) -> np.ndarray:
                if model_name == "xgboost":
                    assert xgb_model is not None
                    feats = np.stack([compute_engineered_features(sig) for sig in batch], axis=0)
                    return xgb_model.predict_logits(feats)
                assert torch_model is not None
                with torch.no_grad():
                    return torch_model(torch.tensor(batch, dtype=torch.float32, device=device)).cpu().numpy()

            def predict_raw_fn(batch: np.ndarray) -> np.ndarray:
                return sigmoid(predict_logits_fn(batch))

            def predict_cal_fn(batch: np.ndarray) -> np.ndarray:
                return apply_calibrator(best_calibrator, predict_logits_fn(batch))

            perturbation_specs = build_severity_perturbation_suite(
                sampling_rate=int(run_config["data"]["sampling_rate"]),
                seed=int(run_config["project"]["seed"]),
            )
            rob_results = evaluate_under_perturbations(signals, labels, predict_raw_fn, perturbation_specs)
            rob_results["perturbation_specs"] = perturbation_specs
            rob_results["nstdb_reference"] = MIT_BIH_NSTDB_REFERENCE
            rob_results["scientific_scope"] = "synthetic_stress_test_not_clinical_claim"

            rob_dir.mkdir(parents=True, exist_ok=True)
            if thresholds_dict is not None:
                dec_rob = evaluate_decision_layer_under_perturbations(
                    signals,
                    labels,
                    predict_cal_fn,
                    perturbation_specs,
                    fn_weights=fn_weights,
                    fp_weights=fp_weights,
                    thresholds=thresholds_dict,
                    abstain_cost=float(run_config["decision"]["abstain_cost"]),
                )
                rob_results["decision_layer_available"] = True
                rob_results["decision_layer"] = dec_rob
                pd.DataFrame(dec_rob.get("decision_delta_summary", [])).to_csv(rob_dir / "decision_delta_summary.csv", index=False)

            save_json(rob_dir / "results.json", rob_results)
            pd.DataFrame(rob_results.get("delta_summary", [])).to_csv(rob_dir / "delta_summary.csv", index=False)
            run_manifest["metrics"]["robustness_delta_summary"] = rob_results["delta_summary"]
            run_manifest["artifacts"]["robustness_results"] = str(rob_dir / "results.json")
            mark_stage_status(run_manifest, "robustness", "completed", runs_root=runs_root)

        # 6. Explainability + Attribution Sanity Checks (PyTorch models)
        if "explainability" in suites and model_name != "xgboost":
            current_stage = "explainability"
            assert torch_model is not None
            sample_manifest, target_manifest, sel_summary = build_xai_selection_manifests(
                ecg_ids=test_out["ecg_ids"],
                probabilities=test_cal_probs,
                num_samples=int(xai_samples),
                thresholds=thresholds_dict,
                fn_weights=fn_weights if thresholds_dict is not None else None,
                fp_weights=fp_weights if thresholds_dict is not None else None,
            )
            xai_dir.mkdir(parents=True, exist_ok=True)
            sample_manifest.to_csv(xai_dir / "xai_sample_manifest.csv", index=False)
            target_manifest.to_csv(xai_dir / "xai_target_manifest.csv", index=False)

            cache_lookup = (
                manifest_df.loc[manifest_df["split"] == "test", ["ecg_id", "cache_path"]]
                .drop_duplicates(subset=["ecg_id"])
                .set_index("ecg_id")
            )
            stability_rows: list[dict[str, Any]] = []
            sanity_reports: list[dict[str, Any]] = []
            for t_idx, row in target_manifest.head(max(int(xai_samples), 1)).iterrows():
                eid = int(row["ecg_id"])
                t_label = str(row["target_label"])
                t_cls = TARGET_LABELS.index(t_label)
                sig_np = np.load(Path(str(cache_lookup.loc[eid, "cache_path"]))).astype(np.float32)
                sig_t = torch.tensor(sig_np[None, ...], dtype=torch.float32, device=device)
                ig_t = compute_integrated_gradients(torch_model, sig_t, target_class=t_cls, steps=16)
                gx_t = compute_gradient_x_input(torch_model, sig_t, target_class=t_cls)
                occ_np = None
                if t_idx == 0:
                    occ_t = compute_occlusion(torch_model, sig_t, target_class=t_cls, sliding_window_shapes=(1, 100), strides=(1, 50))
                    occ_np = occ_t.cpu().numpy()[0]

                ig_np = ig_t.cpu().numpy()[0]
                gx_np = gx_t.cpu().numpy()[0]
                noisy = gaussian_noise(sig_np, std=0.05, seed=int(run_config["project"]["seed"]))
                dropped = lead_dropout_single(sig_np, lead_index=0)
                ig_noisy = compute_integrated_gradients(
                    torch_model, torch.tensor(noisy[None, ...], dtype=torch.float32, device=device), target_class=t_cls, steps=12
                ).cpu().numpy()[0]
                ig_dropped = compute_integrated_gradients(
                    torch_model, torch.tensor(dropped[None, ...], dtype=torch.float32, device=device), target_class=t_cls, steps=12
                ).cpu().numpy()[0]

                stem = f"ecg_{eid}_{t_label.lower()}"
                atomic_save_npy(xai_dir / f"{stem}_ig.npy", ig_np.astype(np.float32))
                atomic_save_npy(xai_dir / f"{stem}_gx.npy", gx_np.astype(np.float32))
                if occ_np is not None:
                    atomic_save_npy(xai_dir / f"{stem}_occlusion.npy", occ_np.astype(np.float32))

                if t_idx < 2:
                    sanity = run_attribution_sanity_suite(
                        torch_model,
                        sig_t,
                        ig_t,
                        target_class=t_cls,
                        attr_fn=lambda m, x, c: compute_integrated_gradients(m, x, target_class=c, steps=12),
                        method_name="integrated_gradients",
                        steps=6,
                        seed=int(run_config["project"]["seed"]),
                    )
                    sanity["ecg_id"] = eid
                    sanity["target_label"] = t_label
                    sanity_reports.append(sanity)

                stability_rows.append(
                    {
                        "ecg_id": eid,
                        "target_label": t_label,
                        "case_type": str(row["case_type"]),
                        "ig_gx_cosine": cosine_similarity(ig_np, gx_np),
                        "ig_noise_cosine": cosine_similarity(ig_np, ig_noisy),
                        "ig_dropout_cosine": cosine_similarity(ig_np, ig_dropped),
                        "ig_gx_spearman": spearman_similarity(ig_np, gx_np),
                        "ig_noise_topk_overlap": topk_overlap(ig_np, ig_noisy, k=200),
                        "lead_entropy": lead_attribution_entropy(ig_np),
                        "occlusion_computed": occ_np is not None,
                    }
                )

            pd.DataFrame(stability_rows).to_csv(xai_dir / "stability.csv", index=False)
            save_json(xai_dir / "sanity_checks.json", {"selection_summary": sel_summary, "checks": sanity_reports})
            run_manifest["metrics"]["explainability_summary"] = (
                pd.DataFrame(stability_rows).mean(numeric_only=True).to_dict() if stability_rows else {}
            )
            run_manifest["artifacts"]["xai_stability"] = str(xai_dir / "stability.csv")
            run_manifest["artifacts"]["xai_sanity_checks"] = str(xai_dir / "sanity_checks.json")
            mark_stage_status(run_manifest, "explainability", "completed", runs_root=runs_root)

        # 7. Patient-Cluster Bootstrap CIs
        if "bootstrap" in suites:
            current_stage = "bootstrap"
            cluster_ids = _extract_patient_clusters(manifest_path, test_out["ecg_ids"])
            boot_payload: dict[str, Any] = {}
            for m_name, m_fn in [
                ("macro_auroc", lambda yt, ys: float(compute_classification_metrics(yt, ys)["macro_auroc"])),
                ("macro_auprc", lambda yt, ys: float(compute_classification_metrics(yt, ys)["macro_auprc"])),
                ("macro_ece", lambda yt, ys: float(compute_calibration_report(yt, ys)["macro_ece"])),
                ("macro_brier", lambda yt, ys: float(compute_calibration_report(yt, ys)["macro_brier"])),
            ]:
                boot_payload[m_name] = bootstrap_metric(
                    test_out["labels"],
                    test_cal_probs,
                    m_fn,
                    n_bootstrap=int(n_bootstrap),
                    seed=int(run_config["project"]["seed"]),
                    cluster_ids=cluster_ids,
                    ci_level=0.95,
                )
            save_json(analysis_dir / "bootstrap_ci.json", boot_payload)
            run_manifest["metrics"]["bootstrap_ci"] = boot_payload
            run_manifest["artifacts"]["bootstrap_ci"] = str(analysis_dir / "bootstrap_ci.json")
            mark_stage_status(run_manifest, "bootstrap", "completed", runs_root=runs_root)

        # 8. Failure Analysis
        if "failure_analysis" in suites:
            current_stage = "failure_analysis"
            sel_thresh = None
            if thresholds_dict:
                pref_key = min(thresholds_dict.keys(), key=lambda k: abs(float(k) - 0.95))
                sel_thresh = thresholds_dict[pref_key]
            failures = analyze_model_failures(
                test_out["ecg_ids"],
                test_out["labels"],
                test_cal_probs,
                fn_weights=fn_weights,
                fp_weights=fp_weights,
                selective_threshold=sel_thresh,
                top_k=20,
            )
            save_json(analysis_dir / "failure_analysis.json", failures)
            run_manifest["artifacts"]["failure_analysis"] = str(analysis_dir / "failure_analysis.json")
            mark_stage_status(run_manifest, "failure_analysis", "completed", runs_root=runs_root)

        mark_run_completed(run_manifest, runs_root=runs_root)
        return run_manifest
    except BaseException as exc:
        mark_run_failed(run_manifest, stage_name=current_stage, exc=exc, runs_root=runs_root)
        raise


def run_benchmark_comparison(
    *,
    base_config: dict[str, Any],
    models: list[str],
    manifest_path: str | Path,
    execution_tier: str = "synthetic_smoke",
    suites: tuple[str, ...] = ALL_SUITES,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    dataset_provenance: dict[str, Any] | None = None,
    n_bootstrap: int = 100,
    robustness_limit: int = 32,
    xai_samples: int = 4,
) -> dict[str, Any]:
    """Run multiple models on an identical split/preprocessing protocol and compute paired comparisons."""
    completed_runs: dict[str, dict[str, Any]] = {}
    failed_runs: dict[str, str] = {}
    pred_bundles: dict[str, dict[str, np.ndarray]] = {}

    for model_name in models:
        try:
            manifest = run_model_experiment(
                config=base_config,
                model_name=model_name,
                manifest_path=manifest_path,
                execution_tier=execution_tier,
                suites=suites,
                runs_root=runs_root,
                dataset_provenance=dataset_provenance,
                n_bootstrap=n_bootstrap,
                robustness_limit=robustness_limit,
                xai_samples=xai_samples,
            )
            completed_runs[model_name] = manifest
            run_dir = Path(runs_root) / manifest["run_id"]
            best_cal = manifest.get("metrics", {}).get("selected_calibration_method")
            cal_test = run_dir / "predictions" / "calibrated" / str(best_cal) / "test" if best_cal else None
            src_dir = cal_test if (cal_test is not None and cal_test.exists()) else (run_dir / "predictions" / "raw" / "test")
            pred_bundles[model_name] = {
                "ecg_ids": np.load(src_dir / "ecg_ids.npy"),
                "probabilities": np.load(src_dir / "probabilities.npy"),
                "labels": np.load(src_dir / "labels.npy"),
            }
        except Exception as exc:
            failed_runs[model_name] = f"{exc.__class__.__name__}: {exc}"

    comparisons_dir = Path(runs_root) / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    disagreement = analyze_model_disagreement(pred_bundles, top_k=25)
    save_json(comparisons_dir / "model_disagreement.json", disagreement)

    paired: list[dict[str, Any]] = []
    for m_a, m_b in combinations(sorted(pred_bundles.keys()), 2):
        ba, bb = pred_bundles[m_a], pred_bundles[m_b]
        if not np.array_equal(ba["ecg_ids"], bb["ecg_ids"]):
            continue
        cluster_ids = _extract_patient_clusters(manifest_path, ba["ecg_ids"])
        for metric_name, fn in [
            ("macro_auroc", lambda yt, ys: float(compute_classification_metrics(yt, ys)["macro_auroc"])),
            ("macro_auprc", lambda yt, ys: float(compute_classification_metrics(yt, ys)["macro_auprc"])),
            ("macro_brier", lambda yt, ys: float(compute_calibration_report(yt, ys)["macro_brier"])),
        ]:
            comp = paired_bootstrap_comparison(
                ba["labels"],
                ba["probabilities"],
                bb["probabilities"],
                fn,
                n_bootstrap=n_bootstrap,
                seed=int(base_config["project"]["seed"]),
                cluster_ids=cluster_ids,
            )
            paired.append({"model_a": m_a, "model_b": m_b, "metric": metric_name, **comp})

    summary = {
        "execution_tier": execution_tier,
        "models_requested": models,
        "completed_run_ids": {m: r["run_id"] for m, r in completed_runs.items()},
        "failed_models": failed_runs,
        "paired_comparisons": paired,
        "disagreement_summary": disagreement.get("pairwise_summary", []),
    }
    save_json(comparisons_dir / "paired_comparisons.json", summary)
    for run_manifest in completed_runs.values():
        run_manifest["artifacts"]["paired_comparisons"] = str(comparisons_dir / "paired_comparisons.json")
        run_manifest["artifacts"]["model_disagreement"] = str(comparisons_dir / "model_disagreement.json")
        save_run_manifest(run_manifest, runs_root=runs_root)
    return summary
