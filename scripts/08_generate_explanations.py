from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from ecg_reliability.data.labels import TARGET_LABELS
from ecg_reliability.decision.harm import weight_arrays
from ecg_reliability.evaluation.model_selection import resolve_calibration_method, resolve_selected_model
from ecg_reliability.evaluation.reports import save_json
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
from ecg_reliability.performance.device import configure_torch_runtime, cuda_memory_snapshot, resolve_device
from ecg_reliability.performance.runtime_profile import StageTimer, write_runtime_report
from ecg_reliability.plotting.plot_xai import plot_ecg_with_attribution
from ecg_reliability.robustness.perturbations import gaussian_noise, lead_dropout_single
from ecg_reliability.runtime import (
    checkpoint_dir,
    explanations_dir,
    load_runtime_config,
    load_torch_model,
    model_name_from_config,
)


def _load_prediction_bundle(model_name: str, method: str | None) -> tuple[np.ndarray, np.ndarray, str | None]:
    raw_dir = Path(f"outputs/predictions/{model_name}/raw/test")
    raw_ecg_ids = np.load(raw_dir / "ecg_ids.npy")
    if method is None:
        probabilities = np.load(raw_dir / "probabilities.npy")
        return raw_ecg_ids, probabilities, "raw_argmax_debug_only"
    calibrated_dir = Path(f"outputs/predictions/{model_name}/calibrated/{method}/test")
    if calibrated_dir.exists():
        probabilities = np.load(calibrated_dir / "probabilities.npy")
        ecg_ids = np.load(calibrated_dir / "ecg_ids.npy")
        return ecg_ids, probabilities, None
    probabilities = np.load(raw_dir / "probabilities.npy")
    return raw_ecg_ids, probabilities, "raw_argmax_debug_only"


def _load_thresholds(model_name: str, method: str | None) -> dict[str, float] | None:
    if method is None:
        return None
    path = Path(f"outputs/decision/{model_name}/{method}/thresholds.json")
    if not path.exists():
        return None
    import json

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
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    configured_model = model_name_from_config(config, args.model_name)
    model_name = resolve_selected_model(
        configured_model if args.model_name is not None else None,
        args.selection_manifest,
        synthetic_smoke=args.synthetic_smoke,
        synthetic_fallback_model=configured_model,
    )
    if model_name == "xgboost":
        raise ValueError("Explanation generation currently supports torch models only.")

    device = resolve_device(config["training"]["device"])
    configure_torch_runtime(config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, _ = load_torch_model(checkpoint_dir(model_name) / "best.pt", device=device)

    method = resolve_calibration_method(model_name, args.calibration_method)
    ecg_ids, probabilities, debug_warning = _load_prediction_bundle(model_name, method)
    thresholds = _load_thresholds(model_name, method)
    fn_weights, fp_weights = weight_arrays(config["decision"])
    sample_manifest, target_manifest, selection_summary = build_xai_selection_manifests(
        ecg_ids=ecg_ids,
        probabilities=probabilities,
        num_samples=int(config["explainability"]["num_samples"]),
        thresholds=thresholds if debug_warning is None else None,
        fn_weights=fn_weights if debug_warning is None else None,
        fp_weights=fp_weights if debug_warning is None else None,
    )
    if debug_warning is not None and debug_warning not in selection_summary["warnings"]:
        selection_summary["warnings"].append(debug_warning)

    manifest_frame = read_table(config["data"]["cache_manifest_path"])
    cache_lookup = (
        manifest_frame.loc[manifest_frame["split"] == "test", ["ecg_id", "cache_path"]]
        .drop_duplicates(subset=["ecg_id"])
        .set_index("ecg_id")
    )

    out_dir = explanations_dir(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_manifest.to_csv(out_dir / "xai_sample_manifest.csv", index=False)
    target_manifest.to_csv(out_dir / "xai_target_manifest.csv", index=False)

    summary_rows = []
    sanity_checks = []
    stage_timings: dict[str, float] = {}
    occlusion_budget = int(config["explainability"].get("occlusion_num_samples", 0))
    sanity_budget = int(config["explainability"].get("sanity_check_samples", 2))
    with StageTimer("xai_generation", stage_timings):
        for target_idx, row in target_manifest.iterrows():
            ecg_id = int(row["ecg_id"])
            target_label = str(row["target_label"])
            target_class = TARGET_LABELS.index(target_label)
            cache_path = Path(cache_lookup.loc[ecg_id, "cache_path"])
            signal_np = np.load(cache_path).astype(np.float32)
            signal = torch.tensor(signal_np[None, ...], dtype=torch.float32, device=device)

            ig = compute_integrated_gradients(
                model,
                signal,
                target_class=target_class,
                steps=int(config["explainability"]["ig_steps"]),
            )
            gx = compute_gradient_x_input(model, signal, target_class=target_class)
            occ_np = None
            if target_idx < occlusion_budget:
                occ = compute_occlusion(
                    model,
                    signal,
                    target_class=target_class,
                    sliding_window_shapes=(1, int(config["explainability"]["occlusion_window"])),
                    strides=(1, int(config["explainability"]["occlusion_stride"])),
                )
                occ_np = occ.cpu().numpy()[0]

            ig_np = ig.cpu().numpy()[0]
            gx_np = gx.cpu().numpy()[0]
            noisy = gaussian_noise(signal_np, std=0.02, seed=int(config["project"]["seed"]))
            dropped = lead_dropout_single(signal_np, lead_index=0)
            noisy_t = torch.tensor(noisy[None, ...], dtype=torch.float32, device=device)
            dropped_t = torch.tensor(dropped[None, ...], dtype=torch.float32, device=device)
            ig_noisy = compute_integrated_gradients(model, noisy_t, target_class=target_class, steps=16).cpu().numpy()[0]
            ig_dropped = compute_integrated_gradients(model, dropped_t, target_class=target_class, steps=16).cpu().numpy()[0]

            stem = f"ecg_{ecg_id}_{target_label.lower()}"
            np.save(out_dir / f"{stem}_ig.npy", ig_np.astype(np.float32))
            np.save(out_dir / f"{stem}_gx.npy", gx_np.astype(np.float32))
            if occ_np is not None:
                np.save(out_dir / f"{stem}_occlusion.npy", occ_np.astype(np.float32))
            if target_idx == 0:
                plot_ecg_with_attribution(signal_np, ig_np, out_dir / "example_xai.png")

            if target_idx < sanity_budget:
                check = run_attribution_sanity_suite(
                    model,
                    signal,
                    ig,
                    target_class=target_class,
                    attr_fn=lambda m, x, c: compute_integrated_gradients(m, x, target_class=c, steps=16),
                    method_name="integrated_gradients",
                    steps=6,
                    seed=int(config["project"]["seed"]),
                )
                check["ecg_id"] = ecg_id
                check["target_label"] = target_label
                sanity_checks.append(check)

            summary_rows.append(
                {
                    "ecg_id": ecg_id,
                    "target_label": target_label,
                    "case_type": str(row["case_type"]),
                    "decision_source": str(row["decision_source"]),
                    "ig_gx_cosine": cosine_similarity(ig_np, gx_np),
                    "ig_noise_cosine": cosine_similarity(ig_np, ig_noisy),
                    "ig_dropout_cosine": cosine_similarity(ig_np, ig_dropped),
                    "ig_gx_spearman": spearman_similarity(ig_np, gx_np),
                    "ig_noise_topk_overlap": topk_overlap(ig_np, ig_noisy, k=200),
                    "lead_entropy": lead_attribution_entropy(ig_np),
                    "occlusion_computed": occ_np is not None,
                }
            )

    pd.DataFrame(summary_rows).to_csv(out_dir / "stability.csv", index=False)
    save_json(out_dir / "sanity_checks.json", {"selection_summary": selection_summary, "checks": sanity_checks})
    runtime_report = {
        "number_of_samples": int(selection_summary["selected_sample_count"]),
        "number_of_targets": int(selection_summary["selected_target_count"]),
        "sanity_check_count": int(len(sanity_checks)),
        "device": str(device),
        "elapsed_seconds": float(stage_timings["xai_generation"]),
        "samples_per_second": float(len(summary_rows) / max(stage_timings["xai_generation"], 1e-9)),
        "peak_cuda_memory": cuda_memory_snapshot()["peak_allocated"],
        "calibration_method": method,
        **selection_summary,
    }
    write_runtime_report(out_dir / "runtime_report.json", runtime_report)
    print(f"Explanation artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
