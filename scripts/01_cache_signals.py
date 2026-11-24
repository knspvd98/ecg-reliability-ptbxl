from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_reliability.data.labels import TARGET_LABELS
from ecg_reliability.data.preprocessing import inspect_wfdb_header, load_ptbxl_signal
from ecg_reliability.evaluation.reports import save_json
from ecg_reliability.io import read_table, write_table
from ecg_reliability.paths import resolve_path
from ecg_reliability.runtime import load_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    index_df = read_table(config["data"]["index_path"])
    ptbxl_root = Path(config["data"]["ptbxl_root"])
    cache_dir = resolve_path(config["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failed_records = []
    example_resolved_paths: list[str] = []
    observed_shapes: list[tuple[int, ...]] = []
    observed_dtypes: list[str] = []
    observed_sampling_rates: list[float] = []
    observed_lead_orders: list[list[str]] = []
    nan_count = 0
    inf_count = 0
    for _, row in tqdm(index_df.iterrows(), total=len(index_df), desc="cache"):
        cache_path = cache_dir / f"{int(row['ecg_id'])}.npy"
        record_path = ptbxl_root / str(row["record_path"])
        if len(example_resolved_paths) < 5:
            example_resolved_paths.append(str(record_path))
        try:
            header_info = inspect_wfdb_header(record_path)
            if args.overwrite or not cache_path.exists():
                signal = load_ptbxl_signal(
                    record_path=record_path,
                    signal_length=int(config["data"]["signal_length"]),
                    normalize=bool(config["data"].get("normalize_per_lead", True)),
                )
                np.save(cache_path, signal.astype("float32"))
            signal = np.load(cache_path)
        except Exception as exc:
            failed_records.append(
                {
                    "ecg_id": int(row["ecg_id"]),
                    "record_path": str(record_path),
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            continue

        observed_shapes.append(tuple(int(dim) for dim in signal.shape))
        observed_dtypes.append(str(signal.dtype))
        if header_info["sampling_rate"] is not None:
            observed_sampling_rates.append(float(header_info["sampling_rate"]))
        if header_info["lead_order"]:
            observed_lead_orders.append([str(name) for name in header_info["lead_order"]])
        nan_count += int(np.isnan(signal).sum())
        inf_count += int(np.isinf(signal).sum())
        rows.append(
            {
                "ecg_id": int(row["ecg_id"]),
                "cache_path": str(cache_path),
                "split": row["split"],
                **{label: int(row[label]) for label in TARGET_LABELS},
            }
        )
    manifest = pd.DataFrame(rows)
    write_table(manifest, config["data"]["cache_manifest_path"])
    first_shape = list(observed_shapes[0]) if observed_shapes else None
    first_dtype = observed_dtypes[0] if observed_dtypes else None
    unique_lead_orders = []
    seen_lead_orders = set()
    for lead_order in observed_lead_orders:
        key = tuple(lead_order)
        if key in seen_lead_orders:
            continue
        seen_lead_orders.add(key)
        unique_lead_orders.append(lead_order)
    unique_sampling_rates = sorted({float(rate) for rate in observed_sampling_rates})
    expected_shape = [int(config["data"]["num_leads"]), int(config["data"]["signal_length"])]
    warning_lead_order_mismatch = len(unique_lead_orders) > 1
    warning_sampling_rate_mismatch = any(abs(rate - float(config["data"]["sampling_rate"])) > 1e-6 for rate in unique_sampling_rates)
    warning_signal_shape_mismatch = any(list(shape) != expected_shape for shape in observed_shapes)
    summary = {
        "status": "real_ptbxl_cache_prepared",
        "n_index_records": int(len(index_df)),
        "n_cached": int(len(manifest)),
        "n_failed": int(len(failed_records)),
        "failed_records": failed_records,
        "example_resolved_record_paths": example_resolved_paths,
        "observed_signal_shape": first_shape,
        "signal_shape_observed": first_shape,
        "expected_signal_shape": expected_shape,
        "dtype": first_dtype,
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "observed_sampling_rates": unique_sampling_rates,
        "observed_lead_orders": unique_lead_orders,
        "warning_lead_order_mismatch": warning_lead_order_mismatch,
        "warning_sampling_rate_not_100": warning_sampling_rate_mismatch,
        "warning_signal_shape_mismatch": warning_signal_shape_mismatch,
        "cache_dir": str(cache_dir),
        "cache_manifest_path": str(config["data"]["cache_manifest_path"]),
        "overwrite": bool(args.overwrite),
    }
    save_json("outputs/cache/cache_summary.json", summary)
    if len(manifest) == 0:
        raise RuntimeError("Signal caching failed for every record. See outputs/cache/cache_summary.json.")
    print(f"Cached {len(manifest)} signals to {cache_dir}")
    print("Cache summary: outputs/cache/cache_summary.json")


if __name__ == "__main__":
    main()
