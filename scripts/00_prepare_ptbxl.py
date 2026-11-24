from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_reliability.data.ptbxl_index import (
    build_index_summary,
    label_cooccurrence,
    prepare_ptbxl_index,
    save_prepared_index,
)
from ecg_reliability.evaluation.reports import save_json
from ecg_reliability.paths import snapshot_environment
from ecg_reliability.runtime import load_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    snapshot_environment("outputs/summary/environment.json")
    index_df = prepare_ptbxl_index(config)
    save_prepared_index(index_df, config["data"]["index_path"])
    metadata = index_df.attrs.get("ptbxl_prepare_metadata", {})
    summary = build_index_summary(
        index_df=index_df,
        config=config,
        empty_target_count_before_filter=metadata.get("empty_target_count_before_filter"),
        overlap_report=metadata.get("overlap_report", {}),
        metadata=metadata,
    )
    save_json("outputs/index/index_summary.json", summary)
    label_cooccurrence(index_df).to_csv("outputs/index/label_cooccurrence.csv")
    print(f"Prepared PTB-XL index with {summary['total_records']} records: {summary['split_counts']}")
    print("Index summary: outputs/index/index_summary.json")


if __name__ == "__main__":
    main()
