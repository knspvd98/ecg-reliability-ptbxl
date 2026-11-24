from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_reliability.evaluation.reports import save_json
from ecg_reliability.external.cache_external import cache_external_manifest
from ecg_reliability.external.challenge2021 import DEFAULT_SOURCE_ENV_MAP, build_challenge2021_manifest
from ecg_reliability.io import write_table
from ecg_reliability.runtime import load_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    external_cfg = config["external"]
    summary_path = Path("outputs/external/external_cache_summary.json")

    if not bool(external_cfg.get("enabled", False)):
        save_json(summary_path, {"enabled": False, "status": "external_disabled"})
        print("External validation disabled; skipping Challenge 2021 preparation.")
        return

    nested_root = os.environ.get(str(external_cfg.get("root_env", "CHALLENGE2021_ROOT")))
    source_envs = {**DEFAULT_SOURCE_ENV_MAP, **{str(k): str(v) for k, v in external_cfg.get("source_root_envs", {}).items()}}
    source_roots = {
        source: os.environ.get(source_envs[source])
        for source in external_cfg["sources"]
        if source in source_envs and os.environ.get(source_envs[source])
    }
    if not source_roots and not nested_root:
        expected = ", ".join(source_envs.get(source, f"<env for {source}>") for source in external_cfg["sources"])
        raise RuntimeError(
            "External validation enabled, but no direct source roots or nested Challenge root were provided. "
            f"Expected one or more of: {expected}, or {external_cfg.get('root_env', 'CHALLENGE2021_ROOT')}."
        )
    external_cfg["root"] = {
        "direct_source_roots": source_roots,
        "nested_root": nested_root,
    }

    manifest = build_challenge2021_manifest(
        root=nested_root,
        sources=list(external_cfg["sources"]),
        normal_snomed_codes=[str(code) for code in external_cfg["normal_snomed_codes"]],
        allow_sources={str(key): bool(value) for key, value in external_cfg["allow_sources"].items()},
        max_records_per_source=external_cfg.get("max_records_per_source"),
        source_roots=source_roots,
    )
    cached_manifest, cache_summary = cache_external_manifest(manifest, config)

    manifest_path = Path(external_cfg["manifest_path"])
    csv_path = manifest_path.with_suffix(".csv")
    write_table(cached_manifest, manifest_path)
    write_table(cached_manifest, csv_path)
    save_json(summary_path, cache_summary)
    print(str(manifest_path))
    print(str(csv_path))
    print(str(summary_path))


if __name__ == "__main__":
    main()
