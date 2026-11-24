from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_reliability.campaign.artifact_checks import selected_model_from_manifest, validate_claim_artifacts
from ecg_reliability.evaluation.reports import save_json
from ecg_reliability.runtime import load_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--selection-manifest", default="outputs/summary/model_selection.json")
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    selected_model = selected_model_from_manifest(args.selection_manifest)
    payload = validate_claim_artifacts(
        selected_model=selected_model,
        require_external=False,
        external_enabled=bool(config.get("external", {}).get("enabled", False)),
    )
    output_path = Path("outputs/summary/claim_artifact_validation.json")
    save_json(output_path, payload)
    print(output_path)


if __name__ == "__main__":
    main()
