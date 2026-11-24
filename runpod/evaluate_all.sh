#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate 2>/dev/null || true

for model in xgboost resnet1d inceptiontime tcn transformer; do
  python scripts/04_eval_model.py --config configs/base.yaml --model-name "${model}"
  python scripts/05_fit_calibration.py --config configs/base.yaml --model-name "${model}"
  python scripts/06_eval_selective_policy.py --config configs/base.yaml --model-name "${model}" --calibration-method best
done
