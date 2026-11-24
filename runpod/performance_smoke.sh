#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate 2>/dev/null || true

python scripts/11_runtime_preflight.py --config configs/debug.yaml --model-name simple_cnn --max-batches 10
python scripts/11_runtime_preflight.py --config configs/resnet1d.yaml --model-name resnet1d --max-batches 10

printf '%s\n' "outputs/summary/runtime_preflight.json"
