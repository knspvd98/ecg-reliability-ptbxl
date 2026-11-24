#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate 2>/dev/null || true

python scripts/02_train_xgboost.py --config configs/xgboost.yaml
python scripts/03_train_torch_model.py --config configs/resnet1d.yaml
python scripts/03_train_torch_model.py --config configs/inceptiontime.yaml
python scripts/03_train_torch_model.py --config configs/tcn.yaml
python scripts/03_train_torch_model.py --config configs/transformer.yaml
