#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate 2>/dev/null || true
python scripts/09_build_figures.py --config configs/base.yaml
python scripts/10_build_results_summary.py --config configs/base.yaml
tar -czf outputs/artifacts.tar.gz outputs
