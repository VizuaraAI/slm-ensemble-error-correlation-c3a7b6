#!/usr/bin/env bash
# Full pipeline: train the recipe grid, trace every member, analyse, plot.
# Expects the virtualenv to be active and `pip install -r requirements.txt` done.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
cd "$(dirname "$0")/.."

echo "==> 0/5 smoke test (no GPU, no downloads)"
python scripts/smoke_test.py

echo "==> 1/5 training plan"
python -m src.train --config "$CONFIG" --dry-run

echo "==> 2/5 distilling the student grid"
python -m src.train --config "$CONFIG"

echo "==> 3/5 collecting behavioural traces"
python -m src.evaluate --config "$CONFIG"

echo "==> 4/5 analysis"
python -m src.analyze --config "$CONFIG"

echo "==> 5/5 figures"
OUT_DIR="$(python -c "from src.config import load_config; print(load_config('$CONFIG').output_dir())")"
python -m src.plots --analysis "$OUT_DIR/analysis" --out figures

echo "done. tables in $OUT_DIR/analysis, figures in figures/"
