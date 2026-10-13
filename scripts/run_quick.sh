#!/usr/bin/env bash
# Fast loop for day-to-day work: one student, 100 probe items, no full grid.
# Use this while you are still changing the measurement code.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
RECIPE="${2:-T1,hard_label,0,qwen05b}"
cd "$(dirname "$0")/.."

python scripts/smoke_test.py
python -m src.train    --config "$CONFIG" --recipe "$RECIPE"
python -m src.evaluate --config "$CONFIG" --limit 100
python -m src.analyze  --config "$CONFIG" --bootstrap 200

echo "done. check table2_inherit.csv: error_lift well above 1.0 is the signal."
