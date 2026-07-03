#!/usr/bin/env bash
set -euo pipefail

ROWS=(50000 500000 2000000)
for n in "${ROWS[@]}"; do
  echo "--- Tier rows=${n} ---"
  python3 run_all.py \
    --rows "$n" \
    --facilities 5 \
    --backend auto \
    --data "data/er_stress_${n}.parquet" \
    --run-baselines
done
