#!/usr/bin/env bash
set -euo pipefail

python3 run_all.py \
  --rows 50000 \
  --facilities 3 \
  --backend auto \
  --run-baselines \
  --data data/er_stress_smoke.parquet
