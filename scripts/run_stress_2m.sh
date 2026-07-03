#!/usr/bin/env bash
set -euo pipefail

python3 run_all.py \
  --rows 2000000 \
  --facilities 10 \
  --backend auto \
  --run-baselines \
  --data data/er_stress_2m.parquet
