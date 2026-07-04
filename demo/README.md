# Hospital Schedule Optimizer Demo

This folder is a static GitHub Pages scheduling-console demo built from the repository's synthetic fixed-rotation overflow workflow.

The demo is designed for scheduling and operations users. It shows:

- the protected initial schedule;
- scenario controls for budget, demand pressure, minimum covered gap, and shift length;
- a regenerate button that updates the business-case metrics;
- the optimized overflow schedule;
- compact impact, budget, and remaining-gap views.

## Build the data

From the repository root:

```bash
python3 run_fixed_rotation_overflow_workflow.py \
  --seed 42 \
  --facilities 4 \
  --days 30 \
  --overflow-shift-hours 10 \
  --overflow-budget-share-md 0.10 \
  --overflow-budget-share-np 0.10 \
  --overflow-budget-share-rn 0.10 \
  --overflow-budget-share-hcw 0.10 \
  --demand-pressure-multiplier 1.00 \
  --min-overflow-covered-gap-hours 4.0 \
  --out outputs/demo_pressure_normal \
  --print-top-n 0

python3 run_fixed_rotation_overflow_workflow.py \
  --seed 42 \
  --facilities 4 \
  --days 30 \
  --overflow-shift-hours 10 \
  --overflow-budget-share-md 0.10 \
  --overflow-budget-share-np 0.10 \
  --overflow-budget-share-rn 0.10 \
  --overflow-budget-share-hcw 0.10 \
  --demand-pressure-multiplier 1.25 \
  --min-overflow-covered-gap-hours 4.0 \
  --out outputs/demo_pressure_moderate \
  --print-top-n 0

python3 run_fixed_rotation_overflow_workflow.py \
  --seed 42 \
  --facilities 4 \
  --days 30 \
  --overflow-shift-hours 10 \
  --overflow-budget-share-md 0.10 \
  --overflow-budget-share-np 0.10 \
  --overflow-budget-share-rn 0.10 \
  --overflow-budget-share-hcw 0.10 \
  --demand-pressure-multiplier 1.35 \
  --min-overflow-covered-gap-hours 4.0 \
  --out outputs/demo_pressure_surge \
  --print-top-n 0

python3 run_fixed_rotation_overflow_workflow.py \
  --seed 42 \
  --facilities 4 \
  --days 30 \
  --overflow-shift-hours 10 \
  --overflow-budget-share-md 0.10 \
  --overflow-budget-share-np 0.10 \
  --overflow-budget-share-rn 0.10 \
  --overflow-budget-share-hcw 0.10 \
  --demand-pressure-multiplier 1.60 \
  --min-overflow-covered-gap-hours 4.0 \
  --out outputs/demo_pressure_severe \
  --print-top-n 0

python3 run_fixed_rotation_overflow_workflow.py \
  --seed 42 \
  --facilities 4 \
  --days 30 \
  --overflow-shift-hours 10 \
  --overflow-budget-share-md 0.10 \
  --overflow-budget-share-np 0.10 \
  --overflow-budget-share-rn 0.10 \
  --overflow-budget-share-hcw 0.10 \
  --demand-pressure-multiplier 2.00 \
  --min-overflow-covered-gap-hours 4.0 \
  --out outputs/demo_pressure_extreme \
  --print-top-n 0

python3 demo/build_demo_data.py \
  --results-root outputs/demo_pressure_surge \
  --pressure-scenario normal=1.00=outputs/demo_pressure_normal \
  --pressure-scenario moderate=1.25=outputs/demo_pressure_moderate \
  --pressure-scenario surge=1.35=outputs/demo_pressure_surge \
  --pressure-scenario severe=1.60=outputs/demo_pressure_severe \
  --pressure-scenario extreme=2.00=outputs/demo_pressure_extreme \
  --out demo/data/demo_case.json
```

## Preview locally

Serve the repo root or the `demo` folder over HTTP:

```bash
python3 -m http.server 8000
```

Then open `http://127.0.0.1:8000/demo/`.

## Publish with GitHub Pages

This repository publishes the `demo` folder through `.github/workflows/pages.yml` because GitHub Pages branch-source mode only supports `/` and `/docs`.

The demo is plain HTML, CSS, JavaScript, and JSON. It does not need a frontend build step or backend.
