from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run provider scheduling optimization workflow")
    parser.add_argument("--in", dest="input_path", default="data/er_stress_50000.parquet")
    parser.add_argument("--backend", choices=["auto", "jax", "pytorch", "skip"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--run-baselines", action="store_true", default=True)
    parser.add_argument("--no-tabfm", action="store_true", default=False)
    parser.add_argument("--shift-hours", type=int, default=4)
    parser.add_argument("--doc-capacity-per-hour", type=float, default=4.0)
    parser.add_argument("--np-capacity-per-hour", type=float, default=3.0)
    parser.add_argument("--results-root", default="outputs/scheduling_run")
    return parser.parse_args()


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, text=True)


def _best_forecast_csv(reg_dir: Path, use_tabfm: bool) -> Path:
    tabfm_pred = reg_dir / "predictions_tabfm.csv"
    baseline_pred = reg_dir / "predictions_baseline.csv"
    if use_tabfm and tabfm_pred.exists():
        return tabfm_pred
    if baseline_pred.exists():
        return baseline_pred
    if tabfm_pred.exists():
        return tabfm_pred
    raise FileNotFoundError(f"no prediction file in {reg_dir}")


def main() -> None:
    args = parse_args()
    python = sys.executable

    reg_dir = Path(args.results_root) / "regression"
    cls_dir = Path(args.results_root) / "classification"
    reg_dir.mkdir(parents=True, exist_ok=True)
    cls_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--in",
        args.input_path,
        "--seed",
        str(args.seed),
    ]
    if args.rows:
        common.extend(["--rows", str(args.rows)])
    if args.run_baselines:
        common.append("--run-baseline")
    if args.no_tabfm:
        common.append("--no-tabfm")

    regression_cmd = [
        python,
        "exp_er_demand_forecast.py",
        *common,
        "--backend",
        args.backend,
        "--results-dir",
        str(reg_dir),
    ]

    classification_cmd = [
        python,
        "exp_shortage_risk.py",
        *common,
        "--backend",
        args.backend,
        "--results-dir",
        str(cls_dir),
    ]

    _run(regression_cmd)
    _run(classification_cmd)

    use_tabfm = not args.no_tabfm
    forecast_csv = _best_forecast_csv(reg_dir, use_tabfm=use_tabfm)

    out_summary = Path(args.results_root) / "schedule_summary.json"
    schedule_cmd = [
        python,
        "simulate_shift_plan.py",
        "--forecast-csv",
        str(forecast_csv),
        "--actual-csv",
        args.input_path,
        "--out",
        str(out_summary),
        "--shift-hours",
        str(args.shift_hours),
        "--doc-capacity-per-hour",
        str(args.doc_capacity_per_hour),
        "--np-capacity-per-hour",
        str(args.np_capacity_per_hour),
    ]
    _run(schedule_cmd)

    with open(out_summary, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print("SCHEDULING SUMMARY")
    print(json.dumps(metrics[0] if isinstance(metrics, list) else metrics, indent=2))
    print(f"Forecast source: {forecast_csv}")


if __name__ == "__main__":
    main()
