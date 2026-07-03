from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run full TabFM healthcare evaluation suite")
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--facilities", type=int, default=3)
    parser.add_argument("--backend", choices=["jax", "pytorch", "auto"], default="auto")
    parser.add_argument("--run-jax", action="store_true", default=False)
    parser.add_argument("--run-pytorch", action="store_true", default=False)
    parser.add_argument("--run-baselines", action="store_true", default=True)
    parser.add_argument("--skip-tabfm", action="store_true", default=False)
    parser.add_argument("--data", default="data/er_stress.parquet")
    return parser.parse_args()


def _run(cmd: list[str], env=None):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, text=True, env=env)


def main():
    args = parse_args()

    data_path = Path(args.data)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    gen_cmd = [
        "python3",
        "generate_er_stress_data.py",
        "--rows",
        str(args.rows),
        "--seed",
        str(args.seed),
        "--facilities",
        str(args.facilities),
        "--out",
        str(data_path),
    ]
    _run(gen_cmd)

    _run(["python3", "run_data_contract_checks.py", "--in", str(data_path)])

    if args.run_jax and args.run_pytorch:
        backends = ["jax", "pytorch"]
    elif args.run_jax:
        backends = ["jax"]
    elif args.run_pytorch:
        backends = ["pytorch"]
    elif args.backend in {"jax", "pytorch"}:
        backends = [args.backend]
    else:
        # backend=auto: run both for the full stress matrix
        backends = ["jax", "pytorch"]

    common = ["--in", str(data_path), "--seed", str(args.seed)]
    if args.run_baselines:
        common.append("--run-baseline")
    if args.skip_tabfm:
        common.append("--no-tabfm")

    if not args.skip_tabfm:
        for backend in dict.fromkeys(backends):
            _run(
                [
                    "python3",
                    "exp_er_demand_forecast.py",
                    *common,
                    "--backend",
                    backend,
                    "--results-dir",
                    f"outputs/regression/{backend}",
                ]
            )
            _run(
                [
                    "python3",
                    "exp_shortage_risk.py",
                    *common,
                    "--backend",
                    backend,
                    "--results-dir",
                    f"outputs/classification/{backend}",
                ]
            )
    elif args.run_baselines:
        _run(
            [
                "python3",
                "exp_er_demand_forecast.py",
                *common,
                "--backend",
                "auto",
                "--results-dir",
                "outputs/regression/baseline",
            ]
        )
        _run(
            [
                "python3",
                "exp_shortage_risk.py",
                *common,
                "--backend",
                "auto",
                "--results-dir",
                "outputs/classification/baseline",
            ]
        )
    else:
        raise ValueError("skip-tabfm and run_baselines both false has no model path")

    print("Completed end-to-end suite.")


if __name__ == "__main__":
    main()
