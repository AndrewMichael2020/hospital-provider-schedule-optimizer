from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tabfm_healthcare_eval.data import evaluate_dataset_contract, facility_time_split
from tabfm_healthcare_eval.utils import safe_json_dump


def run_checks(df: pd.DataFrame) -> dict:
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    contract = evaluate_dataset_contract(df, timestamp_col="timestamp")
    train, val, test = facility_time_split(
        df,
        facility_col="facility_id",
        timestamp_col="timestamp",
        test_size=0.2,
        val_size=0.2,
    )
    contract["split_rows"] = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    contract["split_ratios"] = {
        "train": len(train) / len(df),
        "val": len(val) / len(df),
        "test": len(test) / len(df),
    }

    contract["date_span"] = {
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "n_days": float((df["timestamp"].max() - df["timestamp"].min()).days),
    }
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run healthcare dataset contract checks")
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input_path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    result = run_checks(df)
    print(result)
    if args.output_path:
        safe_json_dump(result, args.output_path)
        print(f"Wrote contract report: {args.output_path}")


if __name__ == "__main__":
    main()

