from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd


def load_table(path: Path) -> tuple[str, pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return "parquet", df
    if suffix in {".csv"}:
        df = pd.read_csv(path)
        return "csv", df
    if suffix in {".jsonl"}:
        df = pd.read_json(path, lines=True)
        return "jsonl", df
    if suffix in {".json"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return "json-list", pd.DataFrame(payload)
        if isinstance(payload, dict):
            return "json-dict", pd.DataFrame([payload])
        return "json-raw", pd.DataFrame()
    raise ValueError(f"Unsupported file type: {path}")


def print_top_for_paths(paths: list[str] | tuple[str, ...], top_n: int = 5) -> None:
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[missing] {path}")
            continue
        source, df = load_table(path)
        shape = df.shape
        cols = list(df.columns)
        print(f"=== {path} ({source}) ===")
        print(f"shape={shape}")
        print(f"columns={cols}")
        if not df.empty:
            print(df.head(top_n).to_string(index=False))
        else:
            print("empty")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print top rows and metadata for parquet/csv/json artifacts")
    parser.add_argument("paths", nargs="+", help="artifact file paths")
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_top_for_paths(args.paths, top_n=max(1, int(args.top_n)))


if __name__ == "__main__":
    main()
