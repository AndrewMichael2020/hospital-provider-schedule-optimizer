from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict
from typing import Callable, Iterable

import numpy as np
import pandas as pd


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_kv_list(values: str | None) -> list[str]:
    if not values:
        return []
    return [v.strip() for v in values.split(",") if v.strip()]


def safe_json_dump(payload: dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv_if_dataframe(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False)


def write_parquet_if_dataframe(df: pd.DataFrame, out_path: str) -> None:
    df.to_parquet(out_path, index=False)


def elapsed_timer() -> Callable[[], float]:
    start = time.perf_counter()

    def _elapsed() -> float:
        return time.perf_counter() - start

    return _elapsed


def to_serializable_dataclass(cfg):
    if hasattr(cfg, "__dict__") or hasattr(cfg, "__dataclass_fields__"):
        return asdict(cfg)
    return str(cfg)


def parse_args_list(name: str, default: Iterable[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--{name}")
    args = vars(parser.parse_args([]))[name]
    if args:
        return parse_kv_list(args)
    return list(default) if default is not None else []
