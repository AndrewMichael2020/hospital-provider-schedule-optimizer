from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42
    n_rows: int = 50_000
    n_facilities: int = 3
    start_datetime: str = "2024-01-01 00:00:00"
    step_hours: int = 1
    output_path: str | None = None
    facility_scale: float = 1.0
    provider_pool_per_facility: int = 220
    room_pool_per_facility: int = 120
    drift_start_month: int = 7
    surge_rate: float = 0.025
    surge_magnitude: float = 2.8


@dataclass(frozen=True)
class SplitConfig:
    timestamp_col: str = "timestamp"
    test_size: float = 0.2
    val_size: float = 0.2
    shuffle: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    random_state: int = 42
    backend: Literal["jax", "pytorch", "auto"] = "auto"
    max_rows: int | None = None
    cat_features: list[str] | None = None
    datetime_features: list[str] | None = None
    n_jobs: int = 4
    timeout_seconds: int | None = None
    results_dir: str = "outputs"

