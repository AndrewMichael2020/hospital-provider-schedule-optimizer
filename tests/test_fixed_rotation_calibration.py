from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_fixed_rotation_overflow_workflow import build_ptd_scenarios, generate_canonical_mock, optimize_overflow


def _run_case(tmp_path: Path, multiplier: float) -> tuple[dict, pd.Series]:
    out = tmp_path / f"pressure_{multiplier:.2f}"
    budget_shares = {"md": 0.10, "np": 0.10, "rn": 0.10, "hcw": 0.10}
    paths = generate_canonical_mock(
        seed=42,
        facilities=4,
        days=30,
        budget_shares=budget_shares,
        out_dir=out,
        overflow_shift_hours=10,
        demand_pressure_multiplier=multiplier,
    )
    ptd = build_ptd_scenarios(paths, out)
    optimize_overflow(paths, ptd, out, budget_shares, min_overflow_covered_gap_hours=4.0)
    meta = json.loads((out / "fixed_rotation_overflow.meta.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(out / "overflow_reallocation_summary.csv")
    row = summary[summary["scenario_name"] == "tabfm_guarded"].iloc[0]
    return meta, row


def test_normal_pressure_has_slack_and_no_overflow_need(tmp_path: Path) -> None:
    meta, row = _run_case(tmp_path, 1.00)
    assert meta["pressure_regime"] == "normal"
    assert meta["pre_overflow_gap_hours"] == 0.0
    assert row["static_shortage_hours"] == 0.0
    assert row["total_optimized_overflow_hours_used"] == 0.0
    for slack_pct in meta["slack_pct_by_role"].values():
        assert 0.10 <= slack_pct <= 0.30


def test_pressure_regimes_increase_overflow_need_monotonically(tmp_path: Path) -> None:
    normal_meta, normal = _run_case(tmp_path, 1.00)
    moderate_meta, moderate = _run_case(tmp_path, 1.25)
    surge_meta, surge = _run_case(tmp_path, 1.35)
    severe_meta, severe = _run_case(tmp_path, 1.60)
    extreme_meta, extreme = _run_case(tmp_path, 2.00)

    assert normal_meta["pre_overflow_gap_hours"] == 0.0
    assert 0.0 < moderate_meta["pre_overflow_gap_hours"] < 700.0
    assert surge_meta["pre_overflow_gap_hours"] > moderate_meta["pre_overflow_gap_hours"] * 3
    assert severe_meta["pre_overflow_gap_hours"] > surge_meta["pre_overflow_gap_hours"]
    assert extreme_meta["pre_overflow_gap_hours"] >= severe_meta["pre_overflow_gap_hours"]
    assert normal["total_optimized_overflow_hours_used"] == 0.0
    assert 0.0 < moderate["total_optimized_overflow_hours_used"] < surge["total_optimized_overflow_hours_used"]
    assert surge["total_optimized_overflow_hours_used"] <= surge["total_overflow_budget_hours"]
    assert surge["total_optimized_overflow_hours_used"] <= severe["total_optimized_overflow_hours_used"] <= severe["total_overflow_budget_hours"]
    assert severe["total_optimized_overflow_hours_used"] <= extreme["total_optimized_overflow_hours_used"] <= extreme["total_overflow_budget_hours"]
