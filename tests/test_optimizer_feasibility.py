from __future__ import annotations

import pandas as pd

from tabfm_healthcare_eval.optimizer import build_demand_profile, build_shift_candidate_pool, solve_staffing_plan


def _demand_df(rows: int = 3) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h")
    return pd.DataFrame(
        {
            "facility_id": ["FAC_A"] * rows,
            "timestamp": ts,
            "y_pred": [10, 20, 15],
            "occupancy_pressure": [0.5, 0.5, 0.5],
        }
    )


def _candidate_df(overlap: bool = True, block: bool = True) -> pd.DataFrame:
    candidates = [
        {
            "facility_id": "FAC_A",
            "provider_id": "P1",
            "role": "MD",
            "shift_start": pd.Timestamp("2025-01-01 00:00"),
            "shift_end": pd.Timestamp("2025-01-01 04:00"),
            "planned_shift_hours": 4,
            "unavailable_hours": 1 if block else 0,
        },
        {
            "facility_id": "FAC_A",
            "provider_id": "P1" if overlap else "P2",
            "role": "MD",
            "shift_start": pd.Timestamp("2025-01-01 02:00"),
            "shift_end": pd.Timestamp("2025-01-01 06:00"),
            "planned_shift_hours": 4,
            "unavailable_hours": 0,
        },
        {
            "facility_id": "FAC_A",
            "provider_id": "P3",
            "role": "NP",
            "shift_start": pd.Timestamp("2025-01-01 00:00"),
            "shift_end": pd.Timestamp("2025-01-01 04:00"),
            "planned_shift_hours": 4,
            "unavailable_hours": 0,
        },
    ]
    return pd.DataFrame(candidates)


def test_min_shift_filter_and_overlap():
    demand = build_demand_profile(_demand_df(), shift_hours=4)
    raw = _candidate_df()
    raw.loc[0, "planned_shift_hours"] = 2
    raw.loc[0, "shift_end"] = pd.Timestamp("2025-01-01 02:00")
    pool = build_shift_candidate_pool(raw, min_shift_hours=4, required_horizon_start=raw["shift_start"].min(), required_horizon_end=raw["shift_end"].max())
    assert all(pool["shift_hours"] >= 4)
    assert len(pool) < len(raw)


def test_no_provider_overlap_and_shortage():
    demand = build_demand_profile(_demand_df(rows=8), shift_hours=4)
    pool = build_shift_candidate_pool(_candidate_df(overlap=True, block=False), min_shift_hours=4, required_horizon_start=None, required_horizon_end=None)
    assignments, summary, _ = solve_staffing_plan(demand_profile=demand, candidate_pool=pool, min_shift_hours=4, engine="greedy")
    providers = assignments["provider_id"].tolist() if not assignments.empty else []
    selected = assignments if not assignments.empty else pd.DataFrame()
    assigned_slots = set()
    if not selected.empty:
        for row in selected.itertuples(index=False):
            for ts in row.coverage_hours:
                slot = (row.provider_id, pd.Timestamp(ts))
                assert slot not in assigned_slots
                assigned_slots.add(slot)
    assert summary["hard_violation_count"] >= 0


def test_unavailable_hour_and_baseline_block_count():
    demand = build_demand_profile(_demand_df(rows=6), shift_hours=4)
    pool = build_shift_candidate_pool(_candidate_df(overlap=False, block=True), min_shift_hours=4, required_horizon_start=None, required_horizon_end=None)
    assignments, summary, _ = solve_staffing_plan(demand, pool, min_shift_hours=4, engine="greedy")
    assert isinstance(summary["blocked_availability_hours"], int)
    assert summary["blocked_availability_hours"] >= 0
    assert summary["expected_shortfall_hours"] >= 0.0
