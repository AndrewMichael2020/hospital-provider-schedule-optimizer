from __future__ import annotations

import pandas as pd
from pathlib import Path

from run_daily_audit_simulation import run_daily_audit


def _build_tables(days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timestamps = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    facilities = ["FAC_A", "FAC_B"]
    rows: list[dict] = []
    for idx, ts in enumerate(timestamps):
        facility = facilities[idx % len(facilities)]
        rows.append(
            {
                "facility_id": facility,
                "timestamp": ts,
                "y_pred": 20 + (idx % 5),
                "current_docs_on_duty": 2.0,
                "current_nps_on_duty": 1.0,
                "occupancy_pressure": 0.6,
            }
        )
    forecast = pd.DataFrame(rows)

    # use same table as actual in this synthetic sanity case
    actual = forecast.copy()
    actual["shortage_risk_next_shift"] = 0

    # minimal candidate shifts over each day for each provider
    providers = [("MD", "PR001"), ("NP", "PR002"), ("MD", "PR003")]
    candidate_rows: list[dict] = []
    shift_hours = 24
    for day in range(days):
        for role, pid in providers:
            start = timestamps[day * 24]
            end = start + pd.Timedelta(hours=shift_hours)
            candidate_rows.append(
                {
                    "facility_id": "FAC_A",
                    "provider_id": f"{pid}_{day}",
                    "role": role,
                    "shift_start": start,
                    "shift_end": end,
                    "planned_shift_hours": shift_hours,
                    "unavailable_hours": 0,
                }
            )
    providers_df = pd.DataFrame(candidate_rows)
    return forecast, actual, providers_df


def test_audit_rolls_over_requested_days(tmp_path: Path) -> None:
    forecast, actual, providers = _build_tables(days=14)
    out = tmp_path / "audit"
    out.mkdir(parents=True, exist_ok=True)
    run_daily_audit(
        forecast=forecast,
        actual=actual,
        provider_df=providers,
        out_dir=out,
        audit_days=14,
        optimizer="greedy",
        shift_hours=4,
        min_shift_hours=4,
        plan_horizon_hours=24,
        doc_capacity_per_hour=4.0,
        np_capacity_per_hour=3.0,
        print_top_n=3,
        closed_loop=False,
    )
    summary = out / "daily_audit_summary.json"
    detail = out / "daily_audit_detail.csv"
    assert summary.exists()
    assert detail.exists()
    detail_df = pd.read_csv(detail)
    assert len(detail_df) == 14
    assert set(detail_df.columns) >= {"date", "shortage_hours"}
