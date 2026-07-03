from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StaffingResult:
    facility_id: str
    shift_start: pd.Timestamp
    shift_end: pd.Timestamp
    shift_hours: int
    predicted_arrivals: float
    required_docs: float
    required_nps: float
    shortage_hours: float
    baseline_shortage_hours: float
    utilization: float
    recommended_docs: int
    recommended_nps: int
    availability_blocked: bool
    docs_availability: float
    nps_availability: float


def _shift_label(ts: pd.Timestamp, shift_hours: int) -> pd.Timestamp:
    hour = int(ts.floor("h").hour)
    start = hour - (hour % shift_hours)
    return pd.Timestamp(ts.floor("h")).replace(hour=start, minute=0, second=0, microsecond=0)


def to_shift_load(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    arrival_col: str = "predicted_arrivals",
    shift_hours: int = 4,
    facility_col: str = "facility_id",
) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values([facility_col, timestamp_col])
    data["shift_start"] = data.apply(
        lambda r: _shift_label(pd.to_datetime(r[timestamp_col]), shift_hours),
        axis=1,
    )
    grouped = data.groupby([facility_col, "shift_start"], as_index=False).agg(
        predicted_arrivals=(arrival_col, "sum"),
        current_docs_on_duty=("current_docs_on_duty", "mean"),
        current_nps_on_duty=("current_nps_on_duty", "mean"),
        docs_availability_min=("current_docs_on_duty", "min"),
        nps_availability_min=("current_nps_on_duty", "min"),
        bed_occupancy_pct=("bed_occupancy_pct", "mean"),
        shortage_risk_next_shift=("shortage_risk_next_shift", "max") if "shortage_risk_next_shift" in data.columns else (arrival_col, "mean"),
    )
    grouped["shift_end"] = grouped["shift_start"] + pd.to_timedelta(shift_hours, unit="h")
    grouped["shift_hours"] = int(shift_hours)
    return grouped


def recommend_staffing(
    shift_df: pd.DataFrame,
    doc_capacity_per_hour: float = 4.0,
    np_capacity_per_hour: float = 3.0,
    np_ratio_max: float = 0.7,
    min_shift_hours: int = 4,
) -> list[StaffingResult]:
    default_shift_hours = int(min_shift_hours) if int(min_shift_hours) > 0 else 4
    shift_hours = int(shift_df["shift_hours"].iloc[0]) if not shift_df.empty else default_shift_hours
    shift_hours = max(default_shift_hours, shift_hours)
    min_shift_hours = max(1, int(min_shift_hours))
    doc_capacity_per_hour = float(np.nan_to_num(doc_capacity_per_hour, nan=0.0))
    np_capacity_per_hour = float(np.nan_to_num(np_capacity_per_hour, nan=0.0))
    np_ratio_max = float(np.clip(np_ratio_max, 1e-6, 1.0))

    out = []
    for row in shift_df.itertuples(index=False):
        facility = getattr(row, "facility_id")
        shift_start = getattr(row, "shift_start")
        pred_arrivals = float(np.nan_to_num(getattr(row, "predicted_arrivals"), nan=0.0))
        curr_docs = float(np.nan_to_num(getattr(row, "current_docs_on_duty", 0.0), nan=0.0))
        curr_nps = float(np.nan_to_num(getattr(row, "current_nps_on_duty", 0.0), nan=0.0))
        docs_avail = float(np.nan_to_num(getattr(row, "docs_availability_min", curr_docs), nan=curr_docs))
        nps_avail = float(np.nan_to_num(getattr(row, "nps_availability_min", curr_nps), nan=curr_nps))
        # Conservative capacity is based on minimum staffing across the shift window
        bed_pressure = float(np.nan_to_num(getattr(row, "bed_occupancy_pct", 0.5), nan=0.5))

        shift_len = int(np.nan_to_num(getattr(row, "shift_hours", shift_hours), nan=shift_hours))
        shift_len = max(shift_len, min_shift_hours)

        current_capacity = (
            curr_docs * doc_capacity_per_hour * shift_len
            + curr_nps * np_capacity_per_hour * shift_len
        )
        # Demand should be at least non-zero and grows with occupancy pressure.
        required_capacity = max(pred_arrivals, 1.0) * (0.55 + 0.30 * bed_pressure)

        # Assume 1 staff-hour covers demand units; use integer rounding for simplicity.
        doc_share = 0.65
        np_share = 1.0 - doc_share
        required_docs = int(np.ceil(required_capacity * doc_share / (doc_capacity_per_hour * shift_len)))
        required_nps = int(np.ceil(required_capacity * np_share / (np_capacity_per_hour * shift_len)))
        required_docs = max(0, required_docs)
        required_nps = max(0, required_nps)

        # Enforce minimum 4h block and provider availability sanity checks.
        if required_docs > 0:
            required_docs = max(1, required_docs)
        if required_nps > 0:
            required_nps = max(1, required_nps)
        max_doc_by_ratio = max(0, int(np.ceil(required_docs / np_ratio_max)))
        if required_nps > max_doc_by_ratio:
            required_nps = max_doc_by_ratio

        # Keep recommendations feasible under present-hour availability constraints.
        # If availability is lower than planned staff we cap and mark shortfall separately.
        feasible_docs = min(required_docs, max(0, int(np.floor(max(docs_avail, 0.0)))))
        feasible_nps = min(required_nps, max(0, int(np.floor(max(nps_avail, 0.0)))))
        # Keep non-zero recommendation only where true demand exists.
        planned_docs = max(0, feasible_docs)
        planned_nps = max(0, feasible_nps)

        baseline_shortage = max(0.0, required_capacity - current_capacity)
        planned_capacity = (
            planned_docs * doc_capacity_per_hour * shift_len
            + planned_nps * np_capacity_per_hour * shift_len
        )
        if planned_capacity <= 0.0:
            shortage_hours = float(required_capacity)
            utilization = 0.0
        else:
            shortage_hours = max(0.0, (required_capacity - planned_capacity) / planned_capacity)
            utilization = min(1.0, required_capacity / planned_capacity)
        blocked = bool(planned_capacity < required_capacity)

        out.append(
            StaffingResult(
                facility_id=facility,
                shift_start=shift_start,
                shift_end=shift_start + pd.Timedelta(hours=shift_len),
                shift_hours=shift_len,
                predicted_arrivals=pred_arrivals,
                required_docs=float(planned_docs),
                required_nps=float(planned_nps),
                shortage_hours=float(shortage_hours),
                baseline_shortage_hours=float(baseline_shortage),
                utilization=float(utilization),
                recommended_docs=int(planned_docs),
                recommended_nps=int(planned_nps),
                availability_blocked=blocked,
                docs_availability=float(docs_avail),
                nps_availability=float(nps_avail),
            )
        )
    return out


def summarize_coverage_kpis(recs: list[StaffingResult], penalty_doc=1.0, penalty_np=0.6) -> dict:
    if not recs:
        return {"expected_shortfall_hours": 0.0, "sla_risk_reduction_proxy": 0.0, "utilization": 0.0, "overtime_proxy": 0.0}
    shortage_sum = sum(r.shortage_hours for r in recs)
    baseline_shortage = sum(r.baseline_shortage_hours for r in recs)
    util = float(np.nanmean([r.utilization for r in recs]))
    overtime = float(np.mean([max(r.recommended_docs, 0) for r in recs]) * penalty_doc + np.mean([max(r.recommended_nps, 0) for r in recs]) * penalty_np)
    return {
        "expected_shortfall_hours": float(shortage_sum),
        "sla_risk_reduction_proxy": float(max(0.0, baseline_shortage - shortage_sum)),
        "utilization": util,
        "overtime_proxy": float(overtime),
    }
