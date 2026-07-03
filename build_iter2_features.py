from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from tabfm_healthcare_eval.utils import set_deterministic_seed, safe_json_dump, write_parquet_if_dataframe


def _safe_mode(series: pd.Series) -> str | float | pd._libs.missing.NAType:
    if series.empty:
        return pd.NA
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.mode(dropna=True).iloc[0]


def _build_shift_labels(ts: pd.Series) -> pd.Series:
    # Keep the same 4-hour bins used by the scheduler.
    return ts.dt.floor("h").dt.strftime("%Y-%m-%d %H:00:00").astype("datetime64[ns]")


def build_iter2_feature_table(
    visits_path: str,
    capacity_path: str,
    present_shifts_path: str,
    output_path: str,
    seed: int = 42,
) -> pd.DataFrame:
    set_deterministic_seed(seed)
    rng = np.random.default_rng(seed)

    visits = pd.read_parquet(visits_path)
    capacity = pd.read_parquet(capacity_path)
    present = pd.read_parquet(present_shifts_path)

    visits = visits.copy()
    capacity = capacity.copy()
    present = present.copy()

    for col in ["event_ts", "timestamp_hour"]:
        if col in visits.columns:
            visits[col] = pd.to_datetime(visits[col])
        if col in capacity.columns:
            capacity[col] = pd.to_datetime(capacity[col])

    visits["timestamp_hour"] = visits["event_ts"].dt.floor("h")

    visit_type_map = {
        "arrive": 1,
        "handoff": 2,
        "assign": 3,
        "disposition": 4,
        "transfer_out": 5,
    }
    visits["event_type_code"] = visits["event_type"].map(visit_type_map).fillna(9).astype(float)

    # Build visit-level operational aggregates per facility-hour.
    grouped = (
        visits.groupby(["facility_id", "timestamp_hour"], as_index=False)
        .agg(
            arrivals_count=("event_type", lambda s: int((s == "arrive").sum())),
            handoffs_count=("event_type", lambda s: int((s == "handoff").sum())),
            transitions_count=("event_type", lambda s: int(((s == "assign") | (s == "transfer_out")).sum())),
            distinct_visits=("visit_id", "nunique"),
            triage_entropy=("triage_level", pd.Series.nunique),
            complaints_nunique=("chief_complaint", pd.Series.nunique),
            room_nunique=("room_id", pd.Series.nunique),
            avg_los=("length_of_stay_min", "mean"),
            p90_los=("length_of_stay_min", lambda s: float(np.nanpercentile(s.astype(float), 90)) if len(s) else 0.0),
            avg_duration_to_next=("duration_to_next_event_min", "mean"),
            provider_missing_rate=("provider_id", lambda s: float(s.isna().mean())),
            avg_acuity=("acuity_score", "mean"),
            event_type_code_mean=("event_type_code", "mean"),
            arrivals_mode=(
                "arrival_mode",
                lambda s: s.mode(dropna=True).iloc[0] if len(s.dropna()) else pd.NA,
            ),
            )
        .sort_values(["facility_id", "timestamp_hour"])
        .reset_index(drop=True)
    )

    grouped["facility_timestamp_rank"] = grouped.groupby("facility_id").cumcount()

    # high-cardinality proxies from per-hour activity
    top_provider = (
        visits.groupby(["facility_id", "timestamp_hour"])["provider_id"].agg(_safe_mode).reset_index().rename(columns={"provider_id": "top_provider"})
    )
    top_room = visits.groupby(["facility_id", "timestamp_hour"])["room_id"].agg(_safe_mode).reset_index().rename(columns={"room_id": "top_room"})
    top_to_provider = (
        visits.groupby(["facility_id", "timestamp_hour"])["to_provider_id"].agg(_safe_mode).reset_index().rename(columns={"to_provider_id": "top_to_provider"})
    )
    grouped = grouped.merge(top_provider, on=["facility_id", "timestamp_hour"], how="left")
    grouped = grouped.merge(top_room, on=["facility_id", "timestamp_hour"], how="left")
    grouped = grouped.merge(top_to_provider, on=["facility_id", "timestamp_hour"], how="left")

    # Deterministic high-cardinality stable hash buckets for provider/room IDs.
    for col in ["top_provider", "top_room", "top_to_provider"]:
        grouped[f"{col}_bucket"] = grouped[col].astype(str).map(
            lambda s: f"{col}_{int(hashlib.md5(str(s).encode('utf-8')).hexdigest()[:6], 16)}" if pd.notna(s) else pd.NA
        )

    # Temporal signals
    ts = pd.to_datetime(grouped["timestamp_hour"])
    grouped["hour_of_day"] = ts.dt.hour.astype(int)
    grouped["day_of_week"] = ts.dt.dayofweek.astype(int)
    grouped["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    grouped["is_holiday"] = ts.dt.strftime("%m-%d").isin({"01-01", "07-01", "12-25"}).astype(int)
    grouped["shift_label"] = _build_shift_labels(ts)
    grouped["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    grouped["hour_sin"] = np.sin(2.0 * np.pi * grouped["hour_of_day"] / 24.0)
    grouped["hour_cos"] = np.cos(2.0 * np.pi * grouped["hour_of_day"] / 24.0)
    grouped["dow_sin"] = np.sin(2.0 * np.pi * grouped["day_of_week"] / 7.0)
    grouped["dow_cos"] = np.cos(2.0 * np.pi * grouped["day_of_week"] / 7.0)
    grouped["is_month_start"] = ts.dt.is_month_start.astype(int)
    grouped["is_month_end"] = ts.dt.is_month_end.astype(int)

    grouped = grouped.sort_values(["facility_id", "timestamp_hour"]).reset_index(drop=True)
    for lag in (1, 2, 3, 4):
        grouped[f"arrivals_lag_{lag}h"] = grouped.groupby("facility_id")["arrivals_count"].shift(lag)
    grouped["demand_delta_1h"] = grouped.groupby("facility_id")["arrivals_count"].diff(1)
    grouped["demand_delta_2h"] = grouped.groupby("facility_id")["arrivals_count"].diff(2)
    for window in (3, 6, 12):
        grouped[f"arrivals_roll_mean_{window}h"] = grouped.groupby("facility_id")["arrivals_count"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        grouped[f"arrivals_roll_std_{window}h"] = grouped.groupby("facility_id")["arrivals_count"].transform(
            lambda s: s.rolling(window=window, min_periods=1).std(ddof=0)
        )

    # Merge capacity and staffing context
    capacity_cols = [c for c in capacity.columns if c not in {"facility_id", "timestamp_hour"}]
    grouped = grouped.merge(capacity, on=["facility_id", "timestamp_hour"], how="left", suffixes=("", "_cap"))
    present_cols = ["facility_id", "timestamp_hour", "md_on_duty", "np_on_duty", "rn_on_duty"]
    grouped = grouped.merge(present[present_cols], on=["facility_id", "timestamp_hour"], how="left")

    # Replace missing staffing with facility-level medians for robustness.
    grouped["md_on_duty"] = grouped["md_on_duty"].fillna(grouped.groupby("facility_id")["md_on_duty"].transform("median"))
    grouped["np_on_duty"] = grouped["np_on_duty"].fillna(grouped.groupby("facility_id")["np_on_duty"].transform("median"))
    grouped["rn_on_duty"] = grouped["rn_on_duty"].fillna(grouped.groupby("facility_id")["rn_on_duty"].transform("median"))
    grouped[["md_on_duty", "np_on_duty", "rn_on_duty"]] = grouped[["md_on_duty", "np_on_duty", "rn_on_duty"]].fillna(0.0)

    grouped["room_capacity"] = grouped["open_rooms"]
    grouped["current_docs_on_duty"] = grouped["md_on_duty"] + grouped["rn_on_duty"] * 0.22
    grouped["current_nps_on_duty"] = grouped["np_on_duty"] * 0.9

    # Arrival-pressure derived labels for future horizon.
    grouped["expected_arrivals_next_hour"] = grouped.groupby("facility_id")["arrivals_count"].shift(-1).fillna(0)
    demand_4h = (
        grouped.groupby("facility_id")["arrivals_count"]
        .transform(
            lambda s: s.shift(-1)
            + s.shift(-2).fillna(0)
            + s.shift(-3).fillna(0)
            + s.shift(-4).fillna(0)
        )
    )
    grouped["expected_arrivals_next_4h"] = demand_4h.fillna(0)

    effective_staff_capacity = grouped["current_docs_on_duty"] * 4.0 + grouped["current_nps_on_duty"] * 2.8
    margin = grouped["expected_arrivals_next_4h"] - effective_staff_capacity
    margin_scale = margin.std() if margin.std() and not np.isnan(margin.std()) else 1.0
    if margin_scale == 0:
        margin_scale = 1.0
    risk_score = 1.0 / (1.0 + np.exp(-margin / margin_scale))
    # sample a noisy binary label so that scheduling risk is not purely deterministic
    grouped["shortage_risk_next_shift"] = [
        int(rng.binomial(1, float(np.clip(p, 0.01, 0.99)))) for p in (risk_score.to_numpy())
    ]

    lag_columns = [c for c in grouped.columns if "arrivals_lag_" in c or c.startswith("demand_delta_")]
    grouped[lag_columns] = grouped[lag_columns].fillna(0.0)

    # Keep only supervised-ready rows
    grouped = grouped.dropna(subset=["expected_arrivals_next_hour"]).reset_index(drop=True)

    # Keep deterministic schema
    grouped["timestamp"] = grouped["timestamp_hour"].dt.floor("h")
    feature_cols = [c for c in grouped.columns if c not in {"timestamp_hour", "event_type_code", "shift_label"}]
    grouped = grouped[feature_cols].copy()

    grouped["expected_arrivals_next_shift_bucket"] = pd.cut(
        grouped["expected_arrivals_next_4h"],
        bins=[-1, 8, 16, 24, 40, 80, 160, 1e9],
        labels=["vlow", "low", "med", "high", "xhigh", "critical", "extreme"],
    )

    grouped["facility_key"] = grouped["facility_id"].astype("category")
    grouped["arrival_mode_mode"] = grouped["arrivals_mode"]
    grouped["arrival_mode_mode"] = grouped["arrival_mode_mode"].fillna("missing")

    grouped["nurse_like_staff_ratio"] = grouped["current_nps_on_duty"] / (grouped["current_docs_on_duty"] + 1.0)
    grouped["occupancy_pressure"] = (
        grouped["bed_occupancy_pct"].astype(float) * 0.7 + grouped["rooms_occupied_pct"].astype(float) * 0.3
    )
    grouped["case_mix_proxy"] = grouped["avg_acuity"].fillna(grouped["avg_acuity"].median()) * grouped["complaints_nunique"].fillna(0)
    grouped["occupancy_pressure_change"] = grouped.groupby("facility_id")["occupancy_pressure"].diff().fillna(0.0)
    grouped["rooms_change"] = grouped.groupby("facility_id")["rooms_occupied_pct"].diff().fillna(0.0)
    grouped = grouped.sort_values(["facility_id", "timestamp"]).reset_index(drop=True)

    # Optional quality report saved to metadata
    grouped["is_impossible_arrival"] = (grouped["expected_arrivals_next_hour"] < 0).astype(int)

    write_parquet_if_dataframe(grouped, output_path)
    safe_json_dump(
        {
            "n_rows": int(len(grouped)),
            "n_facilities": int(grouped["facility_id"].nunique()),
            "timestamp_min": str(grouped["timestamp"].min()),
            "timestamp_max": str(grouped["timestamp"].max()),
            "missing_fraction": float(grouped.isna().mean().mean()),
        },
        str(Path(output_path).with_suffix(".meta.json")),
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build supervised features from iteration-2 scheduling raw data")
    parser.add_argument("--visits", required=True)
    parser.add_argument("--capacity", required=True)
    parser.add_argument("--present-shifts", required=True, dest="present_shifts")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-top", action="store_true", default=True)
    args = parser.parse_args()

    df = build_iter2_feature_table(
        args.visits,
        args.capacity,
        args.present_shifts,
        args.out,
        seed=args.seed,
    )

    if args.print_top:
        print("=== model-table top rows ===")
        print(df.head(10).to_string(index=False))
        print("=== model-table tail rows ===")
        print(df.tail(5).to_string(index=False))
        missing = float(df.isna().mean().max())
        print(f"max missing fraction per column={missing:.4f}")
        print(f"dataset shape={df.shape}")


if __name__ == "__main__":
    main()
