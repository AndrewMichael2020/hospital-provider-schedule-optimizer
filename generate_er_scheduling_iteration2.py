from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tabfm_healthcare_eval.utils import set_deterministic_seed, write_parquet_if_dataframe, safe_json_dump


@dataclass(frozen=True)
class Iter2GeneratorConfig:
    seed: int = 42
    n_facilities: int = 4
    n_days: int = 14
    start_datetime: str = "2025-01-01 00:00:00"
    output_prefix: str = "data/er_iter2"
    min_shift_hours: int = 4


def _build_facilities(rng: np.random.Generator, n: int) -> pd.DataFrame:
    facilities = []
    for facility_idx in range(n):
        facilities.append(
            {
                "facility_id": f"FAC_{facility_idx:03d}",
                "facility_ed_type": rng.choice(["urban", "academic", "community", "rural"]),
                "region": rng.choice(["north", "south", "east", "west", "central"]),
                "bed_capacity": int(rng.integers(40, 130)),
                "room_capacity": int(rng.integers(30, 80)),
                "base_md": int(rng.integers(10, 28)),
                "base_np": int(rng.integers(10, 26)),
                "base_rn": int(rng.integers(8, 24)),
            }
        )
    return pd.DataFrame(facilities)


def _build_unavailable_offsets(rng: np.random.Generator, shift_len: int) -> set[int]:
    unavailable: set[int] = set()
    if shift_len <= 1:
        return unavailable
    # ~25% of providers have at least one blocked block inside a shift.
    if rng.random() > 0.24:
        return unavailable
    n_blocks = int(rng.integers(1, 3))
    max_block_len = max(1, min(3, max(1, shift_len // 3)))
    for _ in range(n_blocks):
        block_len = int(rng.integers(1, max_block_len + 1))
        if block_len >= shift_len:
            block_len = max(1, shift_len - 1)
        start_off = int(rng.integers(0, max(1, shift_len - block_len + 1)))
        for h in range(start_off, start_off + block_len):
            unavailable.add(int(h) if h < shift_len else int(shift_len - 1))
    return set(i for i in unavailable if i < shift_len)


def _sample_shift_start_hour(rng: np.random.Generator, horizon_hours: int, shift_len: int) -> int:
    """Sample realistic shift start hour in a 24h day."""
    day_anchor = int(rng.integers(0, max(1, horizon_hours - shift_len)))
    hour_of_day = int(rng.integers(0, 24))
    day_window = int(day_anchor / 24) if horizon_hours >= 24 else 0

    # Morning/evening/night start buckets with bias.
    buckets = np.array(
        [0, 0, 1, 1, 2, 3, 6, 6, 7, 7, 8, 14, 14, 15, 16, 18, 18, 19, 20, 21, 21, 22, 23]
    )
    weights = np.array([0.35, 0.35, 0.2, 0.1, 1.0, 1.0, 1.6, 1.6, 1.5, 1.2, 1.4, 1.4, 1.1, 1.1, 1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7]) / 2.0
    # Keep lengths and weights aligned for robust sampling.
    if len(weights) != len(buckets):
        weights = np.ones(len(buckets), dtype=float)
    weights = weights / weights.sum()
    sampled_hour = int(rng.choice(buckets, p=weights))
    preferred_hour = day_window * 24 + sampled_hour

    day_start = day_window * 24
    day_end = min(horizon_hours - shift_len, day_start + 23)
    if day_start <= day_end:
        start_hour = int(np.clip(preferred_hour, day_start, day_end))
    else:
        start_hour = int(max(0, min(day_anchor, horizon_hours - shift_len)))
    return start_hour


def _expand_shift_rows(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    provider_id: str,
    role: str,
    facility_id: str,
    unavailable_offsets: set[int] | None = None,
    is_baseline_roster: bool = True,
    recovery_lever: str = "baseline",
) -> list[dict]:
    start = start_ts.floor("h")
    end = end_ts.floor("h")
    unavailable_offsets = set() if unavailable_offsets is None else set(int(i) for i in unavailable_offsets)
    hours = pd.date_range(start, end, freq="1h", inclusive="left")
    rows: list[dict] = []
    for idx, ts in enumerate(hours):
        if idx in unavailable_offsets:
            continue
        rows.append(
            {
                "facility_id": facility_id,
                "provider_id": provider_id,
                "role": role,
                "shift_start": start,
                "shift_end": end,
                "timestamp_hour": ts,
                "is_available_hour": 1,
                "is_baseline_roster": int(bool(is_baseline_roster)),
                "recovery_lever": recovery_lever,
            }
        )
    return rows


def _build_provider_shifts(
    rng: np.random.Generator,
    facilities: pd.DataFrame,
    horizon_hours: int,
    min_shift_hours: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shift_rows: list[dict] = []
    provider_rows: list[dict] = []
    provider_id = 0

    for facility in facilities.itertuples(index=False):
        facility_id = facility.facility_id
        n_providers = int(np.clip(rng.normal(facility.base_md + facility.base_np + 32, 14), 45, 125))
        n_providers = int(max(30, min(130, n_providers)))
        for _ in range(n_providers):
            role = rng.choice(["MD", "NP", "RN"], p=[0.45, 0.35, 0.20])
            candidate_lengths = np.array([4, 6, 8, 10, 12], dtype=int)
            candidate_probs = np.array([0.03, 0.17, 0.38, 0.24, 0.18], dtype=float)
            valid = candidate_lengths >= min_shift_hours
            lengths = candidate_lengths[valid]
            probs = candidate_probs[valid]
            probs = probs / probs.sum()
            p_id = f"PR_{provider_id:05d}"
            provider_id += 1

            # Each provider can have a handful of candidate shifts across the horizon,
            # including split availability patterns that reflect real rosters.
            shifts_per_provider = int(rng.integers(1, 4))
            for shift_idx in range(shifts_per_provider):
                shift_len = int(rng.choice(lengths, p=probs))
                shift_len = max(min_shift_hours, int(shift_len))
                first_hour = _sample_shift_start_hour(rng, horizon_hours, shift_len)
                on_call = bool(rng.random() < 0.12)
                start = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=int(first_hour))
                end = start + pd.Timedelta(hours=int(shift_len))
                if end <= start or shift_len < min_shift_hours:
                    continue
                unavailable_offsets = _build_unavailable_offsets(rng, shift_len)
                available_hours = max(0, shift_len - len(unavailable_offsets))
                candidate_id = f"{p_id}_{shift_idx:02d}"
                provider_rows.append(
                    {
                        "provider_id": candidate_id,
                        "facility_id": facility_id,
                        "role": role,
                        "shift_start": start,
                        "shift_end": end,
                        "on_call": on_call,
                        "max_contiguous_hours": shift_len,
                        "planned_shift_hours": shift_len,
                        "available_hours": available_hours,
                        "unavailable_hours": int(len(unavailable_offsets)),
                        "availability_ratio": float(available_hours / max(1, shift_len)),
                        "base_provider_id": p_id,
                        "is_baseline_roster": 1,
                        "is_float_pool": 0,
                        "is_cross_facility": 0,
                        "recovery_lever": "baseline",
                        "max_monthly_hours": float(max(40, shift_len * shifts_per_provider)),
                    }
                )
                shift_rows.extend(
                    _expand_shift_rows(
                        start,
                        end,
                        candidate_id,
                        role,
                        facility_id,
                        unavailable_offsets=unavailable_offsets,
                        is_baseline_roster=True,
                        recovery_lever="baseline",
                    )
                )
        for day in range(int(np.ceil(horizon_hours / 24))):
            for start_hour, shift_len in [(0, 7), (7, 12), (19, 12)]:
                start_offset = day * 24 + start_hour
                if start_offset >= horizon_hours:
                    continue
                for role in ["MD", "NP"]:
                    for pool_idx in range(2):
                        start = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=int(start_offset))
                        end = min(start + pd.Timedelta(hours=int(shift_len)), pd.Timestamp("2025-01-01") + pd.Timedelta(hours=int(horizon_hours)))
                        if (end - start).total_seconds() / 3600 < min_shift_hours:
                            continue
                        p_id = f"FP_{facility_id}_{day:03d}_{start_hour:02d}_{role}_{pool_idx}"
                        unavailable_offsets = set()
                        provider_rows.append(
                            {
                                "provider_id": p_id,
                                "facility_id": facility_id,
                                "role": role,
                                "shift_start": start,
                                "shift_end": end,
                                "on_call": True,
                                "max_contiguous_hours": int((end - start).total_seconds() // 3600),
                                "planned_shift_hours": int((end - start).total_seconds() // 3600),
                                "available_hours": int((end - start).total_seconds() // 3600),
                                "unavailable_hours": 0,
                                "availability_ratio": 1.0,
                                "base_provider_id": p_id,
                                "is_baseline_roster": 0,
                                "is_float_pool": 1,
                                "is_cross_facility": 0,
                                "recovery_lever": "float_pool",
                        "max_monthly_hours": 24.0,
                            }
                        )
                        shift_rows.extend(
                            _expand_shift_rows(
                                start,
                                end,
                                p_id,
                                role,
                                facility_id,
                                unavailable_offsets=unavailable_offsets,
                                is_baseline_roster=False,
                                recovery_lever="float_pool",
                            )
                        )

            for role in ["MD", "NP"]:
                start = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=int(day * 24 + 11))
                end = start + pd.Timedelta(hours=4)
                if start >= pd.Timestamp("2025-01-01") + pd.Timedelta(hours=int(horizon_hours)):
                    continue
                p_id = f"OT_{facility_id}_{day:03d}_{role}"
                provider_rows.append(
                    {
                        "provider_id": p_id,
                        "facility_id": facility_id,
                        "role": role,
                        "shift_start": start,
                        "shift_end": end,
                        "on_call": True,
                        "max_contiguous_hours": 4,
                        "planned_shift_hours": 4,
                        "available_hours": 4,
                        "unavailable_hours": 0,
                        "availability_ratio": 1.0,
                        "base_provider_id": p_id,
                        "is_baseline_roster": 0,
                        "is_float_pool": 0,
                        "is_cross_facility": 0,
                        "recovery_lever": "overtime_extension",
                        "max_monthly_hours": 16.0,
                    }
                )
                shift_rows.extend(
                    _expand_shift_rows(
                        start,
                        end,
                        p_id,
                        role,
                        facility_id,
                        is_baseline_roster=False,
                        recovery_lever="overtime_extension",
                    )
                )

            # Limited hospitalist-like overflow bank: MD-equivalent surge support only.
            # These rows are intentionally sparse and must be budget-capped by the optimizer.
            for start_hour in [10, 16, 20]:
                start_offset = day * 24 + start_hour
                if start_offset >= horizon_hours:
                    continue
                start = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=int(start_offset))
                end = min(start + pd.Timedelta(hours=4), pd.Timestamp("2025-01-01") + pd.Timedelta(hours=int(horizon_hours)))
                if (end - start).total_seconds() / 3600 < min_shift_hours:
                    continue
                p_id = f"HOSP_{facility_id}_{day:03d}_{start_hour:02d}"
                provider_rows.append(
                    {
                        "provider_id": p_id,
                        "facility_id": facility_id,
                        "role": "HOSPITALIST_OVERFLOW",
                        "shift_start": start,
                        "shift_end": end,
                        "on_call": True,
                        "max_contiguous_hours": 4,
                        "planned_shift_hours": 4,
                        "available_hours": 4,
                        "unavailable_hours": 0,
                        "availability_ratio": 1.0,
                        "base_provider_id": p_id,
                        "is_baseline_roster": 0,
                        "is_float_pool": 0,
                        "is_cross_facility": 1,
                        "recovery_lever": "hospitalist_overflow",
                        "max_monthly_hours": 12.0,
                    }
                )
                shift_rows.extend(
                    _expand_shift_rows(
                        start,
                        end,
                        p_id,
                        "HOSPITALIST_OVERFLOW",
                        facility_id,
                        is_baseline_roster=False,
                        recovery_lever="hospitalist_overflow",
                    )
                )
    return pd.DataFrame(provider_rows), pd.DataFrame(shift_rows)


def _build_weather_pattern(rng: np.random.Generator, timestamps: pd.DatetimeIndex, facility_id: str) -> np.ndarray:
    hours = timestamps.hour.to_numpy()
    dow = timestamps.dayofweek.to_numpy()
    month = timestamps.month.to_numpy()
    holiday = np.array([1.0 if (ts.month == 1 and ts.day == 1) or (ts.month == 12 and ts.day in {24, 25}) else 0.0 for ts in timestamps], dtype=float)
    base = 0.42 + 0.35 * ((np.sin((hours + 1) / 24 * 2 * np.pi) + 1) / 2)
    weekend = (dow >= 5).astype(float) * 0.08
    winter = np.isin(month, [11, 12, 1, 2]).astype(float) * 0.15
    flu = holiday * 0.20
    noise = rng.normal(0.0, 0.06, size=len(timestamps))
    score = np.clip(base + weekend + winter + flu + noise, 0.05, 0.95)
    buckets = np.array(rng.choice(["mild", "rain", "snow", "storm", "cold", "clear"], size=len(timestamps), p=[0.34, 0.17, 0.08, 0.04, 0.14, 0.23]), dtype=object)
    # facility-local perturbation to make IDs matter
    buckets[score > 0.85] = rng.choice(["rain", "storm", "snow"], size=(score > 0.85).sum(), replace=True, p=[0.5, 0.2, 0.3])
    buckets[score < 0.2] = "cold"
    return buckets


def generate_er_iteration2_data(cfg: Iter2GeneratorConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_deterministic_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    facilities = _build_facilities(rng, cfg.n_facilities)
    start = pd.Timestamp(cfg.start_datetime)
    horizon_hours = cfg.n_days * 24
    timestamps = pd.date_range(start, periods=horizon_hours, freq="h")

    shift_table, active_shift_hours = _build_provider_shifts(
        rng,
        facilities,
        horizon_hours,
        min_shift_hours=cfg.min_shift_hours,
    )
    baseline_active = active_shift_hours[active_shift_hours.get("is_baseline_roster", 1).astype(int) == 1].copy()
    present = baseline_active.groupby(["facility_id", "timestamp_hour", "role"]).size().unstack(fill_value=0)
    for col in ["MD", "NP", "RN"]:
        if col not in present.columns:
            present[col] = 0
    present = present.reset_index()
    present = present.rename(columns={"MD": "md_on_duty", "NP": "np_on_duty", "RN": "rn_on_duty"})

    chief = ["trauma", "cardiac", "respiratory", "neurologic", "obstetric", "infectious", "behavioral", "orthopedic"]
    triage = ["1", "2", "3", "4", "5", "urgent", "critical"]
    arrival_mode = ["walk-in", "ambulance", "transfer", "referral", "tele-triage"]
    event_types = ["arrive", "handoff", "assign", "disposition", "transfer_out"]

    visit_rows: list[dict] = []
    provider_lookup: dict[str, list[str]] = {}
    room_lookup: dict[str, list[str]] = {}

    for _, facility in facilities.iterrows():
        facility_id = facility.facility_id
        facility_providers = shift_table.loc[shift_table.facility_id == facility_id, "provider_id"].to_numpy()
        provider_lookup[facility_id] = facility_providers.tolist()
        room_lookup[facility_id] = [f"{facility_id}_R{r:03d}" for r in range(1, int(facility.room_capacity) + 1)]

    visit_id = 0
    for facility in facilities.itertuples(index=False):
        facility_id = facility.facility_id
        facility_weather = _build_weather_pattern(rng, timestamps, facility_id)
        base_load = facility.base_md + facility.base_np
        row_count_target = 0

        # Facility hour-level hidden intensity used to drive dirty event simulation.
        hour_of_day = timestamps.hour.to_numpy()
        dow = timestamps.dayofweek.to_numpy()
        hour_night = np.isin(hour_of_day, [0, 1, 2, 3, 4, 21, 22, 23]).astype(float)
        hour_evening = np.isin(hour_of_day, [16, 17, 18, 19, 20]).astype(float)
        weekend = (dow >= 5).astype(float)
        flu = pd.Series(facility_weather).isin(["snow", "storm"]).astype(float).to_numpy() * 0.25
        intensity = (
            12
            + 0.6 * base_load
            + 6 * hour_night
            + 5 * hour_evening
            + 4 * weekend
            + 3 * flu
            + rng.normal(0, 1.5, len(timestamps))
        )
        intensity = np.clip(intensity, 0.5, None)

        # Inject drift by day window and facility-specific behavior changes.
        half = len(timestamps) // 2
        if len(intensity) > half:
            intensity[half:] = intensity[half:] * (1.0 + 0.12 + 0.08 * rng.random())
        if facility.region in {"west", "south"}:
            intensity += 1.5

        for idx, ts in enumerate(timestamps):
            ts_arrival = ts
            arrivals = int(np.clip(rng.poisson(max(1.0, intensity[idx] / 6.0)), 0, 80))
            row_count_target += arrivals
            for _ in range(arrivals):
                room = rng.choice(room_lookup[facility_id])
                tri = rng.choice(triage, p=[0.05, 0.10, 0.25, 0.35, 0.18, 0.05, 0.02])
                complaint = rng.choice(chief)
                mode = rng.choice(arrival_mode, p=[0.40, 0.35, 0.15, 0.05, 0.05])
                prov = rng.choice(provider_lookup[facility_id]) if rng.random() > 0.15 else pd.NA
                acuity = {"critical": 1.9, "urgent": 1.6, "1": 1.8, "2": 1.6, "3": 1.2, "4": 1.0, "5": 0.8}.get(tri, 1.0)
                los_minutes = float(np.clip(rng.exponential(140) * (0.8 + 0.4 * acuity), 10, 720))
                visit_ts = ts_arrival + pd.Timedelta(minutes=float(rng.uniform(-20, 28)))
                visit_rows.append(
                    {
                        "visit_id": f"V_{visit_id:07d}",
                        "facility_id": facility_id,
                        "event_ts": visit_ts,
                        "event_type": "arrive",
                        "event_seq": 0,
                        "provider_id": prov,
                        "from_provider_id": pd.NA,
                        "to_provider_id": prov,
                        "room_id": room,
                        "triage_level": tri,
                        "chief_complaint": complaint,
                        "arrival_mode": mode,
                        "expected_length_of_stay_band": np.random.choice(["0-2", "2-4", "4-8", "8-12", "12-24", "24+"]),
                        "length_of_stay_min": los_minutes,
                        "acuity_score": float(acuity),
                        "duration_to_next_event_min": float(rng.exponential(12) + 1),
                        "weather_bucket": facility_weather[idx],
                    }
                )

                # Optional transition rows for dirty row-level scheduling logs.
                n_transition = int(rng.integers(0, 3))
                current_provider = prov
                current_time = visit_ts
                for event_seq in range(1, n_transition + 2):
                    to_provider = rng.choice(provider_lookup[facility_id])
                    if rng.random() < 0.20:
                        to_provider = pd.NA
                    event_type = rng.choice(event_types[1:], p=[0.44, 0.23, 0.26, 0.07])
                    current_time += pd.Timedelta(minutes=float(rng.exponential(18) + 3))
                    visit_rows.append(
                        {
                            "visit_id": f"V_{visit_id:07d}",
                            "facility_id": facility_id,
                            "event_ts": current_time,
                            "event_type": event_type,
                            "event_seq": int(event_seq),
                            "provider_id": current_provider,
                            "from_provider_id": current_provider,
                            "to_provider_id": to_provider,
                            "room_id": room if rng.random() < 0.92 else pd.NA,
                            "triage_level": tri,
                            "chief_complaint": complaint,
                            "arrival_mode": mode,
                            "expected_length_of_stay_band": np.random.choice(["0-2", "2-4", "4-8", "8-12", "12-24", "24+"]),
                            "length_of_stay_min": float(np.clip(los_minutes - (event_seq * 12), 5, 720)),
                            "acuity_score": float(acuity),
                            "duration_to_next_event_min": float(rng.exponential(24)),
                            "weather_bucket": facility_weather[idx],
                        }
                    )
                    current_provider = to_provider
                visit_id += 1

    visit_df = pd.DataFrame(visit_rows)
    visit_df = visit_df.sort_values(["facility_id", "visit_id", "event_ts"]).reset_index(drop=True)
    if "event_ts" in visit_df.columns:
        visit_df["event_ts"] = pd.to_datetime(visit_df["event_ts"])

    # Non-random missingness on key columns during high activity periods.
    for col, base_rate in [("provider_id", 0.07), ("room_id", 0.05), ("triage_level", 0.015), ("chief_complaint", 0.02)]:
        if col in visit_df.columns:
            high_load = visit_df["event_type"].eq("handoff").to_numpy()
            miss_mask = rng.random(len(visit_df)) < (base_rate + 0.22 * high_load.astype(float))
            visit_df.loc[miss_mask, col] = pd.NA

    # EHR-style outage blocks.
    for facility_id, g in visit_df.groupby("facility_id"):
        facility_indices = g.index.to_numpy()
        if len(facility_indices) < 2000:
            continue
        for _ in range(3):
            block_start = int(rng.integers(0, max(1, len(facility_indices) - 240)))
            block_end = min(len(facility_indices), block_start + int(rng.integers(80, 220)))
            hit = facility_indices[block_start:block_end]
            visit_df.loc[hit, ["arrival_mode", "from_provider_id", "to_provider_id"]] = pd.NA

    # Facility-hour capacity features.
    capacity_rows: list[dict] = []
    for facility in facilities.itertuples(index=False):
        facility_id = facility.facility_id
        fac_mask = visit_df["facility_id"] == facility_id
        visit_counts = visit_df.loc[fac_mask, "event_ts"].dt.floor("h").value_counts()
        for ts in timestamps:
            hourly_visits = float(visit_counts.get(ts, 0.0))
            h = int(ts.hour)
            dow = int(ts.dayofweek)
            flu = 1.0 + (ts.month in {11, 12, 1, 2}) * 0.35
            weekend = 1.0 if dow >= 5 else 0.0
            discharge_rate = float(np.clip(rng.normal(12 + 2 * weekend, 3), 2, 30))
            boarding = float(np.clip(rng.lognormal(1.9 + 0.12 * flu, 0.65), 0.4, 15.0))
            occupancy = np.clip(
                0.22 + 0.005 * facility.bed_capacity + 0.012 * hourly_visits + 0.08 * weekend + 0.04 * (h in {0, 1, 2, 3, 4, 21, 22, 23}),
                0.04,
                0.99,
            )
            rooms_occ = np.clip(0.16 + 0.004 * facility.room_capacity + 0.009 * hourly_visits + 0.05 * weekend, 0.04, 0.99)
            ambulances = float(np.clip(rng.poisson(1.2 + 0.45 * flu + 0.3 * weekend), 0, 20))
            capacity_rows.append(
                {
                    "facility_id": facility_id,
                    "timestamp_hour": ts,
                    "open_beds": int(facility.bed_capacity),
                    "open_rooms": int(facility.room_capacity),
                    "bed_occupancy_pct": float(np.round(occupancy, 4)),
                    "rooms_occupied_pct": float(np.round(rooms_occ, 4)),
                    "recent_discharge_rate": float(np.round(discharge_rate, 3)),
                    "inpatient_boarding_delay": float(np.round(boarding, 3)),
                    "flu_index": float(np.round(flu, 4)),
                    "ambulance_diversions": float(ambulances),
                    "is_holiday": float(1.0 if ts.strftime("%m-%d") in {"01-01", "12-25", "07-01"} else 0.0),
                    "arrival_pressure_proxy": float(np.round(hourly_visits, 3)),
                }
            )

    capacity_df = pd.DataFrame(capacity_rows)
    present["timestamp_hour"] = pd.to_datetime(present["timestamp_hour"])
    capacity_df["timestamp_hour"] = pd.to_datetime(capacity_df["timestamp_hour"])

    return visit_df, capacity_df, shift_table, present


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ER scheduling-style synthetic data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--facilities", type=int, default=4)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--min-shift-hours", type=int, default=4)
    parser.add_argument("--out-prefix", default="data/er_iter2")
    parser.add_argument("--rows", type=int, default=None, help="Optional approximate target for max rows across visits")
    args = parser.parse_args()

    cfg = Iter2GeneratorConfig(
        seed=args.seed,
        n_facilities=args.facilities,
        n_days=args.days,
        output_prefix=args.out_prefix,
        min_shift_hours=max(4, int(args.min_shift_hours)),
    )
    visits, capacity, shifts, present = generate_er_iteration2_data(cfg)
    if args.rows and len(visits) > args.rows:
        visits = visits.sample(args.rows, random_state=args.seed).sort_values(["facility_id", "event_ts"]).reset_index(drop=True)

    out_prefix = Path(cfg.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_if_dataframe(visits, str(out_prefix.with_suffix(".visits.parquet")))
    write_parquet_if_dataframe(capacity, str(out_prefix.with_suffix(".capacity.parquet")))
    write_parquet_if_dataframe(shifts, str(out_prefix.with_suffix(".shifts.parquet")))
    write_parquet_if_dataframe(present, str(out_prefix.with_suffix(".present_shift_hours.parquet")))
    safe_json_dump(
        {
            "seed": cfg.seed,
            "facilities": int(cfg.n_facilities),
            "days": int(cfg.n_days),
            "min_shift_hours": int(cfg.min_shift_hours),
            "visits_rows": int(len(visits)),
            "capacity_rows": int(len(capacity)),
            "shift_rows": int(len(shifts)),
            "present_rows": int(len(present)),
            "mean_unavailable_hours": float(shifts["unavailable_hours"].mean()) if not shifts.empty else 0.0,
        },
        str(out_prefix.with_suffix(".meta.json")),
    )

    print("=== visit sample (top rows) ===")
    print(visits.head(5).to_string(index=False))
    print("=== capacity sample (top rows) ===")
    print(capacity.head(5).to_string(index=False))
    print("=== shift sample (top rows) ===")
    print(shifts.head(5).to_string(index=False))

    print(f"Wrote visit table: {out_prefix.with_suffix('.visits.parquet')}")
    print(f"Wrote capacity table: {out_prefix.with_suffix('.capacity.parquet')}")
    print(f"Wrote provider shifts: {out_prefix.with_suffix('.shifts.parquet')}")
    print(f"Wrote shift-hour availability: {out_prefix.with_suffix('.present_shift_hours.parquet')}")


if __name__ == "__main__":
    main()
