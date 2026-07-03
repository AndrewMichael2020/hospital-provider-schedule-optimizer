from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from tabfm_healthcare_eval.config import GeneratorConfig
from tabfm_healthcare_eval.utils import set_deterministic_seed, write_parquet_if_dataframe, safe_json_dump


def _make_facilities(rng: np.random.Generator, n: int) -> pd.DataFrame:
    facilities = []
    for facility_idx in range(n):
        base_docs = rng.integers(8, 28)
        base_nps = rng.integers(6, 22)
        bed_capacity = rng.integers(35, 130)
        ed_type = rng.choice(["urban", "community", "academic", "rural"])
        region = rng.choice(["west", "north", "south", "east", "central"])
        facilities.append(
            {
                "facility_id": f"FAC_{facility_idx:03d}",
                "facility_ed_type": ed_type,
                "region": region,
                "bed_capacity": int(bed_capacity),
                "baseline_docs": int(base_docs),
                "baseline_nps": int(base_nps),
            }
        )
    return pd.DataFrame(facilities)


def _seasonal_feature(steps: np.ndarray, period: int = 24 * 365) -> np.ndarray:
    return 0.5 + 0.5 * np.sin(2 * np.pi * steps / period)


def generate_er_stress_data(config: GeneratorConfig) -> pd.DataFrame:
    set_deterministic_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    facilities = _make_facilities(rng, config.n_facilities)

    start = pd.Timestamp(pd.to_datetime(config.start_datetime))
    if not isinstance(start, pd.Timestamp):
        start = pd.Timestamp(start)
    per_fac = np.array([config.n_rows // config.n_facilities] * config.n_facilities)
    for i in range(config.n_rows - per_fac.sum()):
        per_fac[i % config.n_facilities] += 1

    rows = []
    chief_groups = [
        "cardiac",
        "respiratory",
        "trauma",
        "neurologic",
        "gastro",
        "obstetric",
        "pediatric",
        "orthopedic",
        "infectious",
        "toxicology",
        "mental-health",
        "minor",
    ]
    arrival_modes = ["ambulance", "walk-in", "transfer", "tele-triage", "referral"]
    los_bands = ["0-1h", "1-2h", "2-4h", "4-8h", "8-12h", "12h+"]
    triage_levels = ["1", "2", "3", "4", "5", "urgent", "critical"]
    weather_buckets = ["cold", "cool", "mild", "hot", "storm", "snow", "rain"]
    holidays = {("2024-01-01"), ("2024-07-04"), ("2024-12-25")}

    drift_start_month = config.drift_start_month

    for facility_row in facilities.itertuples(index=False):
        n_rows = int(per_fac[int(facility_row.facility_id.split("_")[-1])])
        base_time = start + pd.Timedelta(days=int(rng.integers(0, 7)), hours=int(rng.integers(0, 24)))
        timestamps = pd.date_range(base_time, periods=n_rows, freq=f"{config.step_hours}h")
        steps = np.arange(n_rows)

        # Facility effects
        fac_docs = int(facility_row.baseline_docs)
        fac_nps = int(facility_row.baseline_nps)

        provider_count = max(config.provider_pool_per_facility, 50)
        room_count = max(config.room_pool_per_facility, 40)
        provider_ids = [
            f"{facility_row.facility_id}_P{provider_i:03d}_{'NP' if provider_i % 3 == 0 else 'MD'}"
            for provider_i in range(provider_count)
        ]
        room_ids = [f"{facility_row.facility_id}_R{room_i:03d}" for room_i in range(room_count)]

        dt = pd.DatetimeIndex(timestamps)
        month = dt.month.to_numpy()
        doy = dt.dayofyear.to_numpy()
        hour = dt.hour.to_numpy()
        dow = dt.dayofweek.to_numpy()
        is_weekend = (dow >= 5).astype(float)
        is_holiday = np.array([1.0 if ts.strftime("%Y-%m-%d") in holidays else 0.0 for ts in dt])
        flu_index = 0.55 + 0.35 * _seasonal_feature(doy)
        flu_bump = np.isin(month, [1, 2, 10, 11, 12]).astype(float) * 0.35
        flu_index = np.clip(flu_index + flu_bump, 0.0, 1.5)

        flu_peak = flu_index**2
        night = np.isin(hour, [22, 23, 0, 1, 2]).astype(float)
        evening = np.isin(hour, [17, 18, 19, 20]).astype(float)
        weekend = is_weekend
        drift = (month >= drift_start_month).astype(float) * 0.6

        provider_draw = rng.choice(provider_ids, size=n_rows)
        room_draw = rng.choice(room_ids, size=n_rows)
        provider_type = pd.Series(provider_draw).str.contains("NP").astype(int).to_numpy()
        room_load = rng.integers(1, 4, size=n_rows)

        complaint_idx = rng.choice(chief_groups, size=n_rows)
        # After drift, shift complaint mix.
        if drift.mean() > 0:
            flip = rng.random(n_rows) < (0.08 * drift)
            complaint_idx = np.where(flip, rng.choice(["respiratory", "infectious", "mental-health"], size=n_rows), complaint_idx)

        arrival_mode = rng.choice(arrival_modes, size=n_rows)
        triage = rng.choice(triage_levels, size=n_rows, p=[0.03, 0.08, 0.30, 0.35, 0.18, 0.04, 0.02])
        los = rng.choice(los_bands, size=n_rows, p=[0.05, 0.12, 0.20, 0.35, 0.20, 0.08])

        queue_base = rng.normal(40 + 15 * flu_peak, 18 + 6 * drift, n_rows)
        current_queue = np.clip(np.round(queue_base + 6 * room_load + 3 * provider_type + 10 * weekend + 12 * evening), 0, None)
        discharge_rate = np.clip(rng.normal(12 + 2 * drift, 3 + drift, n_rows), 1, 40)
        inpatient_boarding_delay = np.clip(rng.lognormal(2.2 + 0.2 * flu_peak + 0.25 * drift, 0.7, n_rows), 0.5, 15.0)
        bed_occupancy_pct = np.clip(
            0.45 + 0.35 * flu_peak + 0.1 * weekend + 0.1 * (doy % 7 / 7) + 0.05 * drift + rng.normal(0, 0.05, n_rows),
            0.05,
            0.99,
        )
        base_docs_on_duty = np.clip(
            fac_docs + 4 * night + 3 * evening + 2 * (dow == 0).astype(int) + rng.normal(0, 2, n_rows),
            4,
            60,
        )
        base_nps_on_duty = np.clip(
            fac_nps + 1.5 * night + 1.0 * weekend + rng.normal(0, 1.5, n_rows),
            2,
            35,
        )

        acuity = pd.Series(triage).map(
            {"critical": 1.9, "urgent": 1.4, "1": 1.8, "2": 1.5, "3": 1.1, "4": 0.9, "5": 0.7}
        ).fillna(1.0).to_numpy()
        complaint_weight = pd.Series(complaint_idx).map(
            {
                "respiratory": 1.3,
                "infectious": 1.2,
                "cardiac": 1.1,
                "trauma": 1.1,
                "mental-health": 1.0,
                "obstetric": 1.0,
                "neurologic": 1.05,
                "other": 1.0,
            }
        ).fillna(1.0).to_numpy()

        arrival_mode_factor = pd.Series(arrival_mode).map(
            {"ambulance": 1.25, "transfer": 1.1, "walk-in": 0.95, "tele-triage": 0.8, "referral": 1.05}
        ).to_numpy()
        los_factor = pd.Series(los).map({"12h+": 1.3, "8-12h": 1.15, "4-8h": 1.0, "2-4h": 0.95, "1-2h": 0.85, "0-1h": 0.7}).to_numpy()

        temp = 15 + 8 * np.sin(2 * np.pi * ((dt.hour + 4) / 24))
        weather_idx = np.array([weather_buckets[i] for i in rng.integers(0, len(weather_buckets), n_rows)])
        weather_penalty = pd.Series(weather_idx).map(
            {"snow": 1.35, "storm": 1.5, "rain": 1.2, "cold": 1.1, "cool": 1.0, "mild": 0.95, "hot": 1.05}
        ).to_numpy()

        ambulance_diversions = np.clip(
            rng.poisson(0.8 + 2.2 * flu_peak + 0.35 * (weather_idx == "storm").astype(float) + 0.6 * night + drift * 0.8).astype(float),
            0,
            20,
        )

        intensity = (
            16
            + 22 * flu_peak
            + 12 * night
            + 8 * evening
            + 6 * weekend
            + 4 * is_holiday
            + 10 * drift
            + 0.05 * current_queue
            + 0.6 * inpatient_boarding_delay
            + 0.03 * current_queue * bed_occupancy_pct
            + 2 * los_factor
        ) * arrival_mode_factor * complaint_weight * weather_penalty * acuity * 0.8

        arrivals = rng.negative_binomial(2.0 + 0.5 * drift, 2.0 / (2.0 + np.clip(intensity, 0.2, None)))
        arrivals = np.clip(arrivals.astype(float), 0, 300.0)
        expected_arrivals_next_hour = np.round(arrivals + rng.normal(0, 1.0, n_rows), 1)

        capacity_base = 0.65 * base_docs_on_duty + 0.35 * base_nps_on_duty
        demand_pressure = (
            0.95 * expected_arrivals_next_hour
            + 0.10 * current_queue
            + 2.0 * ambulance_diversions
            + 1.2 * inpatient_boarding_delay
            + 0.6 * (bed_occupancy_pct * 100)
            + 1.7 * acuity
            + 0.4 * flu_peak * 10
        )

        # Rare surges in high-risk windows
        n_events = max(1, int(config.surge_rate * n_rows))
        surge_starts = rng.integers(0, max(1, n_rows - 24), size=n_events)
        for surge_start in surge_starts:
            end = min(n_rows, surge_start + 24 + rng.integers(4, 16))
            expected_arrivals_next_hour[surge_start:end] = np.clip(
                expected_arrivals_next_hour[surge_start:end] * config.surge_magnitude, 0, 999
            )
            demand_pressure[surge_start:end] *= 1.8
            ambulance_diversions[surge_start:end] += rng.integers(6, 14, size=(end - surge_start))
        shortage_margin = capacity_base - (demand_pressure / 35.0 + 2 * drift)
        shortage_logits = -(shortage_margin)
        shortage_prob = 1.0 / (1.0 + np.exp(-(shortage_logits - shortage_logits.mean()) / (shortage_logits.std() + 1e-3)))
        shortage_risk_next_shift = rng.binomial(1, np.clip(shortage_prob, 0.01, 0.99), n_rows)

        weather_hotspot = (weather_idx == "storm") | (weather_idx == "snow")

        facility_df = pd.DataFrame(
            {
                "facility_id": facility_row.facility_id,
                "timestamp": dt,
                "hour_of_day": hour,
                "day_of_week": dow,
                "is_weekend": is_weekend.astype(int),
                "is_holiday": is_holiday,
                "month": month,
                "flu_index": np.round(flu_index, 4),
                "weather_bucket": weather_idx,
                "ambulance_diversions": np.round(ambulance_diversions, 2),
                "triage_level": triage,
                "chief_complaint_group": complaint_idx,
                "arrival_mode": arrival_mode,
                "length_of_stay_band": los,
                "facility_ed_type": facility_row.facility_ed_type,
                "region": facility_row.region,
                "bed_capacity": facility_row.bed_capacity,
                "current_queue": np.round(current_queue, 2),
                "recent_discharge_rate": np.round(discharge_rate, 2),
                "inpatient_boarding_delay": np.round(inpatient_boarding_delay, 3),
                "bed_occupancy_pct": np.round(bed_occupancy_pct, 4),
                "current_docs_on_duty": np.round(base_docs_on_duty, 2),
                "current_nps_on_duty": np.round(base_nps_on_duty, 2),
                "provider_id": provider_draw,
                "room_id": room_draw,
                "case_mix_complexity": np.round(np.clip(0.8 + 0.2 * complaint_weight + 0.4 * acuity + 0.2 * los_factor + 0.2 * drift + rng.normal(0, 0.2, n_rows), 0, 5), 3),
                "expected_arrivals_next_hour": np.round(expected_arrivals_next_hour, 4),
                "shortage_risk_next_shift": shortage_risk_next_shift.astype(int),
                "shortage_risk_probability": np.round(shortage_prob, 6),
                "storm_or_snow_hour": weather_hotspot.astype(int),
            }
        )

        # Not-at-random missingness in high load states.
        high_load = (expected_arrivals_next_hour > np.percentile(expected_arrivals_next_hour, 90)) | (
            current_queue > np.percentile(current_queue, 90)
        )
        for col in ["ambulance_diversions", "arrival_mode", "recent_discharge_rate", "provider_id", "room_id"]:
            if col in facility_df.columns:
                miss_mask = (rng.random(n_rows) < (0.12 + 0.45 * high_load.astype(float))).astype(bool)
                facility_df.loc[miss_mask, col] = pd.NA

        # A second high-load MAR-like pattern for current_queue and discharge rate.
        mar_mask = rng.random(n_rows) < (0.08 + 0.20 * high_load.astype(float))
        facility_df.loc[mar_mask, "current_queue"] = pd.NA
        facility_df.loc[mar_mask, "recent_discharge_rate"] = pd.NA

        # EHR-offline-like blocks.
        n_blocks = max(1, n_rows // 25000)
        for _ in range(n_blocks):
            block_start = int(rng.integers(0, max(1, n_rows - 36)))
            block_end = min(n_rows, block_start + rng.integers(12, 40))
            for col in [
                "triage_level",
                "chief_complaint_group",
                "arrival_mode",
                "facility_ed_type",
            ]:
                facility_df.loc[facility_df.index[block_start:block_end], col] = pd.NA

        rows.append(facility_df)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["facility_id", "timestamp"]).reset_index(drop=True)

    # Enforce explicit target monotonicity by facility-time: no future leakage in generation process.
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ER stress dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--facilities", type=int, default=3)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--step-hours", type=int, default=1)
    parser.add_argument("--surge-rate", type=float, default=0.025)
    parser.add_argument("--surge-magnitude", type=float, default=2.8)
    args = parser.parse_args()
    cfg = GeneratorConfig(
        seed=args.seed,
        n_rows=args.rows,
        n_facilities=args.facilities,
        step_hours=args.step_hours,
        output_path=args.out,
        surge_rate=args.surge_rate,
        surge_magnitude=args.surge_magnitude,
    )
    df = generate_er_stress_data(cfg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_if_dataframe(df, str(out_path))
    safe_json_dump({"config": asdict(cfg), "n_rows": len(df), "n_facilities": int(df["facility_id"].nunique())}, str(out_path.with_suffix(".meta.json")))
    print(f"Wrote dataset to: {out_path}")


if __name__ == "__main__":
    import argparse

    main()
