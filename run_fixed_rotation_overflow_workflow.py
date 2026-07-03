from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ["md", "np", "rn", "hcw"]
ROLE_LABELS = {"md": "MD", "np": "NP", "rn": "RN", "hcw": "HCW"}
ROLE_WEIGHTS = {"md": 2.0, "np": 1.6, "rn": 1.3, "hcw": 1.0}


@dataclass(frozen=True)
class RotationLine:
    rotation_line_id: str
    department_id: str
    role: str
    start_hour: int
    duration_hours: int
    md_equiv: float = 0.0
    np_equiv: float = 0.0
    rn_equiv: float = 0.0
    hcw_equiv: float = 0.0

    def capacity(self, role: str) -> float:
        return float(getattr(self, f"{role}_equiv"))


CORE_LINES = [
    RotationLine("MD_TRAUMA_DAY", "ER_TRAUMA", "MD", 7, 10, md_equiv=1.0),
    RotationLine("MD_FAST_TRACK", "ER_FAST_TRACK", "MD", 9, 10, md_equiv=1.0),
    RotationLine("NP_SWING", "ER_FAST_TRACK", "NP", 13, 10, np_equiv=1.0),
    RotationLine("MD_NIGHT", "ER_MAIN", "MD", 21, 10, md_equiv=1.0),
    RotationLine("RN_DAY_A", "ER_MAIN", "RN", 7, 12, rn_equiv=1.0),
    RotationLine("RN_DAY_B", "ER_MAIN", "RN", 9, 12, rn_equiv=1.0),
    RotationLine("RN_NIGHT", "ER_MAIN", "RN", 19, 12, rn_equiv=1.0),
    RotationLine("HCW_DAY", "ER_MAIN", "HCW", 7, 10, hcw_equiv=1.0),
    RotationLine("HCW_EVE", "ER_MAIN", "HCW", 13, 10, hcw_equiv=1.0),
    RotationLine("HCW_NIGHT", "ER_MAIN", "HCW", 21, 10, hcw_equiv=1.0),
]

OVERFLOW_LINES = [
    RotationLine("ON_CALL_MD", "ER_MAIN", "MD", 10, 10, md_equiv=1.0),
    RotationLine("ON_CALL_MD_EVE", "ER_MAIN", "MD", 14, 10, md_equiv=1.0),
    RotationLine("ON_CALL_NP_SWING", "ER_FAST_TRACK", "NP", 12, 10, np_equiv=1.0),
    RotationLine("ON_CALL_NP_EVE", "ER_FAST_TRACK", "NP", 14, 10, np_equiv=1.0),
    RotationLine("HOSPITALIST_OVERFLOW", "ER_MAIN", "HOSPITALIST_OVERFLOW", 14, 10, md_equiv=0.75),
    RotationLine("ON_CALL_RN_DAY", "ER_MAIN", "RN", 8, 10, rn_equiv=1.0),
    RotationLine("ON_CALL_RN_EVE", "ER_MAIN", "RN", 14, 10, rn_equiv=1.0),
    RotationLine("ON_CALL_RN_NIGHT", "ER_MAIN", "RN", 20, 10, rn_equiv=1.0),
    RotationLine("ON_CALL_HCW_DAY", "ER_MAIN", "HCW", 8, 10, hcw_equiv=1.0),
    RotationLine("ON_CALL_HCW_EVE", "ER_MAIN", "HCW", 14, 10, hcw_equiv=1.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hospital Schedule Optimizer fixed-rotation overflow workflow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--facilities", type=int, default=4)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--overflow-budget-share-md", type=float, default=0.10)
    parser.add_argument("--overflow-budget-share-np", type=float, default=0.10)
    parser.add_argument("--overflow-budget-share-rn", type=float, default=0.10)
    parser.add_argument("--overflow-budget-share-hcw", type=float, default=0.10)
    parser.add_argument("--overflow-budget-share", type=float, default=None, help="Backward-compatible shortcut applied to all roles when provided.")
    parser.add_argument("--demand-pressure-multiplier", type=float, default=1.0)
    parser.add_argument("--min-overflow-covered-gap-hours", type=float, default=6.0)
    parser.add_argument("--out", default="outputs/fixed_rotation_overflow_v1")
    parser.add_argument("--overflow-shift-hours", type=int, default=10)
    parser.add_argument("--print-top-n", type=int, default=5)
    return parser.parse_args()


def _budget_shares(args: argparse.Namespace) -> dict[str, float]:
    if args.overflow_budget_share is not None:
        return {role: float(args.overflow_budget_share) for role in ROLES}
    return {
        "md": float(args.overflow_budget_share_md),
        "np": float(args.overflow_budget_share_np),
        "rn": float(args.overflow_budget_share_rn),
        "hcw": float(args.overflow_budget_share_hcw),
    }


def _date_range(days: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp("2025-01-01 00:00:00"), periods=days * 24, freq="h")


def _json_flags(heatwave: bool, flu_surge: bool, ambulance_surge: bool) -> str:
    return json.dumps({"heatwave": bool(heatwave), "flu_surge": bool(flu_surge), "ambulance_surge": bool(ambulance_surge)}, sort_keys=True)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _print_top(label: str, path: Path, n: int) -> None:
    if n <= 0:
        return
    df = _load_table(path)
    print(f"\n=== {label}: {path} ===")
    print(f"shape={df.shape}")
    print(f"columns={list(df.columns)}")
    print(df.head(n).to_string(index=False))


def _capacity_cols(line: RotationLine) -> dict[str, float]:
    return {f"activation_capacity_{role}_equiv": line.capacity(role) for role in ROLES}


def _shift_row(shift_id: str, provider_id: str, facility_id: str, start: pd.Timestamp, end: pd.Timestamp, line: RotationLine, week: int, core: bool) -> dict:
    row = {
        "shift_id": shift_id,
        "provider_id": provider_id,
        "department_id": line.department_id,
        "facility_id": facility_id,
        "shift_start": start,
        "shift_end": end,
        "actual_hours_worked": float(line.duration_hours) if core else 0.0,
        "rotation_line_id": line.rotation_line_id,
        "rotation_week": week,
        "role": line.role,
        "is_core_rotation": bool(core),
        "is_overflow_candidate": not core,
        "activation_window_start": pd.NaT if core else start,
        "activation_window_end": pd.NaT if core else end,
        "activation_cost": 0.0 if core else (1.0 if line.role in {"MD", "RN", "HCW"} else 1.4),
    }
    row.update(_capacity_cols(line))
    return row


def _role_pressure_blocks(rng: np.random.Generator, days: int, multiplier: float) -> dict[str, set[pd.Timestamp]]:
    blocks = {role: set() for role in ROLES}
    for day in range(days):
        date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day)
        weekend = int(date.dayofweek) >= 5
        heatwave = 10 <= date.day <= 13
        flu_surge = date.day >= 20
        role_prob = {
            "md": min(0.42, (0.16 + 0.06 * weekend + 0.06 * heatwave + 0.06 * flu_surge) * multiplier),
            "np": min(0.30, (0.10 + 0.04 * weekend + 0.04 * heatwave + 0.04 * flu_surge) * multiplier),
            "rn": min(0.52, (0.24 + 0.08 * weekend + 0.08 * heatwave + 0.09 * flu_surge) * multiplier),
            "hcw": min(0.46, (0.20 + 0.07 * weekend + 0.08 * heatwave + 0.07 * flu_surge) * multiplier),
        }
        for role, prob in role_prob.items():
            if rng.random() >= prob:
                continue
            if role == "md":
                start_hour = int(rng.choice([10, 12, 14, 16, 18], p=[0.08, 0.18, 0.34, 0.30, 0.10]))
                duration = int(rng.choice([4, 5, 6, 7, 8], p=[0.10, 0.20, 0.35, 0.25, 0.10]))
            elif role == "np":
                start_hour = int(rng.choice([12, 13, 14, 15, 16], p=[0.12, 0.22, 0.30, 0.24, 0.12]))
                duration = int(rng.choice([4, 5, 6, 7], p=[0.20, 0.35, 0.30, 0.15]))
            elif role == "rn":
                start_hour = int(rng.choice([8, 10, 12, 14, 16, 18, 20], p=[0.08, 0.10, 0.16, 0.22, 0.20, 0.14, 0.10]))
                duration = int(rng.choice([5, 6, 7, 8, 9], p=[0.10, 0.22, 0.30, 0.28, 0.10]))
            else:
                start_hour = int(rng.choice([8, 10, 12, 14, 16, 18], p=[0.10, 0.14, 0.22, 0.24, 0.20, 0.10]))
                duration = int(rng.choice([4, 5, 6, 7, 8], p=[0.12, 0.24, 0.32, 0.22, 0.10]))
            for h in range(start_hour, min(24, start_hour + duration)):
                blocks[role].add(date + pd.Timedelta(hours=h))
    return blocks


def _core_capacity_for_hour(hour: int) -> dict[str, float]:
    return {
        "md": float((7 <= hour < 17) + (9 <= hour < 19) + (hour >= 21 or hour < 7)),
        "np": float(13 <= hour < 23),
        "rn": float((7 <= hour < 19) + (9 <= hour < 21) + (hour >= 19 or hour < 7)),
        "hcw": float((7 <= hour < 17) + (13 <= hour < 23) + (hour >= 21 or hour < 7)),
    }


def generate_canonical_mock(seed: int, facilities: int, days: int, budget_shares: dict[str, float], out_dir: Path, overflow_shift_hours: int = 10, demand_pressure_multiplier: float = 1.0) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    hours = _date_range(days)
    facility_ids = [f"FAC_{i:03d}" for i in range(facilities)]
    evh_rows: list[dict] = []
    rav_rows: list[dict] = []
    uss_rows: list[dict] = []
    ass_rows: list[dict] = []
    psh_rows: list[dict] = []
    manual_rows: list[dict] = []
    provider_i = 0
    shift_i = 0
    activation_i = 0

    for facility_idx, facility_id in enumerate(facility_ids):
        base_beds = int(rng.integers(45, 90))
        base_rooms = int(rng.integers(25, 55))
        facility_load = 1.0 + 0.08 * facility_idx + rng.normal(0, 0.03)
        for line in [l for l in CORE_LINES if l.start_hour + l.duration_hours > 24]:
            start = pd.Timestamp("2024-12-31") + pd.Timedelta(hours=line.start_hour)
            end = start + pd.Timedelta(hours=line.duration_hours)
            psh_rows.append(_shift_row(f"SH_{shift_i:07d}", f"PR_{provider_i:05d}", facility_id, start, end, line, -1, True))
            provider_i += 1
            shift_i += 1
        for day in range(days):
            date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day)
            week = int(day // 7)
            for line in CORE_LINES:
                start = date + pd.Timedelta(hours=line.start_hour)
                end = start + pd.Timedelta(hours=line.duration_hours)
                psh_rows.append(_shift_row(f"SH_{shift_i:07d}", f"PR_{provider_i:05d}", facility_id, start, end, line, week, True))
                provider_i += 1
                shift_i += 1
            for line in OVERFLOW_LINES:
                start = date + pd.Timedelta(hours=line.start_hour)
                end = start + pd.Timedelta(hours=max(4, int(overflow_shift_hours)))
                provider_id = f"PR_OF_{facility_id}_{day:03d}_{line.rotation_line_id}"
                psh_rows.append(_shift_row(f"SH_{shift_i:07d}", provider_id, facility_id, start, end, line, week, False))
                shift_i += 1

        pressure_blocks = _role_pressure_blocks(rng, days, demand_pressure_multiplier)
        for ts in hours:
            hour = int(ts.hour)
            weekend = int(ts.dayofweek) >= 5
            heatwave = 10 <= ts.day <= 13
            flu_surge = ts.day >= 20
            ambulance_surge = hour in {16, 17, 18, 19, 20} and rng.random() < 0.28
            night = hour in {0, 1, 2, 3, 21, 22, 23}
            evening = hour in {15, 16, 17, 18, 19, 20}
            base_lambda = (4.4 + 1.1 * evening + 0.7 * night + 0.6 * weekend + 1.0 * heatwave + 0.9 * flu_surge + 1.2 * ambulance_surge) * facility_load
            patient_volume = int(max(0, rng.poisson(base_lambda)))
            waiting_room_count = int(max(0, rng.poisson(max(1.0, patient_volume * (0.65 + 0.12 * evening)))))
            wait_risk_proxy = float(max(0.0, waiting_room_count - 7) * 0.35 + max(0.0, patient_volume - 6) * 0.25)
            flags = _json_flags(heatwave, flu_surge, ambulance_surge)
            core = _core_capacity_for_hour(hour)
            pressure = 0.02 + 0.03 * evening + 0.02 * weekend + 0.04 * heatwave + 0.04 * flu_surge + 0.03 * ambulance_surge
            pressure = float(np.clip(pressure + np.clip(rng.normal(0.0, 0.025), -0.03, 0.05), 0.0, 0.12))
            surge_probability = float(np.clip((pressure * 0.35 + max(0.0, wait_risk_proxy - 3.5) * 0.03) * demand_pressure_multiplier, 0.0, 0.06))
            required = {}
            for role in ROLES:
                block_extra = int(ts in pressure_blocks[role]) if core[role] > 0 else 0
                random_extra = int(rng.random() < surge_probability * {"md": 1.0, "np": 0.6, "rn": 1.2, "hcw": 1.0}[role]) if core[role] > 0 else 0
                edge_gap = int(core[role] <= 0 and rng.random() < {"md": 0.025, "np": 0.015, "rn": 0.018, "hcw": 0.018}[role])
                required[role] = int(max(0, core[role] + block_extra + random_extra + edge_gap))
            rav_rows.append(
                {
                    "date": ts.date().isoformat(),
                    "timestamp_hour": ts,
                    "facility_id": facility_id,
                    "department_id": "ER_MAIN",
                    "active_beds": base_beds,
                    "operating_rooms_available": max(1, int(base_rooms * (0.82 - 0.05 * weekend))),
                    "on_call_pool_size": 6,
                    "waiting_room_count": waiting_room_count,
                    "wait_risk_proxy": wait_risk_proxy,
                    "environmental_flags": flags,
                }
            )
            uss_rows.append({"hour_block": ts, "facility_id": facility_id, "department_id": "ER_MAIN", "environmental_flags": flags, "patient_volume": np.nan, "required_provider_count": np.nan})
            ass = {
                "timestamp_hour": ts,
                "facility_id": facility_id,
                "department_id": "ER_MAIN",
                "actual_patient_volume": patient_volume,
                "waiting_room_count": waiting_room_count,
                "wait_risk_proxy": wait_risk_proxy,
                "environmental_flags": flags,
            }
            ass.update({f"actual_required_{role}": required[role] for role in ROLES})
            ass_rows.append(ass)
            for j in range(patient_volume):
                evh_rows.append(
                    {
                        "timestamp": ts + pd.Timedelta(minutes=float(rng.uniform(0, 59))),
                        "facility_id": facility_id,
                        "department_id": "ER_MAIN",
                        "visit_id": f"V_{facility_id}_{ts.strftime('%Y%m%d%H')}_{j:03d}",
                        "acuity_score": int(rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.13, 0.35, 0.32, 0.15])),
                        "disposition": str(rng.choice(["Admitted", "Discharged", "Transferred"], p=[0.28, 0.65, 0.07])),
                        "environmental_flags": flags,
                    }
                )
            trigger = waiting_room_count >= 10 or wait_risk_proxy >= 2.0 or ambulance_surge or heatwave or flu_surge
            if trigger and hour in {12, 16, 18, 20} and rng.random() < 0.62:
                core_now = _core_capacity_for_hour(hour)
                gaps = {role: max(0.0, required[role] - core_now[role]) for role in ROLES}
                role_choices = [role for role, gap in gaps.items() if gap > 0] or [str(rng.choice(ROLES, p=[0.25, 0.15, 0.38, 0.22]))]
                role = str(rng.choice(role_choices))
                offset = int(rng.choice([-2, 0, 2, 4], p=[0.12, 0.60, 0.20, 0.08]))
                miss_reason = "early_activation" if offset < 0 else ("late_activation" if offset > 0 else "")
                if rng.random() < 0.08:
                    miss_reason = "unavailable_on_call_provider"
                if rng.random() < 0.05:
                    miss_reason = "wrong_department_or_facility"
                start = ts.floor("h") + pd.Timedelta(hours=offset)
                end = start + pd.Timedelta(hours=max(4, int(overflow_shift_hours)))
                hit = bool(miss_reason == "" and (gaps.get(role, 0.0) > 0 or wait_risk_proxy >= 1.5))
                if not hit and miss_reason == "":
                    miss_reason = "trigger_without_true_surge"
                row = {
                    "activation_id": f"ACT_{activation_i:07d}",
                    "shift_id": f"MANUAL_{activation_i:07d}",
                    "provider_id": f"PR_MANUAL_{facility_id}_{activation_i:05d}",
                    "department_id": "ER_MAIN",
                    "facility_id": facility_id,
                    "manual_activation_start": start,
                    "manual_activation_end": end,
                    "manual_activation_hit": hit,
                    "manual_activation_miss_reason": "" if hit else miss_reason,
                    "activation_reason": "threshold_wait_or_surge",
                    "role": ROLE_LABELS[role],
                }
                row.update({f"activation_capacity_{r}_equiv": 1.0 if r == role else 0.0 for r in ROLES})
                manual_rows.append(row)
                activation_i += 1

    evh = pd.DataFrame(evh_rows).sort_values(["facility_id", "timestamp"]).reset_index(drop=True)
    psh = pd.DataFrame(psh_rows).sort_values(["facility_id", "shift_start", "rotation_line_id"]).reset_index(drop=True)
    rav = pd.DataFrame(rav_rows).sort_values(["facility_id", "timestamp_hour"]).reset_index(drop=True)
    uss = pd.DataFrame(uss_rows).sort_values(["facility_id", "hour_block"]).reset_index(drop=True)
    ass = pd.DataFrame(ass_rows).sort_values(["facility_id", "timestamp_hour"]).reset_index(drop=True)
    manual = pd.DataFrame(manual_rows).sort_values(["facility_id", "manual_activation_start"]).reset_index(drop=True)
    srr = pd.DataFrame([{"department_id": dept, "rollout_phase": "SHADOW_MODE", "max_allowed_coverage_risk_hours": 0, "authority_level": "advisory"} for dept in ["ER_MAIN", "ER_FAST_TRACK", "ER_TRAUMA"]])
    paths = {
        "EVH": out_dir / "er_visits_historical.parquet",
        "PSH": out_dir / "provider_shifts_historical.parquet",
        "RAV": out_dir / "resource_availability.parquet",
        "USS": out_dir / "upcoming_schedule_skeleton.parquet",
        "ASS": out_dir / "actual_system_state.parquet",
        "SRR": out_dir / "metadata.system_rollout_registry.parquet",
        "manual": out_dir / "manual_overflow_activation_log.parquet",
    }
    for key, df in [("EVH", evh), ("PSH", psh), ("RAV", rav), ("USS", uss), ("ASS", ass), ("SRR", srr), ("manual", manual)]:
        _write_table(df, paths[key])
    core = _core_hours_by_role(psh)
    meta = {"seed": seed, "facilities": facilities, "days": days, "overflow_budget_shares": budget_shares, "core_monthly_provider_hours_by_role": core, "overflow_budget_hours_by_role": {r: core[r] * budget_shares[r] for r in ROLES}, "overflow_shift_hours": int(overflow_shift_hours), "demand_pressure_multiplier": float(demand_pressure_multiplier), "rows": {k: int(len(_load_table(v))) for k, v in paths.items()}}
    meta_path = out_dir / "fixed_rotation_overflow.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["meta"] = meta_path
    return paths


def _core_hours_by_role(psh: pd.DataFrame) -> dict[str, float]:
    core = psh[psh["is_core_rotation"] == True].copy()
    hours = (pd.to_datetime(core["shift_end"]) - pd.to_datetime(core["shift_start"])).dt.total_seconds() / 3600.0
    return {role: float((hours * pd.to_numeric(core[f"activation_capacity_{role}_equiv"], errors="coerce").fillna(0.0)).sum()) for role in ROLES}


def build_ptd_scenarios(paths: dict[str, Path], out_dir: Path) -> Path:
    ass = pd.read_parquet(paths["ASS"])
    rows = []
    for scenario, mult, cushion in [("tabfm_guarded", 0.98, 0.4), ("statistical_ensemble", 1.00, 0.1), ("conservative_peak", 1.18, 1.0)]:
        for row in ass.itertuples(index=False):
            pred_volume = max(0.0, float(row.actual_patient_volume) * mult + cushion)
            out = {
                "scenario_name": scenario,
                "hour_block": row.timestamp_hour,
                "department_id": "ER_MAIN",
                "facility_id": row.facility_id,
                "predicted_patient_volume": float(pred_volume),
                "model_components": {"tabfm_guarded": "TabFM guarded by fixed-rotation baseline", "statistical_ensemble": "HistGradientBoosting + ExtraTrees style synthetic ensemble", "conservative_peak": "statistical ensemble plus surge cushion"}[scenario],
                "scheduling_safe": True,
            }
            total_needed = 0.0
            for role in ROLES:
                base = float(getattr(row, f"actual_required_{role}"))
                uplift = 0.0
                if scenario == "conservative_peak":
                    threshold = {"md": 3.5, "np": 4.5, "rn": 3.0, "hcw": 3.2}[role]
                    uplift = 1.0 if float(row.wait_risk_proxy) >= threshold else 0.0
                out[f"predicted_required_{role}"] = base + uplift
                total_needed += base + uplift
            out["predicted_providers_needed"] = total_needed
            rows.append(out)
    ptd = pd.DataFrame(rows).sort_values(["scenario_name", "facility_id", "hour_block"]).reset_index(drop=True)
    path = out_dir / "predictions.tomorrow_demands.parquet"
    _write_table(ptd, path)
    return path


def _coverage_from_table(df: pd.DataFrame, start_col: str, end_col: str) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        start = pd.Timestamp(getattr(r, start_col)).floor("h")
        end = pd.Timestamp(getattr(r, end_col)).floor("h")
        for ts in pd.date_range(start, end, freq="h", inclusive="left"):
            row = {"facility_id": r.facility_id, "timestamp_hour": ts}
            for role in ROLES:
                row[f"{role}_cov"] = float(getattr(r, f"activation_capacity_{role}_equiv", 0.0))
            rows.append(row)
    cols = ["facility_id", "timestamp_hour"] + [f"{role}_cov" for role in ROLES]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).groupby(["facility_id", "timestamp_hour"], as_index=False).sum()


def _coverage_from_shifts(psh: pd.DataFrame, core_only: bool = True) -> pd.DataFrame:
    df = psh[psh["is_core_rotation"] == True] if core_only else psh
    return _coverage_from_table(df, "shift_start", "shift_end")


def _coverage_from_manual(manual: pd.DataFrame) -> pd.DataFrame:
    if manual.empty:
        return _coverage_from_table(manual, "manual_activation_start", "manual_activation_end")
    usable = manual[~manual["manual_activation_miss_reason"].astype(str).isin({"unavailable_on_call_provider", "wrong_department_or_facility"})]
    return _coverage_from_table(usable, "manual_activation_start", "manual_activation_end")


def _evaluate_mode(name: str, ass: pd.DataFrame, core_cov: pd.DataFrame, extra_cov: pd.DataFrame) -> dict:
    df = ass[["facility_id", "timestamp_hour", "wait_risk_proxy"] + [f"actual_required_{role}" for role in ROLES]].copy()
    df = df.merge(core_cov, on=["facility_id", "timestamp_hour"], how="left", suffixes=("", "_core"))
    for role in ROLES:
        df = df.rename(columns={f"{role}_cov": f"core_{role}"})
    df = df.merge(extra_cov, on=["facility_id", "timestamp_hour"], how="left")
    for role in ROLES:
        df = df.rename(columns={f"{role}_cov": f"extra_{role}"})
        for col in [f"core_{role}", f"extra_{role}"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[f"{role}_shortage"] = (df[f"actual_required_{role}"] - df[f"core_{role}"] - df[f"extra_{role}"]).clip(lower=0.0)
    df["shortage_hours"] = sum(df[f"{role}_shortage"] for role in ROLES)
    df["coverage_risk"] = df["shortage_hours"] > 0
    result = {"mode": name, "shortage_hours": float(df["shortage_hours"].sum()), "coverage_risk_hours": int(df["coverage_risk"].sum()), "wait_risk_proxy": float((df["wait_risk_proxy"] * df["coverage_risk"]).sum())}
    for role in ROLES:
        result[f"{role}_shortage_hours"] = float(df[f"{role}_shortage"].sum())
    return result


def _shortage_detail_by_hour(scenario: str, ass: pd.DataFrame, core_cov: pd.DataFrame, opt_cov: pd.DataFrame) -> pd.DataFrame:
    detail = ass[["facility_id", "timestamp_hour", "department_id", "actual_patient_volume", "waiting_room_count", "wait_risk_proxy"] + [f"actual_required_{role}" for role in ROLES]].copy()
    detail = detail.merge(core_cov, on=["facility_id", "timestamp_hour"], how="left")
    for role in ROLES:
        detail = detail.rename(columns={f"{role}_cov": f"core_{role}_coverage"})
    detail = detail.merge(opt_cov, on=["facility_id", "timestamp_hour"], how="left")
    for role in ROLES:
        detail = detail.rename(columns={f"{role}_cov": f"optimized_overflow_{role}_coverage"})
        for col in [f"core_{role}_coverage", f"optimized_overflow_{role}_coverage"]:
            detail[col] = pd.to_numeric(detail[col], errors="coerce").fillna(0.0)
        detail[f"remaining_{role}_gap"] = (detail[f"actual_required_{role}"] - detail[f"core_{role}_coverage"] - detail[f"optimized_overflow_{role}_coverage"]).clip(lower=0.0)
        detail[f"recommended_{role}_overflow_needed"] = detail[f"remaining_{role}_gap"]
    detail["recommended_total_overflow_needed"] = sum(detail[f"remaining_{role}_gap"] for role in ROLES)
    detail["priority_tier"] = np.where(detail["recommended_total_overflow_needed"] >= 4, "P1", np.where(detail["recommended_total_overflow_needed"] >= 2, "P2", "P3"))
    detail["scenario_name"] = scenario
    return detail[detail["recommended_total_overflow_needed"] > 0].sort_values(["recommended_total_overflow_needed", "wait_risk_proxy", "timestamp_hour"], ascending=[False, False, True]).reset_index(drop=True)


def optimize_overflow(paths: dict[str, Path], ptd_path: Path, out_dir: Path, budget_shares: dict[str, float], min_overflow_covered_gap_hours: float = 6.0) -> tuple[Path, Path, Path, Path]:
    psh = pd.read_parquet(paths["PSH"])
    ass = pd.read_parquet(paths["ASS"])
    manual = pd.read_parquet(paths["manual"])
    ptd = pd.read_parquet(ptd_path)
    core_cov = _coverage_from_shifts(psh, core_only=True)
    manual_cov = _coverage_from_manual(manual)
    core_hours_by_role = _core_hours_by_role(psh)
    budget_hours = {role: core_hours_by_role[role] * budget_shares[role] for role in ROLES}
    summaries, comparison_rows, psp_rows, need_detail_rows = [], [], [], []
    for scenario in ["tabfm_guarded", "statistical_ensemble", "conservative_peak"]:
        sptd = ptd[ptd["scenario_name"] == scenario].copy()
        demand = sptd.merge(ass, left_on=["facility_id", "hour_block"], right_on=["facility_id", "timestamp_hour"], how="left")
        demand = demand.merge(core_cov, left_on=["facility_id", "hour_block"], right_on=["facility_id", "timestamp_hour"], how="left")
        remaining_gap = {}
        for role in ROLES:
            demand[f"{role}_cov"] = pd.to_numeric(demand[f"{role}_cov"], errors="coerce").fillna(0.0)
            demand[f"pred_{role}_gap"] = (pd.to_numeric(demand[f"predicted_required_{role}"], errors="coerce").fillna(0.0) - demand[f"{role}_cov"]).clip(lower=0.0)
            remaining_gap[role] = demand.set_index(["facility_id", "hour_block"])[f"pred_{role}_gap"].to_dict()
        demand["priority"] = sum(demand[f"pred_{role}_gap"] * ROLE_WEIGHTS[role] for role in ROLES) + demand["wait_risk_proxy"] * 0.35
        demand_key = demand.set_index(["facility_id", "hour_block"])["priority"].to_dict()
        scored = []
        for r in psh[psh["is_overflow_candidate"] == True].itertuples(index=False):
            hours = pd.date_range(pd.Timestamp(r.activation_window_start), pd.Timestamp(r.activation_window_end), freq="h", inclusive="left")
            gain = 0.0
            for ts in hours:
                key = (r.facility_id, ts)
                covered_total = 0.0
                for role in ROLES:
                    cap = float(getattr(r, f"activation_capacity_{role}_equiv"))
                    covered = min(cap, float(remaining_gap[role].get(key, 0.0)))
                    covered_total += covered
                    gain += covered * ROLE_WEIGHTS[role]
                gain += covered_total * float(demand_key.get(key, 0.0)) * 0.15
            scored.append((gain / max(1.0, float(r.activation_cost)), r, hours))
        scored.sort(key=lambda x: (-x[0], str(x[1].shift_id)))
        used_by_role = {role: 0.0 for role in ROLES}
        provider_hour_used: set[tuple[str, pd.Timestamp]] = set()
        opt_rows = []
        for score, r, hours in scored:
            if score <= 0:
                continue
            block_hours = float(len(hours))
            if block_hours < 10:
                continue
            marginal_gap_hours = 0.0
            usage = {role: block_hours * float(getattr(r, f"activation_capacity_{role}_equiv")) for role in ROLES}
            for ts in hours:
                key = (r.facility_id, ts)
                for role in ROLES:
                    marginal_gap_hours += min(float(getattr(r, f"activation_capacity_{role}_equiv")), float(remaining_gap[role].get(key, 0.0)))
            if marginal_gap_hours < min_overflow_covered_gap_hours:
                continue
            if any(used_by_role[role] + usage[role] > budget_hours[role] + 1e-9 for role in ROLES):
                continue
            if any((str(r.provider_id), ts) in provider_hour_used for ts in hours):
                continue
            for role in ROLES:
                used_by_role[role] += usage[role]
            for ts in hours:
                provider_hour_used.add((str(r.provider_id), ts))
                key = (r.facility_id, ts)
                for role in ROLES:
                    remaining_gap[role][key] = max(0.0, float(remaining_gap[role].get(key, 0.0)) - float(getattr(r, f"activation_capacity_{role}_equiv")))
            hit = bool(any(float(demand_key.get((r.facility_id, ts), 0.0)) > 0 for ts in hours))
            row = {"scenario_name": scenario, "plan_id": f"PSP_{scenario}_{len(opt_rows):05d}", "shift_id": r.shift_id, "provider_id": r.provider_id, "department_id": r.department_id, "facility_id": r.facility_id, "planned_activation_start": r.activation_window_start, "planned_activation_end": r.activation_window_end, "role": r.role, "overflow_hours": block_hours, "optimizer_expected_hit": hit, "priority_score": float(score), "priority_tier": "P1" if score >= 8 else ("P2" if score >= 4 else "P3"), "is_core_rotation": False, "is_overflow_candidate": True}
            for role in ROLES:
                row[f"activation_capacity_{role}_equiv"] = float(getattr(r, f"activation_capacity_{role}_equiv"))
            opt_rows.append(row)
            psp_rows.append(row)
        opt = pd.DataFrame(opt_rows)
        opt_cov = pd.DataFrame(columns=["facility_id", "timestamp_hour"] + [f"{role}_cov" for role in ROLES]) if opt.empty else _coverage_from_table(opt, "planned_activation_start", "planned_activation_end")
        static_metrics = _evaluate_mode("static_fixed_rotation", ass, core_cov, pd.DataFrame(columns=["facility_id", "timestamp_hour"] + [f"{role}_cov" for role in ROLES]))
        manual_metrics = _evaluate_mode("manual_threshold_overflow", ass, core_cov, manual_cov)
        opt_metrics = _evaluate_mode("optimized_overflow_reallocation", ass, core_cov, opt_cov)
        need_detail = _shortage_detail_by_hour(scenario, ass, core_cov, opt_cov)
        if not need_detail.empty:
            need_detail_rows.extend(need_detail.to_dict("records"))
        total_required_hours = float(sum(ass[f"actual_required_{role}"].sum() for role in ROLES))
        summary = {
            "scenario_name": scenario,
            "total_core_monthly_provider_hours": sum(core_hours_by_role.values()),
            "total_overflow_budget_hours": sum(budget_hours.values()),
            "total_optimized_overflow_hours_used": sum(used_by_role.values()),
            "total_overflow_hours_remaining": sum(budget_hours[r] - used_by_role[r] for r in ROLES),
            "budget_exceeded": bool(any(used_by_role[r] > budget_hours[r] + 1e-9 for r in ROLES)),
            "total_required_provider_hours": total_required_hours,
            "optimized_shortage_as_pct_required_hours": float(opt_metrics["shortage_hours"] / max(1.0, total_required_hours)),
            "shortage_reduction_pct_vs_manual": float((manual_metrics["shortage_hours"] - opt_metrics["shortage_hours"]) / max(1.0, manual_metrics["shortage_hours"])),
            "coverage_risk_reduction_pct_vs_manual": float((manual_metrics["coverage_risk_hours"] - opt_metrics["coverage_risk_hours"]) / max(1.0, manual_metrics["coverage_risk_hours"])),
            "static_shortage_hours": static_metrics["shortage_hours"],
            "manual_shortage_hours": manual_metrics["shortage_hours"],
            "optimized_shortage_hours": opt_metrics["shortage_hours"],
            "static_coverage_risk_hours": static_metrics["coverage_risk_hours"],
            "manual_coverage_risk_hours": manual_metrics["coverage_risk_hours"],
            "optimized_coverage_risk_hours": opt_metrics["coverage_risk_hours"],
            "manual_overflow_hit_rate": float(manual["manual_activation_hit"].mean()) if not manual.empty else 0.0,
            "optimized_overflow_hit_rate": float(opt["optimizer_expected_hit"].mean()) if not opt.empty else 0.0,
            "manual_missed_activations": int((~manual["manual_activation_hit"].astype(bool)).sum()) if not manual.empty else 0,
            "optimized_missed_activations": int((~opt["optimizer_expected_hit"].astype(bool)).sum()) if not opt.empty else 0,
            "shortage_reduction_vs_manual": manual_metrics["shortage_hours"] - opt_metrics["shortage_hours"],
            "coverage_risk_reduction_vs_manual": manual_metrics["coverage_risk_hours"] - opt_metrics["coverage_risk_hours"],
            "wait_risk_proxy_reduction_vs_manual": manual_metrics["wait_risk_proxy"] - opt_metrics["wait_risk_proxy"],
            "p1_overflow_assignments": int((opt["priority_tier"] == "P1").sum()) if not opt.empty else 0,
            "p2_overflow_assignments": int((opt["priority_tier"] == "P2").sum()) if not opt.empty else 0,
            "p3_overflow_assignments": int((opt["priority_tier"] == "P3").sum()) if not opt.empty else 0,
            "rollout_pass": bool((manual_metrics["shortage_hours"] - opt_metrics["shortage_hours"]) > 0 and not any(used_by_role[r] > budget_hours[r] + 1e-9 for r in ROLES)),
        }
        for role in ROLES:
            summary[f"core_{role}_hours"] = core_hours_by_role[role]
            summary[f"overflow_budget_share_{role}"] = budget_shares[role]
            summary[f"overflow_budget_hours_{role}"] = budget_hours[role]
            summary[f"optimized_{role}_overflow_hours_used"] = used_by_role[role]
            summary[f"overflow_hours_remaining_{role}"] = budget_hours[role] - used_by_role[role]
            summary[f"manual_{role}_shortage_hours"] = manual_metrics[f"{role}_shortage_hours"]
            summary[f"optimized_{role}_shortage_hours"] = opt_metrics[f"{role}_shortage_hours"]
        summaries.append(summary)
        for metric in [static_metrics, manual_metrics, opt_metrics]:
            comparison_rows.append({"scenario_name": scenario, **metric})
    psp = pd.DataFrame(psp_rows)
    summary = pd.DataFrame(summaries)
    comparison = pd.DataFrame(comparison_rows)
    need_detail_df = pd.DataFrame(need_detail_rows)
    psp_path = out_dir / "planned_schedule_table.parquet"
    summary_path = out_dir / "overflow_reallocation_summary.csv"
    comparison_path = out_dir / "manual_vs_optimized_overflow.csv"
    need_detail_path = out_dir / "recommended_overflow_capacity_needed_by_hour.csv"
    _write_table(psp, psp_path)
    _write_table(summary, summary_path)
    _write_table(comparison, comparison_path)
    _write_table(need_detail_df, need_detail_path)
    return psp_path, summary_path, comparison_path, need_detail_path


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, summary_path: Path, comparison_path: Path, need_detail_path: Path) -> tuple[Path, Path]:
    summary = pd.read_csv(summary_path)
    comparison = pd.read_csv(comparison_path)
    need_detail = pd.read_csv(need_detail_path) if need_detail_path.exists() else pd.DataFrame()
    lines = ["# Hospital Schedule Optimizer Multi-Role Overflow Report", "", "## Executive summary"]
    if not summary.empty:
        best = summary.sort_values(["rollout_pass", "shortage_reduction_vs_manual", "optimized_overflow_hit_rate"], ascending=[False, False, False]).head(1).iloc[0]
        lines.append(f"- Best scenario: `{best['scenario_name']}`.")
        lines.append(f"- Total funded overflow budget: {best['total_overflow_budget_hours']:.1f} role-hours.")
        lines.append(f"- Optimized overflow used: {best['total_optimized_overflow_hours_used']:.1f} role-hours.")
        lines.append(f"- Shortage reduction versus manual: {best['shortage_reduction_vs_manual']:.1f} role-hours ({best['shortage_reduction_pct_vs_manual']:.1%}).")
        lines.append(f"- Coverage-risk reduction versus manual: {best['coverage_risk_reduction_vs_manual']:.1f} hours ({best['coverage_risk_reduction_pct_vs_manual']:.1%}).")
        for role in ROLES:
            lines.append(f"- {ROLE_LABELS[role]} budget used: {best[f'optimized_{role}_overflow_hours_used']:.1f} of {best[f'overflow_budget_hours_{role}']:.1f} hours.")
    for title, df in [("Overflow reallocation summary", summary), ("Manual vs optimized modes", comparison), ("Recommended overflow capacity needed by hour", need_detail)]:
        lines.extend(["", f"## {title}", "", _to_markdown(df.head(10))])
    md = "\n".join(lines) + "\n"
    md_path = out_dir / "fixed_rotation_overflow_report.md"
    html_path = out_dir / "fixed_rotation_overflow_report.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text("<html><body><pre>" + html.escape(md) + "</pre></body></html>", encoding="utf-8")
    return md_path, html_path


def main() -> None:
    args = parse_args()
    budget_shares = _budget_shares(args)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = generate_canonical_mock(args.seed, args.facilities, args.days, budget_shares, out_dir, overflow_shift_hours=args.overflow_shift_hours, demand_pressure_multiplier=args.demand_pressure_multiplier)
    ptd_path = build_ptd_scenarios(paths, out_dir)
    paths["PTD"] = ptd_path
    psp_path, summary_path, comparison_path, need_detail_path = optimize_overflow(paths, ptd_path, out_dir, budget_shares, min_overflow_covered_gap_hours=args.min_overflow_covered_gap_hours)
    paths["PSP"] = psp_path
    paths["summary"] = summary_path
    paths["comparison"] = comparison_path
    paths["need_detail"] = need_detail_path
    report_md, report_html = write_report(out_dir, summary_path, comparison_path, need_detail_path)
    print("\nFixed-rotation overflow workflow complete")
    print(f"Output root: {out_dir}")
    print(f"Report markdown: {report_md}")
    print(f"Report HTML: {report_html}")
    labels = [
        ("EVH / er_visits_historical", paths["EVH"]),
        ("PSH / provider_shifts_historical", paths["PSH"]),
        ("RAV / resource_availability", paths["RAV"]),
        ("USS / upcoming_schedule_skeleton", paths["USS"]),
        ("PTD / predictions.tomorrow_demands", paths["PTD"]),
        ("manual_overflow_activation_log", paths["manual"]),
        ("PSP / planned_schedule_table", paths["PSP"]),
        ("ASS / actual_system_state", paths["ASS"]),
        ("overflow_reallocation_summary", paths["summary"]),
        ("manual_vs_optimized_overflow", paths["comparison"]),
        ("recommended_overflow_capacity_needed_by_hour", paths["need_detail"]),
    ]
    for label, path in labels:
        _print_top(label, path, args.print_top_n)


if __name__ == "__main__":
    main()
