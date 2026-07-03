from __future__ import annotations

from collections import defaultdict
import hashlib
import ast
import time
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import pulp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pulp = None


def _first_available_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"missing one of columns: {list(candidates)}")


def _coerce_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _hour_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start).floor("h")
    end_ts = pd.Timestamp(end).floor("h")
    if end_ts <= start_ts:
        return []
    return pd.date_range(start_ts, end_ts, freq="1h", inclusive="left").tolist()


def _blocked_offsets(seed_key: str, n_hours: int, n_blocks: int) -> set[int]:
    if n_hours <= 0 or n_blocks <= 0:
        return set()
    n_hours = int(n_hours)
    n_blocks = min(int(max(0, n_blocks)), n_hours)
    if n_blocks == 0:
        return set()
    digest = hashlib.md5(seed_key.encode("utf-8")).hexdigest()
    rnd = np.random.default_rng(int(digest[:12], 16))
    return set(int(i) for i in rnd.choice(np.arange(n_hours), size=n_blocks, replace=False).tolist())


def _coerce_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)


def _coverage_list_size(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return 0
        if isinstance(parsed, (list, tuple, set)):
            return len(parsed)
    return 0


def _assignment_coverage_hours(assignments: pd.DataFrame) -> dict[str, float]:
    if assignments is None or assignments.empty:
        return {
            "recommended_provider_hours": 0.0,
            "recommended_doc_hours": 0.0,
            "recommended_np_hours": 0.0,
            "recommended_rn_hours": 0.0,
            "normal_provider_hours": 0.0,
            "overflow_provider_hours": 0.0,
            "float_pool_hours": 0.0,
            "on_call_hours": 0.0,
            "overtime_extension_hours": 0.0,
            "recommended_shift_count": 0,
        }

    out = {
        "recommended_provider_hours": 0.0,
        "recommended_doc_hours": 0.0,
        "recommended_np_hours": 0.0,
        "recommended_rn_hours": 0.0,
        "normal_provider_hours": 0.0,
        "overflow_provider_hours": 0.0,
        "float_pool_hours": 0.0,
        "on_call_hours": 0.0,
        "overtime_extension_hours": 0.0,
        "recommended_shift_count": int(len(assignments)),
    }
    for row in assignments.itertuples(index=False):
        cov = _coverage_list_size(getattr(row, "coverage_hours", None))
        role = str(getattr(row, "role", "")).upper()
        recovery = str(getattr(row, "recovery_lever", "") or "")
        is_overflow = role == "HOSPITALIST_OVERFLOW" or recovery == "hospitalist_overflow"
        if role == "MD":
            out["recommended_doc_hours"] += float(cov)
        elif role == "NP":
            out["recommended_np_hours"] += float(cov)
        elif role == "RN":
            out["recommended_rn_hours"] += float(cov)
        elif role == "HOSPITALIST_OVERFLOW":
            out["recommended_doc_hours"] += 0.75 * float(cov)
        if is_overflow:
            out["overflow_provider_hours"] += float(cov)
        else:
            out["normal_provider_hours"] += float(cov)
        if int(getattr(row, "is_float_pool", 0) or 0) == 1:
            out["float_pool_hours"] += float(cov)
        if bool(getattr(row, "on_call", False)):
            out["on_call_hours"] += float(cov)
        if recovery == "overtime_extension":
            out["overtime_extension_hours"] += float(cov)
        out["recommended_provider_hours"] += float(cov)
    return out


def _required_staff_from_demand(
    demand: pd.Series,
    occupancy: pd.Series,
    doc_capacity: float,
    np_capacity: float,
    doc_share: float,
) -> tuple[pd.Series, pd.Series]:
    doc_capacity = float(max(1e-6, doc_capacity))
    np_capacity = float(max(1e-6, np_capacity))
    occ = _coerce_float(occupancy).clip(0.0, 1.0)
    multiplier = (0.55 + 0.35 * occ).clip(0.25, 1.9)
    effective = _coerce_float(demand) * multiplier
    doc_share = float(np.clip(doc_share, 0.05, 0.95))
    req_doc = np.ceil(np.maximum(effective * doc_share / doc_capacity, 0.0))
    req_np = np.ceil(np.maximum(effective * (1.0 - doc_share) / np_capacity, 0.0))
    return pd.Series(req_doc, index=demand.index), pd.Series(req_np, index=demand.index)


def build_demand_profile(
    forecast_df: pd.DataFrame,
    shift_hours: int,
    demand_mode: str = "next_nh",
    required_horizon_start: str | pd.Timestamp | None = None,
    required_horizon_end: str | pd.Timestamp | None = None,
    doc_capacity_per_hour: float = 4.0,
    np_capacity_per_hour: float = 3.0,
    doc_share: float = 0.65,
) -> pd.DataFrame:
    if forecast_df.empty:
        raise ValueError("forecast_df is empty")
    if "facility_id" not in forecast_df.columns or "timestamp" not in forecast_df.columns:
        raise ValueError("forecast_df requires facility_id and timestamp")

    df = forecast_df.copy()
    df["timestamp"] = _coerce_ts(df["timestamp"])
    df = df.sort_values(["facility_id", "timestamp"]).reset_index(drop=True)

    pred_col = _first_available_col(
        df,
        ["y_pred", "predicted_arrivals", "prediction", "forecast", "expected_arrivals_next_hour"],
    )

    if demand_mode == "next_4h" and "expected_arrivals_next_4h" in df.columns:
        demand_col = "expected_arrivals_next_4h"
        demand = _coerce_float(df[demand_col])
    elif demand_mode == "next_nh":
        demand = _coerce_float(df[pred_col])
    else:
        demand = _coerce_float(df[pred_col])

    profile = df[[c for c in df.columns if c in {"facility_id", "timestamp", "occupancy_pressure", "current_docs_on_duty", "current_nps_on_duty"}]].copy()
    for c in ["occupancy_pressure", "current_docs_on_duty", "current_nps_on_duty"]:
        if c not in profile.columns:
            profile[c] = 0.0
    profile["demand_units"] = demand
    profile["shift_hours"] = int(max(1, shift_hours))
    profile = profile.groupby(["facility_id", "timestamp"], as_index=False).agg(
        demand_units=("demand_units", "mean"),
        occupancy_pressure=("occupancy_pressure", "mean"),
        current_docs_on_duty=("current_docs_on_duty", "mean"),
        current_nps_on_duty=("current_nps_on_duty", "mean"),
    )

    req_doc, req_np = _required_staff_from_demand(
        profile["demand_units"],
        profile["occupancy_pressure"],
        doc_capacity=doc_capacity_per_hour,
        np_capacity=np_capacity_per_hour,
        doc_share=doc_share,
    )
    profile["required_docs_hour"] = req_doc.astype(int)
    profile["required_nps_hour"] = req_np.astype(int)
    profile["shift_start"] = profile["timestamp"].dt.floor(f"{int(max(1, shift_hours))}h")
    profile["required_arrivals_units"] = profile["demand_units"]

    start_ts = pd.to_datetime(required_horizon_start) if required_horizon_start is not None else None
    end_ts = pd.to_datetime(required_horizon_end) if required_horizon_end is not None else None
    if start_ts is not None:
        profile = profile[profile["timestamp"] >= start_ts]
    if end_ts is not None:
        profile = profile[profile["timestamp"] < end_ts]

    return profile.sort_values(["facility_id", "timestamp"]).reset_index(drop=True)


def build_shift_candidate_pool(
    provider_shifts_df: pd.DataFrame,
    min_shift_hours: int,
    required_horizon_start,
    required_horizon_end,
    allow_float_pool: bool = True,
    allow_overtime_extensions: bool = True,
) -> pd.DataFrame:
    if provider_shifts_df.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "facility_id",
                "provider_id",
                "role",
                "shift_start",
                "shift_end",
                "shift_hours",
                "coverage_hours",
                "blocked_hours",
                "availability_ratio",
                "on_call",
                "is_float_pool",
                "is_cross_facility",
                "recovery_lever",
            ]
        )

    required = {"facility_id", "provider_id", "role", "shift_start", "shift_end"}
    if not required.issubset(provider_shifts_df.columns):
        raise ValueError(f"provider_shifts_df missing required columns: {sorted(required)}")

    df = provider_shifts_df.copy()
    df["shift_start"] = _coerce_ts(df["shift_start"])
    df["shift_end"] = _coerce_ts(df["shift_end"])
    if not allow_float_pool and "is_float_pool" in df.columns:
        df = df[pd.to_numeric(df["is_float_pool"], errors="coerce").fillna(0).astype(int) == 0].copy()
    if not allow_overtime_extensions and "recovery_lever" in df.columns:
        df = df[df["recovery_lever"].astype(str) != "overtime_extension"].copy()
    if "planned_shift_hours" in df.columns:
        base_shift_len = pd.to_numeric(df["planned_shift_hours"], errors="coerce")
    elif "max_contiguous_hours" in df.columns:
        base_shift_len = pd.to_numeric(df["max_contiguous_hours"], errors="coerce")
    else:
        base_shift_len = (df["shift_end"] - df["shift_start"]).dt.total_seconds() / 3600

    start_ts = pd.to_datetime(required_horizon_start) if required_horizon_start is not None else None
    end_ts = pd.to_datetime(required_horizon_end) if required_horizon_end is not None else None
    min_shift_hours = int(max(1, min_shift_hours))

    rows: list[dict] = []
    for idx, row in df.iterrows():
        s = row["shift_start"]
        e = row["shift_end"]
        if pd.isna(s) or pd.isna(e):
            continue
        if end_ts is not None and s >= end_ts:
            continue
        if start_ts is not None and e <= start_ts:
            continue

        raw_len = base_shift_len.loc[idx]
        try:
            planned_len = int(float(raw_len))
        except Exception:
            planned_len = int((e - s).total_seconds() // 3600)
        duration = int(max(1, min(24, max(planned_len, int((e - s).total_seconds() // 3600)))))
        if duration < min_shift_hours:
            continue

        all_hours = _hour_range(s, e)
        if not all_hours:
            continue

        unavailable = int(row.get("unavailable_hours", 0) or 0)
        blocked_offsets = _blocked_offsets(f"{row['provider_id']}|{s}|{e}|{idx}", len(all_hours), unavailable)
        available = [ts for i, ts in enumerate(all_hours) if i not in blocked_offsets]
        if not available:
            continue

        rows.append(
            {
                "candidate_id": f"CAND_{int(idx):06d}",
                "facility_id": row["facility_id"],
                "provider_id": row["provider_id"],
                "role": row["role"],
                "shift_start": s.floor("h"),
                "shift_end": e.floor("h"),
                "shift_hours": duration,
                "coverage_hours": available,
                "blocked_hours": len(blocked_offsets),
                "availability_ratio": float(len(available) / len(all_hours)),
                "on_call": bool(row.get("on_call", False)),
                "is_float_pool": int(row.get("is_float_pool", 0) or 0),
                "is_cross_facility": int(row.get("is_cross_facility", 0) or 0),
                "recovery_lever": str(row.get("recovery_lever", "baseline") or "baseline"),
                "max_monthly_hours": float(row.get("max_monthly_hours", 999999) or 999999),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "facility_id",
                "provider_id",
                "role",
                "shift_start",
                "shift_end",
                "shift_hours",
                "coverage_hours",
                "blocked_hours",
                "availability_ratio",
                "on_call",
                "is_float_pool",
                "is_cross_facility",
                "recovery_lever",
            ]
        )

    out = pd.DataFrame(rows).sort_values(["facility_id", "shift_start", "provider_id", "candidate_id"]).reset_index(drop=True)
    return out


def diagnose_staffing_feasibility(
    demand_profile: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    assignments: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    if demand_profile.empty:
        return pd.DataFrame(), {
            "required_role_hours_total": 0.0,
            "available_role_hours_total": 0.0,
            "infeasible_shortage_hours": 0.0,
            "avoidable_shortage_hours": 0.0,
            "feasible_coverage_share": 1.0,
        }

    assignment_cov: dict[tuple[str, pd.Timestamp], dict[str, float]] = defaultdict(lambda: {"MD": 0.0, "NP": 0.0})
    if assignments is not None and not assignments.empty:
        for row in assignments.itertuples(index=False):
            role = str(row.role).upper()
            facility = str(row.facility_id)
            for ts in row.coverage_hours:
                key = (facility, pd.Timestamp(ts))
                if role == "MD":
                    assignment_cov[key]["MD"] += 1.0
                elif role == "NP":
                    assignment_cov[key]["NP"] += 1.0
                elif role == "RN":
                    assignment_cov[key]["MD"] += 0.25
                elif role == "HOSPITALIST_OVERFLOW":
                    assignment_cov[key]["MD"] += 0.75

    available_cov: dict[tuple[str, pd.Timestamp], dict[str, float]] = defaultdict(lambda: {"MD": 0.0, "NP": 0.0})
    if candidate_pool is not None and not candidate_pool.empty:
        for row in candidate_pool.itertuples(index=False):
            role = str(row.role).upper()
            facility = str(row.facility_id)
            for ts in row.coverage_hours:
                key = (facility, pd.Timestamp(ts))
                if role == "MD":
                    available_cov[key]["MD"] += 1.0
                elif role == "NP":
                    available_cov[key]["NP"] += 1.0
                elif role == "RN":
                    available_cov[key]["MD"] += 0.25
                elif role == "HOSPITALIST_OVERFLOW":
                    available_cov[key]["MD"] += 0.75

    for row in demand_profile.itertuples(index=False):
        facility = str(row.facility_id)
        ts = pd.Timestamp(row.timestamp)
        for role, req in [("MD", float(row.required_docs_hour)), ("NP", float(row.required_nps_hour))]:
            avail = float(available_cov[(facility, ts)][role])
            assigned = float(assignment_cov[(facility, ts)][role])
            infeasible = max(0.0, req - avail)
            post_shortage = max(0.0, req - assigned)
            rows.append(
                {
                    "facility_id": facility,
                    "timestamp": ts,
                    "role": role,
                    "required_role_hours": req,
                    "available_role_hours": avail,
                    "assigned_role_hours": assigned,
                    "infeasible_shortage_hours": infeasible,
                    "avoidable_shortage_hours": max(0.0, post_shortage - infeasible),
                    "post_optimization_shortage_hours": post_shortage,
                    "feasible_role_coverage_share": min(1.0, avail / req) if req > 0 else 1.0,
                }
            )

    detail = pd.DataFrame(rows)
    required = float(detail["required_role_hours"].sum()) if not detail.empty else 0.0
    available = float(np.minimum(detail["available_role_hours"], detail["required_role_hours"]).sum()) if not detail.empty else 0.0
    summary = {
        "required_role_hours_total": required,
        "available_role_hours_total": float(detail["available_role_hours"].sum()) if not detail.empty else 0.0,
        "max_feasible_covered_role_hours_total": available,
        "infeasible_shortage_hours": float(detail["infeasible_shortage_hours"].sum()) if not detail.empty else 0.0,
        "avoidable_shortage_hours": float(detail["avoidable_shortage_hours"].sum()) if not detail.empty else 0.0,
        "post_optimization_shortage_hours": float(detail["post_optimization_shortage_hours"].sum()) if not detail.empty else 0.0,
        "feasible_coverage_share": float(available / required) if required > 0 else 1.0,
    }
    return detail, summary


def _plan_kpis(demand_profile: pd.DataFrame, assignments: pd.DataFrame, min_shift_hours: int) -> dict:
    if demand_profile.empty:
        return {
            "expected_shortfall_hours": 0.0,
            "hard_violation_count": 0,
            "gaps": [],
            "utilization": 0.0,
            "shortage_hours": 0.0,
            "overtime_proxy": 0.0,
            "blocked_availability_hours": 0,
            "sla_risk_reduction_proxy": 0.0,
        }

    target_by_hour = {
        (str(r.facility_id), pd.Timestamp(r.timestamp)): {"req_doc": int(r.required_docs_hour), "req_np": int(r.required_nps_hour), "act_doc": 0.0, "act_np": 0.0}
        for r in demand_profile.itertuples(index=False)
    }

    for row in assignments.itertuples(index=False):
        role = str(row.role)
        facility = str(row.facility_id)
        for ts in row.coverage_hours:
            t = pd.Timestamp(ts)
            key = (facility, t)
            if key not in target_by_hour:
                continue
            if role == "MD":
                target_by_hour[key]["act_doc"] += 1
            elif role == "NP":
                target_by_hour[key]["act_np"] += 1
            elif role == "RN":
                target_by_hour[key]["act_doc"] += 0.25
            elif role == "HOSPITALIST_OVERFLOW":
                target_by_hour[key]["act_doc"] += 0.75
            else:
                target_by_hour[key]["act_doc"] += 1

    gaps = []
    shortfall = 0.0
    gap_count = 0
    req_total = 0.0
    covered_total = 0.0
    for (facility, ts), info in target_by_hour.items():
        req_d = float(info["req_doc"])
        req_n = float(info["req_np"])
        act_d = float(info["act_doc"])
        act_n = float(info["act_np"])
        gap_d = max(0.0, req_d - act_d)
        gap_n = max(0.0, req_n - act_n)
        if gap_d > 0.0 or gap_n > 0.0:
            gap_count += 1
            gaps.append(
                {
                    "facility_id": facility,
                    "timestamp": str(ts),
                    "missing_doc_capacity": gap_d,
                    "missing_np_capacity": gap_n,
                    "required_docs": req_d,
                    "required_nps": req_n,
                    "actual_docs": act_d,
                    "actual_nps": act_n,
                }
            )
        shortfall += gap_d + gap_n
        req_total += req_d + req_n
        covered_total += max(0.0, req_d + req_n - (gap_d + gap_n))

    utilization = float(covered_total / req_total) if req_total > 0 else 1.0
    overtime = float(
        sum(max(0, int(r.shift_hours) - int(min_shift_hours)) for r in assignments.itertuples(index=False))
    )
    blocked_hours = int(assignments["blocked_hours"].sum()) if "blocked_hours" in assignments.columns and not assignments.empty else 0

    return {
        "expected_shortfall_hours": float(shortfall),
        "hard_violation_count": int(gap_count),
        "gaps": gaps,
        "utilization": utilization,
        "shortage_hours": float(shortfall),
        "overtime_proxy": overtime,
        "blocked_availability_hours": blocked_hours,
        "sla_risk_reduction_proxy": 0.0,
    }


def _solve_with_milp(
    demand_profile: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    min_shift_hours: int,
    shortfall_penalty: float,
    shift_penalty: float,
    overtime_penalty: float,
    time_limit: float | None,
    normal_provider_hour_budget: float | None = None,
    emergency_hour_bank: float = 0.0,
    overflow_role_policy: str = "hospitalist_md_equiv",
) -> tuple[pd.DataFrame, dict]:
    if pulp is None:
        return pd.DataFrame(), {"status": "skipped", "status_reason": "pulp_not_available"}

    model = pulp.LpProblem("staffing_plan", pulp.LpMinimize)
    demands = [(str(r.facility_id), pd.Timestamp(r.timestamp), int(r.required_docs_hour), int(r.required_nps_hour))
               for r in demand_profile.itertuples(index=False)]
    candidate_ids = candidate_pool["candidate_id"].tolist()
    x = {cid: pulp.LpVariable(f"x_{cid}", lowBound=0, upBound=1, cat="Binary") for cid in candidate_ids}
    s_doc = {(f, t): pulp.LpVariable(f"sd_{i}", lowBound=0) for i, (f, t, _, _) in enumerate(demands)}
    s_np = {(f, t): pulp.LpVariable(f"sn_{i}", lowBound=0) for i, (f, t, _, _) in enumerate(demands)}

    cand_info = {}
    provider_hour_coverage: dict[tuple[str, pd.Timestamp], list[str]] = defaultdict(list)
    for row in candidate_pool.itertuples(index=False):
        cid = row.candidate_id
        facility = str(row.facility_id)
        role = str(row.role)
        hours = [pd.Timestamp(h) for h in row.coverage_hours]
        cand_info[cid] = (facility, role, hours)
        for h in hours:
            provider_hour_coverage[(str(row.provider_id), h)].append(cid)

    for i, (facility, ts, req_doc, req_np) in enumerate(demands):
        doc_vars = []
        np_vars = []
        for cid, (cfac, role, hours) in cand_info.items():
            if cfac != facility or ts not in hours:
                continue
            if role == "MD":
                doc_vars.append(x[cid])
            elif role == "NP":
                np_vars.append(x[cid])
            elif role == "RN":
                doc_vars.append(0.6 * x[cid])
            elif role == "HOSPITALIST_OVERFLOW" and overflow_role_policy == "hospitalist_md_equiv":
                doc_vars.append(0.75 * x[cid])
        model += (pulp.lpSum(doc_vars) + s_doc[(facility, ts)] >= req_doc)
        model += (pulp.lpSum(np_vars) + s_np[(facility, ts)] >= req_np)

    # one provider cannot overlap two shifts in the same hour
    for (provider_id, ts), cands in provider_hour_coverage.items():
        if len(cands) > 1:
            model += (pulp.lpSum(x[c] for c in cands) <= 1)

    normal_hours_expr = []
    overflow_hours_expr = []
    provider_monthly: dict[str, list] = defaultdict(list)
    for row in candidate_pool.itertuples(index=False):
        cid = row.candidate_id
        cov = len(row.coverage_hours)
        role = str(row.role).upper()
        recovery = str(getattr(row, "recovery_lever", "") or "")
        is_overflow = role == "HOSPITALIST_OVERFLOW" or recovery == "hospitalist_overflow"
        if is_overflow:
            overflow_hours_expr.append(cov * x[cid])
        else:
            normal_hours_expr.append(cov * x[cid])
        provider_monthly[str(row.provider_id)].append((cov, x[cid], float(getattr(row, "max_monthly_hours", 999999) or 999999)))
    if normal_provider_hour_budget is not None and normal_provider_hour_budget >= 0:
        model += (pulp.lpSum(normal_hours_expr) <= float(normal_provider_hour_budget))
    if emergency_hour_bank >= 0:
        model += (pulp.lpSum(overflow_hours_expr) <= float(emergency_hour_bank))
    for provider_id, entries in provider_monthly.items():
        cap = min(v[2] for v in entries)
        if cap < 999999:
            model += (pulp.lpSum(cov * var for cov, var, _ in entries) <= float(cap))

    max_staffing_cost = max(
        1.0,
        sum(
            shift_penalty
            + overtime_penalty * max(0, int(row.shift_hours) - min_shift_hours)
            + (2.5 if str(getattr(row, "role", "")).upper() == "HOSPITALIST_OVERFLOW" else 0.0)
            for row in candidate_pool.itertuples(index=False)
        ),
    )
    safety_first_penalty = max(float(shortfall_penalty), (max_staffing_cost + 1.0) * 100.0)

    shortfall = pulp.lpSum(safety_first_penalty * (s_doc[k] + s_np[k]) for k in s_doc)
    staffing_cost_terms = []
    for row in candidate_pool.itertuples(index=False):
        cid = row.candidate_id
        role = str(row.role).upper()
        recovery = str(getattr(row, "recovery_lever", "") or "")
        overflow_penalty = 2.5 if (role == "HOSPITALIST_OVERFLOW" or recovery == "hospitalist_overflow") else 0.0
        staffing_cost_terms.append(
            (shift_penalty + overflow_penalty) * x[cid]
            + overtime_penalty * max(0, int(row.shift_hours) - min_shift_hours) * x[cid]
        )
    staffing_cost = pulp.lpSum(staffing_cost_terms)
    # Safety-first: any avoidable coverage shortfall dominates all hour-saving
    # preferences. Staffing cost is only optimized after coverage is protected.
    model += shortfall + staffing_cost

    solver = pulp.PULP_CBC_CMD(msg=False)
    if time_limit is not None and time_limit > 0:
        solver.timeLimit = float(time_limit)
    model.solve(solver)

    status = pulp.LpStatus.get(model.status, "unknown").lower()
    if status not in {"optimal", "feasible", "suboptimal"}:
        return pd.DataFrame(), {"status": status, "status_reason": "milp_no_solution"}

    selected = []
    for row in candidate_pool.itertuples(index=False):
        val = x[row.candidate_id].value()
        if val is not None and val > 0.5:
            selected.append(row)
    selected_df = pd.DataFrame(selected)
    return selected_df, {
        "status": status,
        "status_reason": "milp_solution",
        "safety_first_shortfall_penalty": float(safety_first_penalty),
    }


def _solve_with_greedy(
    demand_profile: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    min_shift_hours: int,
    shortfall_penalty: float,
    shift_penalty: float,
    overtime_penalty: float,
    time_limit: float | None,
) -> tuple[pd.DataFrame, dict]:
    remaining = {
        (str(r.facility_id), pd.Timestamp(r.timestamp)): {
            "req_doc": float(r.required_docs_hour),
            "req_np": float(r.required_nps_hour),
            "cov_doc": 0.0,
            "cov_np": 0.0,
        }
        for r in demand_profile.itertuples(index=False)
    }

    def unmet_gain(role: str, facility: str, ts: pd.Timestamp) -> float:
        key = (facility, ts)
        info = remaining.get(key, {"req_doc": 0.0, "req_np": 0.0, "cov_doc": 0.0, "cov_np": 0.0})
        if role == "MD":
            return max(0.0, info["req_doc"] - info["cov_doc"])
        if role == "NP":
            return max(0.0, info["req_np"] - info["cov_np"])
        if role == "RN":
            return 0.4 * max(0.0, info["req_doc"] - info["cov_doc"])
        return max(0.0, info["req_doc"] - info["cov_doc"])

    selected_ids: set[str] = set()
    selected_rows = []
    provider_hour_used: set[tuple[str, pd.Timestamp]] = set()

    while True:
        best_score = -1e9
        best_id = ""
        best_payload = None
        for row in candidate_pool.itertuples(index=False):
            cid = row.candidate_id
            if cid in selected_ids:
                continue
            facility = str(row.facility_id)
            role = str(row.role)
            overlap = False
            gain = 0.0
            for ts in row.coverage_hours:
                key = (str(row.provider_id), pd.Timestamp(ts))
                if key in provider_hour_used:
                    overlap = True
                    break
                gain += unmet_gain(role, facility, pd.Timestamp(ts))
            if overlap or gain <= 0:
                continue
            cost = shift_penalty + overtime_penalty * max(0, int(row.shift_hours) - min_shift_hours)
            score = gain - cost
            if (score > best_score) or (
                score == best_score and cid < best_id
            ):
                best_score = score
                best_id = cid
                best_payload = row

        if best_payload is None or best_score <= 0:
            break

        selected_ids.add(best_id)
        selected_rows.append(best_payload)
        facility = str(best_payload.facility_id)
        for ts in best_payload.coverage_hours:
            t = pd.Timestamp(ts)
            key = (facility, t)
            provider_hour_used.add((str(best_payload.provider_id), t))
            if key in remaining:
                if str(best_payload.role) == "MD":
                    remaining[key]["cov_doc"] = min(remaining[key]["req_doc"], remaining[key]["cov_doc"] + 1.0)
                elif str(best_payload.role) == "NP":
                    remaining[key]["cov_np"] = min(remaining[key]["req_np"], remaining[key]["cov_np"] + 1.0)
                elif str(best_payload.role) == "RN":
                    remaining[key]["cov_doc"] = min(remaining[key]["req_doc"], remaining[key]["cov_doc"] + 0.25)

    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        return selected_df, {"status": "success", "status_reason": "greedy_exhausted"}
    return selected_df, {"status": "success", "status_reason": "greedy_complete"}


def solve_staffing_plan(
    demand_profile: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    *,
    min_shift_hours: int = 4,
    doc_capacity_per_hour: float = 4.0,
    np_capacity_per_hour: float = 3.0,
    engine: str = "auto",
    shortfall_penalty: float = 1000.0,
    assignment_penalty: float = 0.45,
    overtime_penalty: float = 0.12,
    time_limit: float | None = None,
    target_utilization_floor: float = 0.55,
    max_provider_hours_change_share: float = 0.10,
    normal_provider_hour_budget: float | None = None,
    emergency_hour_bank: float = 0.0,
    overflow_role_policy: str = "hospitalist_md_equiv",
) -> tuple[pd.DataFrame, dict, str]:
    if demand_profile.empty:
        return pd.DataFrame(), {"status": "empty_demand", "status_reason": "empty_demand"}, "empty"

    if {"required_docs_hour", "required_nps_hour"}.issubset(demand_profile.columns) is False:
        raise ValueError("demand_profile must contain required_docs_hour and required_nps_hour")

    demand_profile = demand_profile.copy()
    demand_profile = demand_profile[
        (demand_profile["required_docs_hour"] > 0)
        | (demand_profile["required_nps_hour"] > 0)
    ].reset_index(drop=True)
    if demand_profile.empty:
        return pd.DataFrame(), {"status": "no_required_staff", "status_reason": "no_required_staff"}, "empty"

    if candidate_pool.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "facility_id",
                "provider_id",
                "role",
                "shift_start",
                "shift_end",
                "shift_hours",
                "coverage_hours",
                "blocked_hours",
                "availability_ratio",
            ]
        ), {
            "status": "infeasible",
            "status_reason": "no_candidates",
            "expected_shortfall_hours": float(demand_profile["required_docs_hour"].sum() + demand_profile["required_nps_hour"].sum()),
            "hard_violation_count": int(len(demand_profile)),
            "gaps": [],
            "utilization": 0.0,
            "sla_risk_reduction_proxy": 0.0,
            "overtime_proxy": 0.0,
            "shortage_hours": float(demand_profile["required_docs_hour"].sum() + demand_profile["required_nps_hour"].sum()),
            "blocked_availability_hours": 0,
        }, "empty"

    engine = str(engine).lower()
    t0 = time.perf_counter()
    if engine in {"milp", "auto"} and pulp is not None:
        selected, kpi = _solve_with_milp(
            demand_profile,
            candidate_pool,
            min_shift_hours=min_shift_hours,
            shortfall_penalty=shortfall_penalty,
            shift_penalty=assignment_penalty,
            overtime_penalty=overtime_penalty,
            time_limit=time_limit,
            normal_provider_hour_budget=normal_provider_hour_budget,
            emergency_hour_bank=emergency_hour_bank,
            overflow_role_policy=overflow_role_policy,
        )
        if kpi.get("status") in {"optimal", "feasible", "suboptimal"}:
            kpi["solve_seconds"] = float(time.perf_counter() - t0)
            kpi.update(_plan_kpis(demand_profile, selected, min_shift_hours))
            kpi["requested_engine"] = engine
            kpi["used_engine"] = "milp"
            return selected, kpi, "milp"
        if kpi.get("status") == "skipped":
            engine = "greedy"

    selected, kpi = _solve_with_greedy(
        demand_profile,
        candidate_pool,
        min_shift_hours=min_shift_hours,
        shortfall_penalty=shortfall_penalty,
        shift_penalty=assignment_penalty,
        overtime_penalty=overtime_penalty,
        time_limit=time_limit,
    )
    kpi.update(_plan_kpis(demand_profile, selected, min_shift_hours))
    kpi["solve_seconds"] = float(time.perf_counter() - t0)
    kpi["requested_engine"] = engine if engine != "milp" else "greedy"
    kpi["used_engine"] = "greedy"
    return selected, kpi, "greedy"


def summarize_plan_against_baseline(
    demand_profile: pd.DataFrame,
    plan_summary: dict,
    provider_staff: pd.DataFrame | None = None,
    plan_assignments: pd.DataFrame | None = None,
) -> dict:
    coverage_summary = _assignment_coverage_hours(plan_assignments) if plan_assignments is not None else {
        "recommended_provider_hours": 0.0,
        "recommended_doc_hours": 0.0,
        "recommended_np_hours": 0.0,
        "recommended_rn_hours": 0.0,
        "recommended_shift_count": 0,
    }
    if "recommended_shift_count" not in plan_summary:
        plan_summary["recommended_shift_count"] = coverage_summary["recommended_shift_count"]
    if "recommended_provider_hours" not in plan_summary:
        plan_summary["recommended_provider_hours"] = coverage_summary["recommended_provider_hours"]
    if "recommended_doc_hours" not in plan_summary:
        plan_summary["recommended_doc_hours"] = coverage_summary["recommended_doc_hours"]
    if "recommended_np_hours" not in plan_summary:
        plan_summary["recommended_np_hours"] = coverage_summary["recommended_np_hours"]
    if "recommended_rn_hours" not in plan_summary:
        plan_summary["recommended_rn_hours"] = coverage_summary["recommended_rn_hours"]
    for recovery_key in ["float_pool_hours", "on_call_hours", "overtime_extension_hours", "overflow_provider_hours", "normal_provider_hours"]:
        if recovery_key not in plan_summary:
            plan_summary[recovery_key] = float(coverage_summary.get(recovery_key, 0.0))

    baseline_provider_hours = 0.0
    if provider_staff is None or provider_staff.empty:
        baseline_shortfall = float(
            demand_profile["required_docs_hour"].sum() + demand_profile["required_nps_hour"].sum()
        )
    else:
        base = provider_staff.copy()
        base["timestamp"] = _coerce_ts(base["timestamp"])
        if "current_docs_on_duty" not in base.columns and "current_md_on_duty" in base.columns:
            base["current_docs_on_duty"] = base["current_md_on_duty"]
        if "current_nps_on_duty" not in base.columns and "current_np_on_duty" in base.columns:
            base["current_nps_on_duty"] = base["current_np_on_duty"]
        merged = demand_profile[["facility_id", "timestamp", "required_docs_hour", "required_nps_hour"]].merge(
            base[["facility_id", "timestamp", "current_docs_on_duty", "current_nps_on_duty"]].rename(
                columns={"current_docs_on_duty": "base_docs", "current_nps_on_duty": "base_nps"}
            ),
            on=["facility_id", "timestamp"],
            how="left",
        )
        merged["base_docs"] = _coerce_float(merged.get("base_docs", pd.Series(0.0, index=merged.index)))
        merged["base_nps"] = _coerce_float(merged.get("base_nps", pd.Series(0.0, index=merged.index)))
        baseline_shortfall = float(
            ((merged["required_docs_hour"] - merged["base_docs"]).clip(lower=0).fillna(0.0)
             + (merged["required_nps_hour"] - merged["base_nps"]).clip(lower=0).fillna(0.0)).sum()
        )
        baseline_provider_hours = float((merged["base_docs"] + merged["base_nps"]).sum())

    plan_summary["baseline_shortfall_hours"] = float(baseline_shortfall)
    plan_summary["sla_risk_reduction_proxy"] = float(plan_summary.get("baseline_shortfall_hours", 0.0) - plan_summary.get("shortage_hours", 0.0))
    denom = max(1.0, plan_summary["baseline_shortfall_hours"])
    plan_summary["shortage_reduction_share"] = float(plan_summary["sla_risk_reduction_proxy"] / denom)
    plan_summary["baseline_provider_hours"] = float(baseline_provider_hours)
    recommended = coverage_summary["recommended_provider_hours"]
    plan_summary["recommended_provider_hours"] = float(recommended)
    plan_summary["recommended_doc_hours"] = float(coverage_summary["recommended_doc_hours"])
    plan_summary["recommended_np_hours"] = float(coverage_summary["recommended_np_hours"])
    plan_summary["recommended_rn_hours"] = float(coverage_summary["recommended_rn_hours"])
    plan_summary["normal_provider_hours"] = float(coverage_summary.get("normal_provider_hours", 0.0))
    plan_summary["overflow_provider_hours"] = float(coverage_summary.get("overflow_provider_hours", 0.0))
    plan_summary["float_pool_hours"] = float(coverage_summary.get("float_pool_hours", 0.0))
    plan_summary["on_call_hours"] = float(coverage_summary.get("on_call_hours", 0.0))
    plan_summary["overtime_extension_hours"] = float(coverage_summary.get("overtime_extension_hours", 0.0))
    if baseline_provider_hours > 0:
        plan_summary["adoption_rate_vs_static_hours"] = float(recommended / max(1.0, baseline_provider_hours))
        plan_summary["provider_hours_saved"] = float(max(0.0, baseline_provider_hours - recommended))
        plan_summary["provider_hours_added"] = float(max(0.0, recommended - baseline_provider_hours))
    else:
        plan_summary["adoption_rate_vs_static_hours"] = 0.0
        plan_summary["provider_hours_saved"] = float(max(0.0, -recommended))
        plan_summary["provider_hours_added"] = float(max(0.0, recommended))

    return plan_summary
