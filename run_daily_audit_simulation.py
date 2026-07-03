from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tabfm_healthcare_eval.optimizer import (
    build_demand_profile,
    build_shift_candidate_pool,
    diagnose_staffing_feasibility,
    solve_staffing_plan,
    summarize_plan_against_baseline,
)
from scripts.show_top_tables import print_top_for_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily scheduling audit across multiple horizon days")
    parser.add_argument("--forecast-csv", required=True)
    parser.add_argument("--actual-csv", required=True)
    parser.add_argument("--provider-shifts", required=True, help="Provider shifts candidate source")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--audit-days", type=int, default=14)
    parser.add_argument("--plan-horizon-hours", type=int, default=24, help="Hours to optimize each day")
    parser.add_argument("--optimizer", choices=["auto", "milp", "greedy"], default="auto")
    parser.add_argument("--shift-hours", type=int, default=4)
    parser.add_argument("--min-shift-hours", type=int, default=4)
    parser.add_argument("--doc-capacity-per-hour", type=float, default=4.0)
    parser.add_argument("--np-capacity-per-hour", type=float, default=3.0)
    parser.add_argument("--print-top-n", type=int, default=5)
    parser.add_argument("--min-coverage-share", type=float, default=0.85)
    parser.add_argument("--max-shortage-increase-share", type=float, default=0.0)
    parser.add_argument("--target-utilization-floor", type=float, default=0.55)
    parser.add_argument("--max-provider-hours-change-share", type=float, default=0.10)
    parser.add_argument("--allow-float-pool", action="store_true", default=True)
    parser.add_argument("--no-float-pool", dest="allow_float_pool", action="store_false")
    parser.add_argument("--allow-overtime-extensions", action="store_true", default=True)
    parser.add_argument("--no-overtime-extensions", dest="allow_overtime_extensions", action="store_false")
    parser.add_argument("--monthly-provider-hour-budget-mode", choices=["baseline", "explicit"], default="baseline")
    parser.add_argument("--monthly-provider-hour-budget", type=float, default=None)
    parser.add_argument("--emergency-hour-bank-share", type=float, default=0.05)
    parser.add_argument("--overflow-role-policy", choices=["disabled", "hospitalist_md_equiv"], default="hospitalist_md_equiv")
    parser.add_argument("--target-utilization-lift", type=float, default=0.05)
    parser.add_argument("--max-utilization-target", type=float, default=0.60)
    parser.add_argument("--closed-loop", action="store_true", default=False, help="Enable closed-loop planning mode")
    parser.add_argument("--open-loop", dest="closed_loop", action="store_false", help="Enable open-loop planning mode")
    parser.add_argument(
        "--audit-mode",
        choices=["open_loop", "closed_loop"],
        default="open_loop",
        help="Planning mode selector. If set to closed_loop, daily planning can be informed by realized demand in the same day.",
    )
    parser.set_defaults(closed_loop=False)
    return parser.parse_args()


def _load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def run_daily_audit(
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
    provider_df: pd.DataFrame,
    out_dir: Path,
    audit_days: int,
    optimizer: str,
    shift_hours: int,
    min_shift_hours: int,
    plan_horizon_hours: int,
    doc_capacity_per_hour: float,
    np_capacity_per_hour: float,
    print_top_n: int = 5,
    closed_loop: bool = False,
    min_coverage_share: float = 0.85,
    max_shortage_increase_share: float = 0.0,
    target_utilization_floor: float = 0.55,
    max_provider_hours_change_share: float = 0.10,
    allow_float_pool: bool = True,
    allow_overtime_extensions: bool = True,
    monthly_provider_hour_budget_mode: str = "baseline",
    monthly_provider_hour_budget: float | None = None,
    emergency_hour_bank_share: float = 0.05,
    overflow_role_policy: str = "hospitalist_md_equiv",
    target_utilization_lift: float = 0.05,
    max_utilization_target: float = 0.60,
) -> None:
    forecast = forecast.copy()
    actual = actual.copy()
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
    actual["timestamp"] = pd.to_datetime(actual["timestamp"])

    if forecast.empty:
        raise ValueError("forecast is empty")

    facility_count = max(1, int(forecast["facility_id"].nunique()))
    day_counts = forecast.groupby(forecast["timestamp"].dt.floor("D")).size()
    min_full_day_rows = max(1, facility_count * 12)
    days = [d for d, n in day_counts.sort_index().items() if int(n) >= min_full_day_rows][: int(audit_days)]
    if not days:
        raise ValueError("forecast has no day buckets")

    baseline_monthly_budget = 0.0
    if monthly_provider_hour_budget_mode == "explicit" and monthly_provider_hour_budget is not None:
        baseline_monthly_budget = float(monthly_provider_hour_budget)
    else:
        baseline_source = actual.copy()
        for c in ["current_docs_on_duty", "current_nps_on_duty"]:
            if c not in baseline_source.columns:
                baseline_source[c] = 0.0
        if days:
            start_budget = pd.Timestamp(days[0])
            end_budget = pd.Timestamp(days[min(len(days), int(audit_days)) - 1]) + pd.Timedelta(days=1)
            baseline_source = baseline_source[(baseline_source["timestamp"] >= start_budget) & (baseline_source["timestamp"] < end_budget)]
        baseline_monthly_budget = float(
            pd.to_numeric(baseline_source["current_docs_on_duty"], errors="coerce").fillna(0).sum()
            + pd.to_numeric(baseline_source["current_nps_on_duty"], errors="coerce").fillna(0).sum()
        )
    emergency_hour_bank_total = float(max(0.0, baseline_monthly_budget * float(emergency_hour_bank_share)))
    normal_budget_used_so_far = 0.0
    emergency_bank_used_so_far = 0.0

    daily_rows: list[dict] = []
    all_assignments = []
    for day_idx, day_start in enumerate(days):
        day_start = pd.Timestamp(day_start)
        effective_horizon_hours = int(max(1, min(int(plan_horizon_hours), 24)))
        day_end = day_start + pd.Timedelta(hours=effective_horizon_hours)
        if day_end <= day_start:
            day_end = day_start + pd.Timedelta(days=1)
        day_forecast = forecast[(forecast["timestamp"] >= day_start) & (forecast["timestamp"] < day_end)].copy()
        day_actual = actual[(actual["timestamp"] >= day_start) & (actual["timestamp"] < day_end)].copy()
        if day_forecast.empty:
            continue

        if closed_loop and not day_actual.empty:
            # In closed-loop mode, use the latest known realized arrivals for the same
            # day as the planning signal. This keeps the loop lightweight while
            # making the signal data-driven from realized operations.
            realized_col = "expected_arrivals_next_hour"
            if realized_col in day_actual.columns and realized_col in day_forecast.columns:
                day_forecast["y_pred"] = day_actual[realized_col].values

        demand_profile = build_demand_profile(
            day_forecast,
            shift_hours=shift_hours,
            required_horizon_start=day_start,
            required_horizon_end=day_end,
            doc_capacity_per_hour=doc_capacity_per_hour,
            np_capacity_per_hour=np_capacity_per_hour,
        )
        if demand_profile.empty:
            continue

        # closed-loop path can refit model/forecast later; for this synthetic harness we retain
        # the forecast snapshot for both open-loop and closed-loop modes.
        if closed_loop and optimizer in {"auto", "greedy", "milp"}:
            pass

        candidates = build_shift_candidate_pool(
            provider_df,
            min_shift_hours=min_shift_hours,
            required_horizon_start=day_start,
            required_horizon_end=day_end,
            allow_float_pool=allow_float_pool,
            allow_overtime_extensions=allow_overtime_extensions,
        )

        remaining_normal_budget = max(0.0, baseline_monthly_budget - normal_budget_used_so_far)
        remaining_emergency_bank = max(0.0, emergency_hour_bank_total - emergency_bank_used_so_far)
        if overflow_role_policy == "disabled" and not candidates.empty:
            candidates = candidates[candidates["role"].astype(str).str.upper() != "HOSPITALIST_OVERFLOW"].copy()
        assignments, summary, used_engine = solve_staffing_plan(
            demand_profile=demand_profile,
            candidate_pool=candidates,
            min_shift_hours=min_shift_hours,
            doc_capacity_per_hour=doc_capacity_per_hour,
            np_capacity_per_hour=np_capacity_per_hour,
            engine=optimizer,
            target_utilization_floor=target_utilization_floor,
            max_provider_hours_change_share=max_provider_hours_change_share,
            normal_provider_hour_budget=remaining_normal_budget,
            emergency_hour_bank=remaining_emergency_bank,
            overflow_role_policy=overflow_role_policy,
        )
        summary = summarize_plan_against_baseline(
            demand_profile,
            summary,
            provider_staff=day_actual,
            plan_assignments=assignments,
        )
        feasibility_detail, feasibility_summary = diagnose_staffing_feasibility(
            demand_profile,
            candidates,
            assignments=assignments,
        )
        summary["demand_rows"] = int(len(demand_profile))
        summary["candidate_count"] = int(len(candidates))
        summary["assignment_count"] = int(len(assignments))
        summary["facility_count"] = int(day_forecast["facility_id"].nunique())
        summary["date"] = str(day_start.date())
        summary["day_index"] = int(day_idx)
        summary["optimizer"] = optimizer
        summary["used_engine"] = used_engine
        summary["plan_horizon_hours"] = int(plan_horizon_hours)
        summary["effective_audit_horizon_hours"] = int(effective_horizon_hours)
        normal_budget_used_so_far += float(summary.get("normal_provider_hours", 0.0))
        emergency_bank_used_so_far += float(summary.get("overflow_provider_hours", 0.0))
        budget_exceeded = bool(normal_budget_used_so_far > baseline_monthly_budget + 1e-6 or emergency_bank_used_so_far > emergency_hour_bank_total + 1e-6)
        summary["monthly_provider_hour_budget"] = float(baseline_monthly_budget)
        summary["normal_budget_used_to_date"] = float(normal_budget_used_so_far)
        summary["normal_budget_remaining_to_date"] = float(max(0.0, baseline_monthly_budget - normal_budget_used_so_far))
        summary["emergency_hour_bank_total"] = float(emergency_hour_bank_total)
        summary["emergency_bank_used_to_date"] = float(emergency_bank_used_so_far)
        summary["emergency_bank_remaining_to_date"] = float(max(0.0, emergency_hour_bank_total - emergency_bank_used_so_far))
        summary["emergency_bank_used_share"] = float(emergency_bank_used_so_far / max(1.0, emergency_hour_bank_total))
        summary["budget_exceeded"] = budget_exceeded
        summary["unfunded_hours_requested"] = float(max(0.0, summary.get("recommended_provider_hours", 0.0) - remaining_normal_budget - remaining_emergency_bank))
        summary["blocked_availability_hours"] = int(summary.get("blocked_availability_hours", 0))
        summary["shortage_reduction_vs_baseline"] = float(summary.get("sla_risk_reduction_proxy", 0.0))
        summary["feasible_coverage_share"] = float(feasibility_summary.get("feasible_coverage_share", 0.0))
        summary["infeasible_shortage_hours"] = float(feasibility_summary.get("infeasible_shortage_hours", 0.0))
        summary["avoidable_shortage_hours"] = float(feasibility_summary.get("avoidable_shortage_hours", 0.0))
        summary["target_utilization_floor"] = float(target_utilization_floor)
        summary["max_provider_hours_change_share"] = float(max_provider_hours_change_share)
        summary["allow_float_pool"] = bool(allow_float_pool)
        summary["allow_overtime_extensions"] = bool(allow_overtime_extensions)
        if "baseline_shortfall_hours" in summary and "shortage_hours" in summary:
            summary["sla_risk_reduction_share"] = float(
                summary["sla_risk_reduction_proxy"] / max(1.0, float(summary["baseline_shortfall_hours"]))
            )
        else:
            summary["sla_risk_reduction_share"] = 0.0
        summary["coverage_gate_ok"] = bool(float(summary.get("utilization", 0.0)) >= float(min_coverage_share))
        summary["utilization_gate_ok"] = bool(float(summary.get("utilization", 0.0)) >= float(target_utilization_floor) and float(summary.get("utilization", 0.0)) <= float(max_utilization_target))
        summary["budget_gate_ok"] = bool(not summary.get("budget_exceeded", False))
        summary["shortage_gate_ok"] = bool(
            float(summary.get("shortage_reduction_share", 0.0)) >= -float(max_shortage_increase_share)
        )
        summary["safety_gate_ok"] = bool(summary["coverage_gate_ok"] and summary["shortage_gate_ok"] and summary["budget_gate_ok"])

        if not assignments.empty:
            assign_day = assignments.copy()
            assign_day["audit_day"] = str(day_start.date())
            all_assignments.append(assign_day)

        daily_rows.append(
            {
                "date": summary["date"],
                "shortage_hours": float(summary.get("shortage_hours", 0.0)),
                "hard_violation_count": int(summary.get("hard_violation_count", 0)),
                "blocked_availability_hours": int(summary.get("blocked_availability_hours", 0)),
                "baseline_provider_hours": float(summary.get("baseline_provider_hours", 0.0)),
                "recommended_provider_hours": float(summary.get("recommended_provider_hours", 0.0)),
                "recommended_doc_hours": float(summary.get("recommended_doc_hours", 0.0)),
                "recommended_np_hours": float(summary.get("recommended_np_hours", 0.0)),
                "float_pool_hours": float(summary.get("float_pool_hours", 0.0)),
                "on_call_hours": float(summary.get("on_call_hours", 0.0)),
                "overtime_extension_hours": float(summary.get("overtime_extension_hours", 0.0)),
                "provider_hours_saved": float(summary.get("provider_hours_saved", 0.0)),
                "provider_hours_added": float(summary.get("provider_hours_added", 0.0)),
                "normal_provider_hours": float(summary.get("normal_provider_hours", 0.0)),
                "overflow_provider_hours": float(summary.get("overflow_provider_hours", 0.0)),
                "monthly_provider_hour_budget": float(summary.get("monthly_provider_hour_budget", 0.0)),
                "normal_budget_used_to_date": float(summary.get("normal_budget_used_to_date", 0.0)),
                "normal_budget_remaining_to_date": float(summary.get("normal_budget_remaining_to_date", 0.0)),
                "emergency_hour_bank_total": float(summary.get("emergency_hour_bank_total", 0.0)),
                "emergency_bank_used_to_date": float(summary.get("emergency_bank_used_to_date", 0.0)),
                "emergency_bank_remaining_to_date": float(summary.get("emergency_bank_remaining_to_date", 0.0)),
                "emergency_bank_used_share": float(summary.get("emergency_bank_used_share", 0.0)),
                "budget_exceeded": bool(summary.get("budget_exceeded", False)),
                "budget_gate_ok": bool(summary.get("budget_gate_ok", False)),
                "unfunded_hours_requested": float(summary.get("unfunded_hours_requested", 0.0)),
                "adoption_rate_vs_static_hours": float(summary.get("adoption_rate_vs_static_hours", 0.0)),
                "sla_risk_reduction_proxy": float(summary.get("sla_risk_reduction_proxy", 0.0)),
                "overtime_proxy": float(summary.get("overtime_proxy", 0.0)),
                "utilization": float(summary.get("utilization", 0.0)),
                "baseline_shortfall_hours": float(summary.get("baseline_shortfall_hours", 0.0)),
                "infeasible_shortage_hours": float(summary.get("infeasible_shortage_hours", 0.0)),
                "avoidable_shortage_hours": float(summary.get("avoidable_shortage_hours", 0.0)),
                "feasible_coverage_share": float(summary.get("feasible_coverage_share", 0.0)),
                "assignment_count": int(summary.get("assignment_count", 0)),
                "demand_rows": int(summary.get("demand_rows", 0)),
                "optimizer": optimizer,
                "used_engine": used_engine,
                "solve_seconds": float(summary.get("solve_seconds", 0.0)),
                "plan_horizon_hours": int(summary.get("plan_horizon_hours", 0)),
                "effective_audit_horizon_hours": int(summary.get("effective_audit_horizon_hours", 0)),
                "coverage_gate_ok": bool(summary.get("coverage_gate_ok", False)),
                "utilization_gate_ok": bool(summary.get("utilization_gate_ok", False)),
                "shortage_gate_ok": bool(summary.get("shortage_gate_ok", False)),
                "safety_gate_ok": bool(summary.get("safety_gate_ok", False)),
                "shortage_reduction_share": float(summary.get("shortage_reduction_share", 0.0)),
            }
        )
        print(
            f"DAY={day_start.date()} shortfall={summary.get('shortage_hours', 0.0):.3f} "
            f"violations={summary.get('hard_violation_count', 0)} assignments={summary.get('assignment_count', 0)} "
            f"engine={summary.get('used_engine')}"
        )

    daily_df = pd.DataFrame(daily_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "audit_days_requested": int(audit_days),
        "days_with_data": int(len(daily_df)),
        "optimizer": optimizer,
        "used_engine": used_engine if "used_engine" in locals() else optimizer,
        "plan_rows_total": int(sum(d.get("assignment_count", 0) for d in daily_rows)),
        "plan_provider_hours_total": float(daily_df["recommended_provider_hours"].sum()) if not daily_df.empty else 0.0,
        "baseline_provider_hours_total": float(daily_df["baseline_provider_hours"].sum()) if not daily_df.empty else 0.0,
        "provider_hours_saved_total": float(daily_df["provider_hours_saved"].sum()) if not daily_df.empty else 0.0,
        "provider_hours_added_total": float(max(0.0, (daily_df["normal_provider_hours"].sum() if not daily_df.empty and "normal_provider_hours" in daily_df else 0.0) - baseline_monthly_budget)),
        "provider_hours_added_outside_budget_total": float(max(0.0, (daily_df["normal_provider_hours"].sum() if not daily_df.empty and "normal_provider_hours" in daily_df else 0.0) - baseline_monthly_budget)),
        "normal_provider_hours_used_total": float(daily_df["normal_provider_hours"].sum()) if not daily_df.empty and "normal_provider_hours" in daily_df else 0.0,
        "overflow_provider_hours_total": float(daily_df["overflow_provider_hours"].sum()) if not daily_df.empty and "overflow_provider_hours" in daily_df else 0.0,
        "monthly_provider_hour_budget": float(baseline_monthly_budget),
        "emergency_hour_bank_total": float(emergency_hour_bank_total),
        "emergency_bank_used_total": float(emergency_bank_used_so_far),
        "emergency_bank_remaining_total": float(max(0.0, emergency_hour_bank_total - emergency_bank_used_so_far)),
        "emergency_bank_used_share": float(emergency_bank_used_so_far / max(1.0, emergency_hour_bank_total)),
        "budget_exceeded": bool((normal_budget_used_so_far > baseline_monthly_budget + 1e-6) or (emergency_bank_used_so_far > emergency_hour_bank_total + 1e-6)),
        "mean_adoption_rate_vs_static_hours": float(daily_df["adoption_rate_vs_static_hours"].mean()) if not daily_df.empty else 0.0,
        "total_shortage_hours": float(daily_df["shortage_hours"].sum()) if not daily_df.empty else 0.0,
        "total_hard_violations": int(daily_df["hard_violation_count"].sum()) if not daily_df.empty else 0,
        "mean_shortage_reduction_proxy": float(daily_df["sla_risk_reduction_proxy"].mean()) if not daily_df.empty else 0.0,
        "mean_utilization": float(daily_df["utilization"].mean()) if not daily_df.empty else 0.0,
        "mean_feasible_coverage_share": float(daily_df["feasible_coverage_share"].mean()) if not daily_df.empty else 0.0,
        "total_infeasible_shortage_hours": float(daily_df["infeasible_shortage_hours"].sum()) if not daily_df.empty else 0.0,
        "total_avoidable_shortage_hours": float(daily_df["avoidable_shortage_hours"].sum()) if not daily_df.empty else 0.0,
        "float_pool_hours_total": float(daily_df["float_pool_hours"].sum()) if not daily_df.empty else 0.0,
        "on_call_hours_total": float(daily_df["on_call_hours"].sum()) if not daily_df.empty else 0.0,
        "overtime_extension_hours_total": float(daily_df["overtime_extension_hours"].sum()) if not daily_df.empty else 0.0,
        "min_coverage_share": float(min_coverage_share),
        "max_shortage_increase_share": float(max_shortage_increase_share),
        "target_utilization_floor": float(target_utilization_floor),
        "max_provider_hours_change_share": float(max_provider_hours_change_share),
        "utilization_gate_pass_rate": float(daily_df["utilization_gate_ok"].mean()) if not daily_df.empty and "utilization_gate_ok" in daily_df else 0.0,
        "days_passing_safety_gate": int(daily_df["safety_gate_ok"].sum()) if not daily_df.empty and "safety_gate_ok" in daily_df else 0,
        "safety_gate_pass_rate": float(daily_df["safety_gate_ok"].mean()) if not daily_df.empty and "safety_gate_ok" in daily_df else 0.0,
        "budget_gate_pass_rate": float(daily_df["budget_gate_ok"].mean()) if not daily_df.empty and "budget_gate_ok" in daily_df else 0.0,
        "overall_budget_gate_ok": bool(daily_df["budget_gate_ok"].all()) if not daily_df.empty and "budget_gate_ok" in daily_df else False,
        "overall_safety_gate_ok": bool(daily_df["safety_gate_ok"].all()) if not daily_df.empty and "safety_gate_ok" in daily_df else False,
        "target_utilization_lift": float(target_utilization_lift),
        "max_utilization_target": float(max_utilization_target),
        "overflow_role_policy": str(overflow_role_policy),
        "plan_horizon_hours_requested": int(plan_horizon_hours),
        "effective_audit_horizon_hours": int(max(1, min(int(plan_horizon_hours), 24))),
    }
    summary_path = out_dir / "daily_audit_summary.json"
    detail_path = out_dir / "daily_audit_detail.csv"
    detail_jsonl_path = out_dir / "daily_audit_rows.jsonl"
    assignments_path = out_dir / "daily_audit_assignments.csv"
    feasibility_summary_path = out_dir / "feasibility_summary.json"
    feasibility_detail_path = out_dir / "feasibility_detail.csv"

    daily_df.to_csv(detail_path, index=False)
    if not daily_df.empty:
        daily_df.to_json(detail_jsonl_path, orient="records", lines=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(feasibility_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_infeasible_shortage_hours": summary["total_infeasible_shortage_hours"],
                "total_avoidable_shortage_hours": summary["total_avoidable_shortage_hours"],
                "mean_feasible_coverage_share": summary["mean_feasible_coverage_share"],
            },
            f,
            indent=2,
        )
    if not daily_df.empty:
        daily_df[[
            "date",
            "infeasible_shortage_hours",
            "avoidable_shortage_hours",
            "feasible_coverage_share",
            "shortage_hours",
            "utilization",
        ]].to_csv(feasibility_detail_path, index=False)
    if all_assignments:
        pd.concat(all_assignments, ignore_index=True).to_csv(assignments_path, index=False)

    print("Daily audit summary:")
    print(json.dumps(summary, indent=2))
    print_top_for_paths(
        [str(assignments_path), str(detail_path), str(detail_jsonl_path), str(summary_path)],
        top_n=print_top_n,
    )
    print_top_for_paths([str(feasibility_detail_path), str(feasibility_summary_path)], top_n=print_top_n)


def main() -> None:
    args = parse_args()
    if args.audit_mode == "closed_loop":
        args.closed_loop = True
    elif args.audit_mode == "open_loop":
        args.closed_loop = False
    forecast = _load_table(args.forecast_csv)
    actual = _load_table(args.actual_csv)
    provider_df = _load_table(args.provider_shifts)

    out_dir = Path(args.out)
    run_daily_audit(
        forecast=forecast,
        actual=actual,
        provider_df=provider_df,
        out_dir=out_dir,
        audit_days=args.audit_days,
        optimizer=args.optimizer,
        shift_hours=args.shift_hours,
        min_shift_hours=args.min_shift_hours,
        plan_horizon_hours=args.plan_horizon_hours,
        doc_capacity_per_hour=args.doc_capacity_per_hour,
        np_capacity_per_hour=args.np_capacity_per_hour,
        print_top_n=args.print_top_n,
        closed_loop=args.closed_loop,
        min_coverage_share=args.min_coverage_share,
        max_shortage_increase_share=args.max_shortage_increase_share,
        target_utilization_floor=args.target_utilization_floor,
        max_provider_hours_change_share=args.max_provider_hours_change_share,
        allow_float_pool=args.allow_float_pool,
        allow_overtime_extensions=args.allow_overtime_extensions,
        monthly_provider_hour_budget_mode=args.monthly_provider_hour_budget_mode,
        monthly_provider_hour_budget=args.monthly_provider_hour_budget,
        emergency_hour_bank_share=args.emergency_hour_bank_share,
        overflow_role_policy=args.overflow_role_policy,
        target_utilization_lift=args.target_utilization_lift,
        max_utilization_target=args.max_utilization_target,
    )


if __name__ == "__main__":
    main()
