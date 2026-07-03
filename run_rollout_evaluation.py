from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_daily_audit_simulation import run_daily_audit
from scripts.show_top_tables import print_top_for_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run phased rollout simulation: shadow -> canary -> full using precomputed forecast "
            "and provider shifts. Shadow keeps recommendations for audit. Canary and full simulate "
            "partial vs full adoption by limiting provider pool."
        )
    )
    parser.add_argument("--forecast-csv", required=True)
    parser.add_argument("--actual-csv", required=True)
    parser.add_argument("--provider-shifts", required=True)
    parser.add_argument("--out", default="outputs/rollout")
    parser.add_argument("--audit-days", type=int, default=30)
    parser.add_argument("--plan-horizon-hours", type=int, default=24)
    parser.add_argument("--optimizer", choices=["auto", "milp", "greedy"], default="auto")
    parser.add_argument("--shift-hours", type=int, default=4)
    parser.add_argument("--min-shift-hours", type=int, default=4)
    parser.add_argument("--doc-capacity-per-hour", type=float, default=4.0)
    parser.add_argument("--np-capacity-per-hour", type=float, default=3.0)
    parser.add_argument("--canary-share", type=float, default=0.30)
    parser.add_argument("--canary-min-providers", type=int, default=1)
    parser.add_argument("--phase-canary", action="store_true", default=True)
    parser.add_argument("--phase-full", action="store_true", default=True)
    parser.add_argument("--print-top-n", type=int, default=5)
    parser.add_argument("--shortage-increase-tolerance", type=float, default=0.10)
    parser.add_argument("--hard-violation-increase-tolerance", type=float, default=0.10)
    parser.add_argument("--min-adoption-rate", type=float, default=0.60)
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
    parser.add_argument("--stop-on-fail", action="store_true", default=False)
    parser.add_argument(
        "--audit-mode",
        choices=["open_loop", "closed_loop"],
        default="open_loop",
        help="closed_loop can be used if you later add real-time reforecasting for each day",
    )
    return parser.parse_args()


def _load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _get_provider_key_col(df: pd.DataFrame) -> str:
    if "base_provider_id" in df.columns:
        return "base_provider_id"
    return "provider_id"


def _pick_canary_providers(df: pd.DataFrame, share: float, min_providers: int) -> list[str]:
    if df.empty:
        return []
    if not (0.0 < share <= 1.0):
        share = max(0.0, min(1.0, share))

    key_col = _get_provider_key_col(df)
    provider_ids = pd.Series(df[key_col].astype(str).tolist())
    selected = []

    # Facility-level fairness: split proportionally by facility.
    for facility in sorted(df["facility_id"].dropna().astype(str).unique().tolist()):
        fac_provider_ids = provider_ids[df["facility_id"].astype(str) == facility].drop_duplicates().tolist()
        if not fac_provider_ids:
            continue
        fac_target = max(int(min_providers), int(round(len(fac_provider_ids) * share)))
        fac_target = max(min_providers, fac_target) if share > 0 else 0
        fac_target = min(len(fac_provider_ids), fac_target)
        selected.extend(sorted(fac_provider_ids)[:fac_target])

    # If share is small and we did not pick anything (e.g., one facility with 1 provider),
    # force at least one provider per facility.
    if share > 0 and not selected:
        selected = sorted(provider_ids.drop_duplicates().tolist())

    return selected


def _filter_shifts_for_providers(shifts_df: pd.DataFrame, providers: list[str]) -> pd.DataFrame:
    if not providers:
        return pd.DataFrame(columns=shifts_df.columns)
    key_col = _get_provider_key_col(shifts_df)
    return shifts_df[shifts_df[key_col].astype(str).isin(set(map(str, providers)))].copy()


def _run_phase(
    label: str,
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
    provider_shifts: pd.DataFrame,
    out_dir: Path,
    audit_days: int,
    optimizer: str,
    shift_hours: int,
    min_shift_hours: int,
    plan_horizon_hours: int,
    doc_cap: float,
    np_cap: float,
    print_top_n: int,
    closed_loop: bool,
    min_coverage_share: float,
    max_shortage_increase_share: float,
    target_utilization_floor: float,
    max_provider_hours_change_share: float,
    allow_float_pool: bool,
    allow_overtime_extensions: bool,
    monthly_provider_hour_budget_mode: str,
    monthly_provider_hour_budget: float | None,
    emergency_hour_bank_share: float,
    overflow_role_policy: str,
    target_utilization_lift: float,
    max_utilization_target: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    run_daily_audit(
        forecast=forecast,
        actual=actual,
        provider_df=provider_shifts,
        out_dir=out_dir,
        audit_days=audit_days,
        optimizer=optimizer,
        shift_hours=shift_hours,
        min_shift_hours=min_shift_hours,
        plan_horizon_hours=plan_horizon_hours,
        doc_capacity_per_hour=doc_cap,
        np_capacity_per_hour=np_cap,
        print_top_n=0,
        closed_loop=closed_loop,
        min_coverage_share=min_coverage_share,
        max_shortage_increase_share=max_shortage_increase_share,
        target_utilization_floor=target_utilization_floor,
        max_provider_hours_change_share=max_provider_hours_change_share,
        allow_float_pool=allow_float_pool,
        allow_overtime_extensions=allow_overtime_extensions,
        monthly_provider_hour_budget_mode=monthly_provider_hour_budget_mode,
        monthly_provider_hour_budget=monthly_provider_hour_budget,
        emergency_hour_bank_share=emergency_hour_bank_share,
        overflow_role_policy=overflow_role_policy,
        target_utilization_lift=target_utilization_lift,
        max_utilization_target=max_utilization_target,
    )

    summary_path = out_dir / "daily_audit_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    phase_payload = {
        "phase": label,
        "provider_scope": str(provider_shifts["facility_id"].nunique()) + " facility(s)",
        "candidate_rows": int(summary.get("plan_rows_total", 0)),
        "plan_provider_hours_total": float(summary.get("plan_provider_hours_total", 0.0)),
        "baseline_provider_hours_total": float(summary.get("baseline_provider_hours_total", 0.0)),
        "provider_hours_saved_total": float(summary.get("provider_hours_saved_total", 0.0)),
        "provider_hours_added_total": float(summary.get("provider_hours_added_total", 0.0)),
        "provider_hours_added_outside_budget_total": float(summary.get("provider_hours_added_outside_budget_total", 0.0)),
        "normal_provider_hours_used_total": float(summary.get("normal_provider_hours_used_total", 0.0)),
        "overflow_provider_hours_total": float(summary.get("overflow_provider_hours_total", 0.0)),
        "monthly_provider_hour_budget": float(summary.get("monthly_provider_hour_budget", 0.0)),
        "emergency_hour_bank_total": float(summary.get("emergency_hour_bank_total", 0.0)),
        "emergency_bank_used_total": float(summary.get("emergency_bank_used_total", 0.0)),
        "emergency_bank_used_share": float(summary.get("emergency_bank_used_share", 0.0)),
        "budget_exceeded": bool(summary.get("budget_exceeded", False)),
        "overall_budget_gate_ok": bool(summary.get("overall_budget_gate_ok", False)),
        "mean_adoption_rate_vs_static_hours": float(summary.get("mean_adoption_rate_vs_static_hours", 0.0)),
        "total_shortage_hours": float(summary.get("total_shortage_hours", 0.0)),
        "total_hard_violations": int(summary.get("total_hard_violations", 0)),
        "mean_shortage_reduction_proxy": float(summary.get("mean_shortage_reduction_proxy", 0.0)),
        "mean_utilization": float(summary.get("mean_utilization", 0.0)),
        "utilization_gate_pass_rate": float(summary.get("utilization_gate_pass_rate", 0.0)),
        "total_infeasible_shortage_hours": float(summary.get("total_infeasible_shortage_hours", 0.0)),
        "total_avoidable_shortage_hours": float(summary.get("total_avoidable_shortage_hours", 0.0)),
        "float_pool_hours_total": float(summary.get("float_pool_hours_total", 0.0)),
        "on_call_hours_total": float(summary.get("on_call_hours_total", 0.0)),
        "overtime_extension_hours_total": float(summary.get("overtime_extension_hours_total", 0.0)),
        "safety_gate_pass_rate": float(summary.get("safety_gate_pass_rate", 0.0)),
        "overall_safety_gate_ok": bool(summary.get("overall_safety_gate_ok", False)),
        "audit_days_with_data": int(summary.get("days_with_data", 0)),
        "optimizer": summary.get("used_engine", optimizer),
    }

    phase_payload["summary_path"] = str(summary_path)
    detail_path = out_dir / "daily_audit_detail.csv"
    assignment_path = out_dir / "daily_audit_assignments.csv"
    if assignment_path.exists():
        phase_payload["assignment_rows"] = len(_load_table(str(assignment_path)))
    if summary_path.exists():
        print(f"=== phase:{label} summary ===")
        print(json.dumps(summary, indent=2))
        print_top_for_paths(
            [
                str(summary_path),
                str(detail_path),
                str(assignment_path),
            ],
            top_n=print_top_n,
        )

    return phase_payload


def _evaluate_gate(prev: dict | None, current: dict, *, shortage_tol: float, hard_tol: float, adoption_min: float) -> dict:
    if prev is None:
        safety_gate_ok = bool(current.get("overall_safety_gate_ok", False))
        return {
            "shortage_gate_ok": True,
            "hard_gate_ok": True,
            "adoption_gate_ok": True,
            "safety_gate_ok": safety_gate_ok,
            "shortage_delta_vs_prev": 0.0,
            "hard_delta_vs_prev": 0.0,
            "blocked": [] if safety_gate_ok else [f"safety gate failed; pass rate {current.get('safety_gate_pass_rate', 0):.3f}"],
            "phase_ok": safety_gate_ok,
        }

    prev_shortage = float(prev.get("total_shortage_hours", 0.0))
    prev_hard = int(prev.get("total_hard_violations", 0))
    cur_shortage = float(current.get("total_shortage_hours", 0.0))
    cur_hard = int(current.get("total_hard_violations", 0))

    shortage_delta = 0.0 if prev_shortage <= 0 else (cur_shortage - prev_shortage) / prev_shortage
    hard_delta = 0.0 if prev_hard <= 0 else (cur_hard - prev_hard) / prev_hard
    shortage_gate_ok = shortage_delta <= shortage_tol
    hard_gate_ok = hard_delta <= hard_tol
    adoption_gate_ok = float(current.get("mean_adoption_rate_vs_static_hours", 0.0)) >= float(adoption_min)
    safety_gate_ok = bool(current.get("overall_safety_gate_ok", False))

    blocked: list[str] = []
    if not shortage_gate_ok:
        blocked.append(
            f"shortage degradation {shortage_delta:.3f} > tolerance {shortage_tol:.3f}"
        )
    if not hard_gate_ok:
        blocked.append(f"hard-violation degradation {hard_delta:.3f} > tolerance {hard_tol:.3f}")
    if not adoption_gate_ok:
        blocked.append(
            f"adoption {current.get('mean_adoption_rate_vs_static_hours', 0):.3f} < threshold {adoption_min:.3f}"
        )
    if not safety_gate_ok:
        blocked.append(
            f"safety gate failed; pass rate {current.get('safety_gate_pass_rate', 0):.3f}"
        )

    return {
        "shortage_gate_ok": bool(shortage_gate_ok),
        "hard_gate_ok": bool(hard_gate_ok),
        "adoption_gate_ok": bool(adoption_gate_ok),
        "safety_gate_ok": bool(safety_gate_ok),
        "shortage_delta_vs_prev": float(shortage_delta),
        "hard_delta_vs_prev": float(hard_delta),
        "blocked": blocked,
        "phase_ok": bool(shortage_gate_ok and hard_gate_ok and adoption_gate_ok and safety_gate_ok),
    }


def main() -> None:
    args = parse_args()
    forecast = _load_table(args.forecast_csv)
    actual = _load_table(args.actual_csv)
    provider_df = _load_table(args.provider_shifts)

    forecast = forecast.copy()
    actual = actual.copy()
    provider_df = provider_df.copy()

    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
    actual["timestamp"] = pd.to_datetime(actual["timestamp"])
    provider_df["shift_start"] = pd.to_datetime(provider_df["shift_start"])
    provider_df["shift_end"] = pd.to_datetime(provider_df["shift_end"])

    if provider_df.empty:
        raise ValueError("provider-shifts table is empty")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loaded rollout inputs")
    print_top_for_paths([args.forecast_csv, args.actual_csv, args.provider_shifts], top_n=args.print_top_n)

    closed_loop = args.audit_mode == "closed_loop"

    all_providers = sorted(provider_df[_get_provider_key_col(provider_df)].astype(str).drop_duplicates().tolist())
    canary_providers = _pick_canary_providers(provider_df, args.canary_share, args.canary_min_providers)
    canary_provider_set = set(canary_providers)

    phases: list[tuple[str, pd.DataFrame]] = [("shadow", provider_df)]
    if args.phase_canary and 0.0 < args.canary_share < 1.0:
        phases.append(("canary", _filter_shifts_for_providers(provider_df, canary_providers)))
    if args.phase_full:
        phases.append(("full", provider_df))

    phase_results: list[dict] = []
    prev_payload = None
    all_pass = True

    for label, provider_subset in phases:
        phase_out = out_root / f"phase_{label}"
        payload = _run_phase(
            label=label,
            forecast=forecast,
            actual=actual,
            provider_shifts=provider_subset,
            out_dir=phase_out,
            audit_days=args.audit_days,
            optimizer=args.optimizer,
            shift_hours=args.shift_hours,
            min_shift_hours=args.min_shift_hours,
            plan_horizon_hours=args.plan_horizon_hours,
            doc_cap=args.doc_capacity_per_hour,
            np_cap=args.np_capacity_per_hour,
            print_top_n=args.print_top_n,
            closed_loop=closed_loop,
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
        payload["provider_rows"] = int(len(provider_subset))
        payload["provider_unique_candidates"] = int(provider_subset[_get_provider_key_col(provider_subset)].nunique()) if not provider_subset.empty else 0

        if not canary_providers and label == "canary":
            payload["provider_selection_note"] = "none"
            payload["coverage_scope"] = "empty"

        if label == "canary":
            payload["coverage_scope"] = f"subset ({len(canary_provider_set)} base providers)"

        if label == "full":
            payload["coverage_scope"] = "all providers"
            payload["canary_provider_count"] = len(canary_provider_set)

        gate_report = _evaluate_gate(
            prev_payload,
            payload,
            shortage_tol=args.shortage_increase_tolerance,
            hard_tol=args.hard_violation_increase_tolerance,
            adoption_min=args.min_adoption_rate,
        )
        payload.update(gate_report)
        payload["status"] = "pass" if gate_report["phase_ok"] else "fail"

        phase_results.append(payload)
        prev_payload = payload

        if not payload["phase_ok"]:
            all_pass = False
            if args.stop_on_fail:
                break

    # Shadow has no gate baseline; mark explicit.
    for item in phase_results:
        if item["phase"] == "shadow":
            item["status"] = "reference"

    rollout_summary = {
        "rollout_mode": "shadow_canary_full",
        "optimizer": args.optimizer,
        "audit_days": int(args.audit_days),
        "plan_horizon_hours": int(args.plan_horizon_hours),
        "shift_hours": int(args.shift_hours),
        "min_shift_hours": int(args.min_shift_hours),
        "canary_share": float(args.canary_share),
        "all_pass": bool(all_pass),
        "phases": phase_results,
        "gates": {
            "shortage_increase_tolerance": float(args.shortage_increase_tolerance),
            "hard_violation_increase_tolerance": float(args.hard_violation_increase_tolerance),
            "min_adoption_rate": float(args.min_adoption_rate),
            "min_coverage_share": float(args.min_coverage_share),
            "max_shortage_increase_share": float(args.max_shortage_increase_share),
            "target_utilization_floor": float(args.target_utilization_floor),
            "max_provider_hours_change_share": float(args.max_provider_hours_change_share),
            "allow_float_pool": bool(args.allow_float_pool),
            "allow_overtime_extensions": bool(args.allow_overtime_extensions),
            "stop_on_fail": bool(args.stop_on_fail),
        },
    }
    rollout_summary_path = out_root / "rollout_summary.json"
    with open(rollout_summary_path, "w", encoding="utf-8") as f:
        json.dump(rollout_summary, f, indent=2)

    phase_paths = [str(out_root / f"phase_{item['phase']}") for item in phase_results]
    phase_summary_rows = pd.DataFrame(
        [
            {
                "phase": item.get("phase"),
                "status": item.get("status"),
                "provider_rows": item.get("provider_rows"),
                "provider_unique_candidates": item.get("provider_unique_candidates"),
                "total_shortage_hours": item.get("total_shortage_hours"),
                "total_hard_violations": item.get("total_hard_violations"),
                "adoption_rate": item.get("mean_adoption_rate_vs_static_hours"),
                "plan_provider_hours_total": item.get("plan_provider_hours_total"),
                "baseline_provider_hours_total": item.get("baseline_provider_hours_total"),
                "phase_ok": item.get("phase_ok"),
                "safety_gate_pass_rate": item.get("safety_gate_pass_rate"),
                "overall_safety_gate_ok": item.get("overall_safety_gate_ok"),
                "mean_utilization": item.get("mean_utilization"),
                "total_infeasible_shortage_hours": item.get("total_infeasible_shortage_hours"),
                "total_avoidable_shortage_hours": item.get("total_avoidable_shortage_hours"),
                "float_pool_hours_total": item.get("float_pool_hours_total"),
                "on_call_hours_total": item.get("on_call_hours_total"),
                "overtime_extension_hours_total": item.get("overtime_extension_hours_total"),
                "overflow_provider_hours_total": item.get("overflow_provider_hours_total"),
                "monthly_provider_hour_budget": item.get("monthly_provider_hour_budget"),
                "normal_provider_hours_used_total": item.get("normal_provider_hours_used_total"),
                "emergency_hour_bank_total": item.get("emergency_hour_bank_total"),
                "emergency_bank_used_total": item.get("emergency_bank_used_total"),
                "emergency_bank_used_share": item.get("emergency_bank_used_share"),
                "budget_exceeded": item.get("budget_exceeded"),
                "overall_budget_gate_ok": item.get("overall_budget_gate_ok"),
            }
            for item in phase_results
        ]
    )
    phase_df_path = out_root / "phase_summary.csv"
    phase_rows_path = out_root / "rollout_phase_rows.jsonl"

    phase_rows_df = phase_summary_rows
    phase_rows_df.to_csv(phase_df_path, index=False)
    with open(phase_rows_path, "w", encoding="utf-8") as f:
        for row in phase_results:
            f.write(json.dumps(row))
            f.write("\n")

    print("ROLLOUT SUMMARY")
    print(json.dumps(rollout_summary, indent=2))
    print_top_for_paths([str(rollout_summary_path), str(phase_df_path), str(phase_rows_path)] + [str(Path(p) / "daily_audit_summary.json") for p in phase_paths], top_n=args.print_top_n)


if __name__ == "__main__":
    main()
