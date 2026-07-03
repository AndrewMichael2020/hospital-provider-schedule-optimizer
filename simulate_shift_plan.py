from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path
import time

import pandas as pd

from tabfm_healthcare_eval.optimizer import (
    build_demand_profile,
    build_shift_candidate_pool,
    diagnose_staffing_feasibility,
    solve_staffing_plan,
    summarize_plan_against_baseline,
)
from scripts.show_top_tables import print_top_for_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate shift staffing planner from TabFM forecasts")
    parser.add_argument("--forecast-csv", required=True)
    parser.add_argument("--actual-csv", required=True)
    parser.add_argument("--provider-shifts", required=True, help="Path to provider candidate shifts/parquet or csv")
    parser.add_argument("--out", default="outputs/scheduling/schedule_summary.json")
    parser.add_argument("--shift-hours", type=int, default=4)
    parser.add_argument("--plan-horizon-hours", type=int, default=24)
    parser.add_argument("--optimizer", choices=["auto", "milp", "greedy"], default="auto")
    parser.add_argument("--doc-capacity-per-hour", type=float, default=4.0)
    parser.add_argument("--np-capacity-per-hour", type=float, default=3.0)
    parser.add_argument("--prediction-col", default="", help="Optional arrival prediction column in forecast file")
    parser.add_argument("--min-shift-hours", type=int, default=4, help="Minimum provider shift block to enforce")
    parser.add_argument("--print-top-n", type=int, default=5)
    parser.add_argument("--daily-audit", action="store_true", default=False)
    parser.add_argument("--audit-output", default="outputs/scheduling_audit")
    parser.add_argument("--audit-days", type=int, default=14, help="Used when --daily-audit is set")
    parser.add_argument("--min-coverage-share", type=float, default=0.85)
    parser.add_argument("--target-utilization-floor", type=float, default=0.55)
    parser.add_argument("--max-provider-hours-change-share", type=float, default=0.10)
    parser.add_argument("--allow-float-pool", action="store_true", default=True)
    parser.add_argument("--no-float-pool", dest="allow_float_pool", action="store_false")
    parser.add_argument("--allow-overtime-extensions", action="store_true", default=True)
    parser.add_argument("--no-overtime-extensions", dest="allow_overtime_extensions", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    forecast_path = Path(args.forecast_csv)
    if forecast_path.suffix.lower() == ".parquet":
        forecast = pd.read_parquet(forecast_path)
    else:
        forecast = pd.read_csv(forecast_path)
    if "timestamp" in forecast.columns:
        forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])

    actual_path = Path(args.actual_csv)
    if actual_path.suffix.lower() in {".parquet", ".pq"}:
        actual = pd.read_parquet(actual_path)
    else:
        actual = pd.read_csv(actual_path)
    if "timestamp" in actual.columns:
        actual["timestamp"] = pd.to_datetime(actual["timestamp"])

    if args.prediction_col:
        prediction_candidates = [args.prediction_col]
    else:
        prediction_candidates = [
            "predicted_arrivals",
            "y_pred",
            "prediction",
            "forecast",
        ]
    arrival_col = next((c for c in prediction_candidates if c in forecast.columns), "")
    if not arrival_col:
        raise ValueError(
            "Could not find a prediction column. Provide one with --prediction-col. "
            f"Expected one of: {prediction_candidates}"
        )
    required_cols = {"facility_id", "timestamp", arrival_col}
    if not required_cols.issubset(forecast.columns):
        missing = sorted(required_cols - set(forecast.columns))
        raise ValueError(f"forecast CSV missing columns {missing}")
    if arrival_col != "predicted_arrivals":
        forecast["predicted_arrivals"] = pd.to_numeric(forecast[arrival_col], errors="coerce")
    else:
        forecast["predicted_arrivals"] = pd.to_numeric(forecast[arrival_col], errors="coerce")

    print("Loaded forecast table:")
    print_top_for_paths([str(forecast_path)], top_n=args.print_top_n)
    print("Loaded actual context table:")
    print_top_for_paths([str(actual_path)], top_n=args.print_top_n)

    if "shortage_risk_next_shift" in actual.columns:
        actual_shortage = float(actual["shortage_risk_next_shift"].mean())
    else:
        actual_shortage = float("nan")

    provider_path = Path(args.provider_shifts)
    if provider_path.suffix.lower() == ".parquet":
        provider_df = pd.read_parquet(provider_path)
    else:
        provider_df = pd.read_csv(provider_path)
    if args.provider_shifts and provider_df.empty:
        raise ValueError(f"provider-shifts file is empty: {args.provider_shifts}")
    print("Loaded provider-shifts table:")
    print_top_for_paths([str(provider_path)], top_n=args.print_top_n)

    horizon_end = forecast["timestamp"].min() + pd.Timedelta(hours=max(1, int(args.plan_horizon_hours)))
    demand_profile = build_demand_profile(
        forecast,
        shift_hours=args.shift_hours,
        demand_mode="next_nh",
        required_horizon_start=forecast["timestamp"].min(),
        required_horizon_end=horizon_end,
        doc_capacity_per_hour=args.doc_capacity_per_hour,
        np_capacity_per_hour=args.np_capacity_per_hour,
    )
    candidate_pool = build_shift_candidate_pool(
        provider_df,
        min_shift_hours=args.min_shift_hours,
        required_horizon_start=demand_profile["timestamp"].min() if not demand_profile.empty else None,
        required_horizon_end=demand_profile["timestamp"].max() + pd.Timedelta(hours=1) if not demand_profile.empty else None,
        allow_float_pool=args.allow_float_pool,
        allow_overtime_extensions=args.allow_overtime_extensions,
    )

    t0 = time.perf_counter()
    assignments, metrics, used_engine = solve_staffing_plan(
        demand_profile=demand_profile,
        candidate_pool=candidate_pool,
        min_shift_hours=args.min_shift_hours,
        doc_capacity_per_hour=args.doc_capacity_per_hour,
        np_capacity_per_hour=args.np_capacity_per_hour,
        engine=args.optimizer,
        target_utilization_floor=args.target_utilization_floor,
        max_provider_hours_change_share=args.max_provider_hours_change_share,
    )
    metrics["solve_seconds"] = float(time.perf_counter() - t0)
    metrics["solver_engine"] = used_engine
    metrics["request_engine"] = args.optimizer

    metrics["plan_rows"] = int(len(assignments))
    metrics["candidate_rows"] = int(len(candidate_pool))
    metrics["demand_rows"] = int(len(demand_profile))
    metrics["actual_shortage_rate"] = float(actual_shortage)
    metrics = summarize_plan_against_baseline(
        demand_profile,
        metrics,
        provider_staff=actual,
        plan_assignments=assignments,
    )
    feasibility_detail, feasibility_summary = diagnose_staffing_feasibility(
        demand_profile,
        candidate_pool,
        assignments=assignments,
    )
    metrics.update(
        {
            "feasible_coverage_share": float(feasibility_summary.get("feasible_coverage_share", 0.0)),
            "infeasible_shortage_hours": float(feasibility_summary.get("infeasible_shortage_hours", 0.0)),
            "avoidable_shortage_hours": float(feasibility_summary.get("avoidable_shortage_hours", 0.0)),
            "target_utilization_floor": float(args.target_utilization_floor),
            "min_coverage_share": float(args.min_coverage_share),
            "max_provider_hours_change_share": float(args.max_provider_hours_change_share),
            "allow_float_pool": bool(args.allow_float_pool),
            "allow_overtime_extensions": bool(args.allow_overtime_extensions),
        }
    )

    print("Shift plan summary:")
    print(f"  status={metrics.get('status')}")
    print(f"  solver={metrics.get('used_engine')}")
    print(f"  expected_shortfall_hours={metrics['expected_shortfall_hours']:.3f}")
    print(f"  sla_risk_reduction_proxy={metrics['sla_risk_reduction_proxy']:.3f}")
    print(f"  utilization={metrics['utilization']:.3f}")
    print(f"  overtime_proxy={metrics['overtime_proxy']:.3f}")
    print(f"  hard_violation_count={metrics.get('hard_violation_count')}")
    print(f"  recommended_provider_hours={metrics.get('recommended_provider_hours', 0):.3f}")
    print(f"  recommended_doc_hours={metrics.get('recommended_doc_hours', 0):.3f}")
    print(f"  recommended_np_hours={metrics.get('recommended_np_hours', 0):.3f}")
    print(f"  baseline_provider_hours={metrics.get('baseline_provider_hours', 0):.3f}")
    print(f"  provider_hours_saved={metrics.get('provider_hours_saved', 0):.3f}")
    print(f"  provider_hours_added={metrics.get('provider_hours_added', 0):.3f}")
    print(f"  adoption_rate_vs_static_hours={metrics.get('adoption_rate_vs_static_hours', 0):.3f}")
    print(f"  feasible_coverage_share={metrics.get('feasible_coverage_share', 0):.3f}")
    print(f"  infeasible_shortage_hours={metrics.get('infeasible_shortage_hours', 0):.3f}")
    print(f"  avoidable_shortage_hours={metrics.get('avoidable_shortage_hours', 0):.3f}")
    print(f"  float_pool_hours={metrics.get('float_pool_hours', 0):.3f}")
    print(f"  on_call_hours={metrics.get('on_call_hours', 0):.3f}")
    print(f"  overtime_extension_hours={metrics.get('overtime_extension_hours', 0):.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = out_path.with_suffix(".json")
    assignments_path = out_path.with_name("scheduling_assignments.csv")
    feasibility_detail_path = out_path.with_name("feasibility_detail.csv")
    feasibility_summary_path = out_path.with_name("feasibility_summary.json")
    pd.DataFrame(assignments).to_csv(assignments_path, index=False)
    feasibility_detail.to_csv(feasibility_detail_path, index=False)
    pd.DataFrame([feasibility_summary]).to_json(feasibility_summary_path, orient="records")
    pd.DataFrame([metrics]).to_json(out_path, orient="records")
    print(f"Wrote schedule artifacts: {assignments_path} , {out_path}")
    print_top_for_paths([str(feasibility_detail_path), str(feasibility_summary_path), str(assignments_path), str(out_path)], top_n=args.print_top_n)

    if args.daily_audit:
        print(f"Starting chained daily-audit run for {args.audit_days} days...")
        audit_output = Path(args.audit_output)
        audit_output.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "run_daily_audit_simulation.py"),
                "--forecast-csv",
                args.forecast_csv,
                "--actual-csv",
                args.actual_csv,
                "--provider-shifts",
                args.provider_shifts,
                "--out",
                str(audit_output),
                "--audit-days",
                str(args.audit_days),
                "--optimizer",
                args.optimizer,
                "--plan-horizon-hours",
                str(args.plan_horizon_hours),
                "--shift-hours",
                str(args.shift_hours),
                "--print-top-n",
                str(args.print_top_n),
                "--open-loop",
                "--doc-capacity-per-hour",
                str(args.doc_capacity_per_hour),
                "--np-capacity-per-hour",
                str(args.np_capacity_per_hour),
                "--min-shift-hours",
                str(args.min_shift_hours),
                "--min-coverage-share",
                str(args.min_coverage_share),
                "--target-utilization-floor",
                str(args.target_utilization_floor),
                "--max-provider-hours-change-share",
                str(args.max_provider_hours_change_share),
                "--max-shortage-increase-share",
                "0.0",
            ],
            check=True,
        )
        audit_jsonl = Path(args.audit_output) / "scheduling_daily_audit.jsonl"
        detail_jsonl = Path(args.audit_output) / "daily_audit_rows.jsonl"
        if detail_jsonl.exists():
            # Backward compatible alias for the planner-specific artifact name.
            audit_jsonl.write_text(detail_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
            print_top_for_paths([str(audit_jsonl)], top_n=args.print_top_n)
        else:
            print(f"[warn] expected chained-audit jsonl not found at {detail_jsonl}")

    print(f"Summary metrics file: {out_path}")


if __name__ == "__main__":
    main()
