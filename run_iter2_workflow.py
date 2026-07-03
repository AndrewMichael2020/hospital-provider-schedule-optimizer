from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tabfm_healthcare_eval.utils import safe_json_dump
from scripts.show_top_tables import print_top_for_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realistic ER scheduling workflow (iteration 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--facilities", type=int, default=4)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--backend", choices=["auto", "jax", "pytorch", "tabfm"], default="auto")
    parser.add_argument("--include-xgboost", action="store_true", default=False)
    parser.add_argument("--run-baseline", action="store_true", default=True)
    parser.add_argument("--no-tabfm", action="store_true", default=False)
    parser.add_argument("--shift-hours", type=int, default=4)
    parser.add_argument("--min-shift-hours", type=int, default=4, help="Minimum staffing block (hours)")
    parser.add_argument("--out-prefix", default="data/iter2/er_iter2")
    parser.add_argument("--results-root", default="outputs/iter2")
    parser.add_argument("--optimizer", choices=["auto", "milp", "greedy"], default="auto")
    parser.add_argument("--print-top-n", type=int, default=5)
    parser.add_argument("--plan-horizon-hours", type=int, default=24)
    parser.add_argument("--audit-days", type=int, default=30)
    parser.add_argument("--daily-audit", action="store_true", default=True)
    parser.add_argument("--no-daily-audit", dest="daily_audit", action="store_false")
    parser.add_argument("--full-model-run", action="store_true", default=False)
    parser.add_argument("--rollout", action="store_true", default=False)
    parser.add_argument("--rollout-canary-share", type=float, default=0.30)
    parser.add_argument("--rollout-canary-min", type=int, default=1)
    parser.add_argument("--rollout-stop-on-fail", action="store_true", default=False)
    parser.add_argument("--rollout-shortage-tolerance", type=float, default=0.10)
    parser.add_argument("--rollout-hard-violation-tolerance", type=float, default=0.10)
    parser.add_argument("--rollout-min-adoption", type=float, default=0.60)
    parser.add_argument("--min-coverage-share", type=float, default=0.85)
    parser.add_argument("--target-utilization-floor", type=float, default=0.55)
    parser.add_argument("--max-shortage-increase-share", type=float, default=0.0)
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
    parser.add_argument("--run-scenarios", action="store_true", default=True)
    parser.add_argument(
        "--target-transform",
        choices=["none", "log1p", "sqrt", "auto"],
        default="none",
        help="Demand target transform for model training.",
    )
    parser.add_argument("--target-transform-offset", type=float, default=1e-6)
    parser.add_argument("--tabfm-n-estimators", type=int, default=64)
    parser.add_argument("--tabfm-num-folds-for-cv", type=int, default=3)
    parser.add_argument("--max-categorical-cardinality", type=int, default=512)
    return parser.parse_args()


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, text=True)


def _load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_best_metric(reg_dirs: list[Path], preferred: str | None = None) -> tuple[Path, float]:
    if preferred and preferred != "auto":
        preferred = preferred.replace("pytorch", "tabfm")
        preferred = preferred.replace("jax", "tabfm")
        preferred_names = [preferred]
        if preferred == "tabfm":
            preferred_names = ["tabfm_guarded", "tabfm"]
        for d in reg_dirs:
            best_pref = None
            best_pref_rmse = float("inf")
            for preferred_name in preferred_names:
                pref_metrics = d / f"metrics_{preferred_name}.json"
                if not pref_metrics.exists():
                    continue
                payload = _load_metrics(pref_metrics)
                rmse = payload.get("rmse_test")
                if rmse is None or rmse != rmse:
                    continue
                if float(rmse) < best_pref_rmse:
                    best_pref = pref_metrics
                    best_pref_rmse = float(rmse)
            if best_pref is None:
                continue
            model_name = best_pref.stem.removeprefix("metrics_")
            candidates = list(best_pref.parent.glob(f"predictions_{model_name}.csv"))
            if not candidates:
                raise RuntimeError(f"No prediction file matched for {model_name} in {best_pref.parent}")
            return candidates[0], best_pref_rmse
    best = None
    best_val = float("inf")
    for d in reg_dirs:
        for metrics_path in sorted(d.glob("metrics_*.json")):
            if not metrics_path.name.startswith("metrics_"):
                continue
            payload = _load_metrics(metrics_path)
            rmse = payload.get("rmse_test")
            if rmse is None or rmse != rmse:
                continue
            if rmse < best_val:
                best_val = rmse
                best = metrics_path
    if best is None:
        raise RuntimeError("No valid regression metrics found")
    # Find matching predictions in the same directory.
    model_name = best.stem.removeprefix("metrics_")
    candidates = list(best.parent.glob(f"predictions_{model_name}.csv"))
    if not candidates:
        raise RuntimeError(f"No prediction file matched for {model_name} in {best.parent}")
    return candidates[0], float(best_val)


def _get_regression_label(prediction_csv: Path) -> str:
    return prediction_csv.stem.removeprefix("predictions_")


def _select_best_cls_metric(cls_dirs: list[Path], preferred: str | None = None) -> Path | None:
    if preferred and preferred != "auto":
        preferred = preferred.replace("pytorch", "tabfm")
        preferred = preferred.replace("jax", "tabfm")
        for d in cls_dirs:
            metric_name = f"metrics_{preferred}.json"
            pref_metrics = d / metric_name
            if not pref_metrics.exists():
                continue
            payload = _load_metrics(pref_metrics)
            score = payload.get("roc_auc_test")
            if score is None or score != score:
                continue
            return pref_metrics

    best_cls = None
    best_cls_score = -1.0
    for d in cls_dirs:
        for mp in sorted(d.glob("metrics_*.json")):
            payload = _load_metrics(mp)
            score = payload.get("roc_auc_test")
            if score is None or score != score:
                continue
            if score > best_cls_score:
                best_cls_score = score
                best_cls = mp
    return best_cls


def main() -> None:
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent

    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = repo_dir / out_prefix
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = repo_dir / results_root
    results_root.mkdir(parents=True, exist_ok=True)

    visits_path = out_prefix.with_suffix(".visits.parquet")
    capacity_path = out_prefix.with_suffix(".capacity.parquet")
    shifts_path = out_prefix.with_suffix(".shifts.parquet")
    present_path = out_prefix.with_suffix(".present_shift_hours.parquet")
    model_path = out_prefix.with_suffix(".features.parquet")
    data_meta_path = out_prefix.with_suffix(".features.meta.json")

    # 1) Generate raw iteration-2 tables
    _run(
        [
            sys.executable,
            str(repo_dir / "generate_er_scheduling_iteration2.py"),
            "--seed",
            str(args.seed),
            "--facilities",
            str(args.facilities),
            "--days",
            str(args.days),
            "--min-shift-hours",
            str(args.min_shift_hours),
            "--out-prefix",
            str(out_prefix),
        ]
    )
    print_top_for_paths(
        [str(visits_path), str(capacity_path), str(shifts_path), str(present_path)],
        top_n=args.print_top_n,
    )

    # 2) Build model feature table from visits/capacity/shifts.
    _run(
        [
            sys.executable,
            str(repo_dir / "build_iter2_features.py"),
            "--visits",
            str(visits_path),
            "--capacity",
            str(capacity_path),
            "--present-shifts",
            str(present_path),
            "--out",
            str(model_path),
            "--seed",
            str(args.seed),
        ]
    )

    model_df = pd.read_parquet(model_path)
    print(f"feature_table_shape={model_df.shape}")
    print_top_for_paths([str(model_path)], top_n=args.print_top_n)

    # 3) Validate simple contract checks.
    _run([sys.executable, str(repo_dir / "run_data_contract_checks.py"), "--in", str(model_path)])

    # 4) Compare regressors and classifiers across candidates.
    reg_candidates = [args.backend]
    if args.include_xgboost:
        reg_candidates.append("xgboost")
    cls_candidates = [args.backend]
    if args.include_xgboost:
        cls_candidates.append("xgboost")

    reg_dirs = []
    for b in reg_candidates:
        reg_out = results_root / f"regression_{b}"
        reg_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(repo_dir / "exp_er_demand_forecast.py"),
            "--in",
            str(model_path),
            "--backend",
            b,
            "--results-dir",
            str(reg_out),
            "--seed",
            str(args.seed),
            "--target-transform",
            args.target_transform,
            "--target-transform-offset",
            str(args.target_transform_offset),
            "--tabfm-n-estimators",
            str(args.tabfm_n_estimators),
            "--tabfm-num-folds-for-cv",
            str(args.tabfm_num_folds_for_cv),
            "--max-categorical-cardinality",
            str(args.max_categorical_cardinality),
        ]
        if args.no_tabfm:
            cmd.append("--no-tabfm")
        if args.run_baseline:
            cmd.append("--run-baseline")
        try:
            _run(cmd)
            reg_dirs.append(reg_out)
        except subprocess.CalledProcessError as exc:
            print(f"Skipping regressor candidate {b}: {exc}")

    if not reg_dirs:
        print("No regression model completed. Falling back to synthetic baseline forecast from generated features.")
        forecast_csv = results_root / "fallback_forecast.csv"
        fallback = model_df.loc[:, ["facility_id", "timestamp", "expected_arrivals_next_hour"]].copy()
        fallback["y_pred"] = fallback["expected_arrivals_next_hour"].astype(float)
        fallback[["facility_id", "timestamp", "y_pred"]].to_csv(forecast_csv, index=False)
        best_rmse = float("nan")
        selected_forecast_metrics = {}
    else:
        forecast_csv, best_rmse = _select_best_metric(reg_dirs, preferred=args.backend if args.backend != "auto" else None)
        if args.full_model_run:
            best_label = _get_regression_label(forecast_csv)
            full_backend = "tabfm" if best_label == "tabfm_guarded" else best_label
            full_reg_out = results_root / "regression_full_horizon"
            full_reg_out.mkdir(parents=True, exist_ok=True)
            full_cmd = [
                sys.executable,
                str(repo_dir / "exp_er_demand_forecast.py"),
                "--in",
                str(model_path),
                "--backend",
                full_backend,
                "--results-dir",
                str(full_reg_out),
                "--seed",
                str(args.seed),
                "--fit-mode",
                "full",
                "--target-transform",
                args.target_transform,
                "--target-transform-offset",
                str(args.target_transform_offset),
                "--tabfm-n-estimators",
                str(args.tabfm_n_estimators),
                "--tabfm-num-folds-for-cv",
                str(args.tabfm_num_folds_for_cv),
                "--max-categorical-cardinality",
                str(args.max_categorical_cardinality),
            ]
            if args.no_tabfm:
                full_cmd.append("--no-tabfm")
            elif full_backend not in {"baseline", "xgboost"}:
                full_cmd.append("--run-baseline")
            _run(full_cmd)
            full_forecast, _ = _select_best_metric([full_reg_out], preferred=best_label)
            forecast_csv = full_forecast
        print(f"Selected best regression candidate={forecast_csv} rmse={best_rmse:.4f}")
        print_top_for_paths([str(forecast_csv)], top_n=args.print_top_n)
        selected_label = _get_regression_label(forecast_csv)
        selected_metrics_path = forecast_csv.parent / f"metrics_{selected_label}.json"
        selected_forecast_metrics = _load_metrics(selected_metrics_path) if selected_metrics_path.exists() else {}

    cls_dirs = []
    for b in cls_candidates:
        cls_out = results_root / f"classification_{b}"
        cls_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(repo_dir / "exp_shortage_risk.py"),
            "--in",
            str(model_path),
            "--backend",
            b,
            "--results-dir",
            str(cls_out),
            "--seed",
            str(args.seed),
        ]
        if args.no_tabfm:
            cmd.append("--no-tabfm")
        if args.run_baseline:
            cmd.append("--run-baseline")
        try:
            _run(cmd)
            cls_dirs.append(cls_out)
        except subprocess.CalledProcessError as exc:
            print(f"Skipping classifier candidate {b}: {exc}")

    # 5) Choose best classifier by ROC-AUC
    best_cls = _select_best_cls_metric(cls_dirs, preferred=args.backend if args.backend != "auto" else None)
    best_cls_score = -1.0
    if best_cls is not None:
        best_cls_payload = _load_metrics(best_cls)
        best_cls_score = best_cls_payload.get("roc_auc_test", best_cls_score)
    if best_cls is not None:
        print(f"Best classifier: {best_cls.parent} roc_auc={best_cls_score:.4f}")

    # 6) Optimization loop from best forecast.
    schedule_out = results_root / "scheduling_summary.json"
    sim_cmd = [
        sys.executable,
        str(repo_dir / "simulate_shift_plan.py"),
        "--forecast-csv",
        str(forecast_csv),
        "--actual-csv",
        str(model_path),
        "--provider-shifts",
        str(shifts_path),
        "--out",
        str(schedule_out),
        "--shift-hours",
        str(args.shift_hours),
        "--plan-horizon-hours",
        str(args.plan_horizon_hours),
        "--optimizer",
        args.optimizer,
        "--min-shift-hours",
        str(args.min_shift_hours),
        "--print-top-n",
        str(args.print_top_n),
        "--min-coverage-share",
        str(args.min_coverage_share),
        "--target-utilization-floor",
        str(args.target_utilization_floor),
        "--max-provider-hours-change-share",
        str(args.max_provider_hours_change_share),
    ]
    if not args.allow_float_pool:
        sim_cmd.append("--no-float-pool")
    if not args.allow_overtime_extensions:
        sim_cmd.append("--no-overtime-extensions")
    if args.daily_audit:
        sim_cmd.extend(
            [
                "--daily-audit",
                "--audit-output",
                str(results_root / "daily_audit_chain"),
                "--audit-days",
                str(args.audit_days),
            ]
        )
        sim_cmd.extend(
            [
                "--doc-capacity-per-hour",
                "4.0",
                "--np-capacity-per-hour",
                "3.0",
            ]
        )
    _run(sim_cmd)
    print_top_for_paths(
        [str(schedule_out), str(schedule_out.with_name("scheduling_assignments.csv"))],
        top_n=args.print_top_n,
    )

    if args.daily_audit:
        audit_out = results_root / "daily_audit"
        _run(
            [
                sys.executable,
                str(repo_dir / "run_daily_audit_simulation.py"),
                "--forecast-csv",
                str(forecast_csv),
                "--actual-csv",
                str(model_path),
                "--provider-shifts",
                str(shifts_path),
                "--out",
                str(audit_out),
                "--optimizer",
                args.optimizer,
                "--audit-days",
                str(args.audit_days),
                "--plan-horizon-hours",
                str(args.plan_horizon_hours),
                "--shift-hours",
                str(args.shift_hours),
                "--min-shift-hours",
                str(args.min_shift_hours),
                "--print-top-n",
                str(args.print_top_n),
                "--min-coverage-share",
                str(args.min_coverage_share),
                "--target-utilization-floor",
                str(args.target_utilization_floor),
                "--max-shortage-increase-share",
                str(args.max_shortage_increase_share),
                "--max-provider-hours-change-share",
                str(args.max_provider_hours_change_share),
                "--monthly-provider-hour-budget-mode",
                args.monthly_provider_hour_budget_mode,
                "--emergency-hour-bank-share",
                str(args.emergency_hour_bank_share),
                "--overflow-role-policy",
                args.overflow_role_policy,
                "--target-utilization-lift",
                str(args.target_utilization_lift),
                "--max-utilization-target",
                str(args.max_utilization_target),
                "--open-loop",
            ]
            + ([] if args.allow_float_pool else ["--no-float-pool"])
            + ([] if args.allow_overtime_extensions else ["--no-overtime-extensions"])
        )
        print_top_for_paths(
            [str(audit_out / "feasibility_detail.csv"), str(audit_out / "feasibility_summary.json"), str(audit_out / "daily_audit_summary.json"), str(audit_out / "daily_audit_detail.csv"), str(audit_out / "daily_audit_assignments.csv")],
            top_n=args.print_top_n,
        )

    if args.rollout:
        rollout_out = results_root / "rollout"
        _run(
            [
                sys.executable,
                str(repo_dir / "run_rollout_evaluation.py"),
                "--forecast-csv",
                str(forecast_csv),
                "--actual-csv",
                str(model_path),
                "--provider-shifts",
                str(shifts_path),
                "--out",
                str(rollout_out),
                "--optimizer",
                args.optimizer,
                "--audit-days",
                str(args.audit_days),
                "--plan-horizon-hours",
                str(args.plan_horizon_hours),
                "--shift-hours",
                str(args.shift_hours),
                "--min-shift-hours",
                str(args.min_shift_hours),
                "--canary-share",
                str(args.rollout_canary_share),
                "--canary-min-providers",
                str(args.rollout_canary_min),
                "--shortage-increase-tolerance",
                str(args.rollout_shortage_tolerance),
                "--hard-violation-increase-tolerance",
                str(args.rollout_hard_violation_tolerance),
                "--min-adoption-rate",
                str(args.rollout_min_adoption),
                "--min-coverage-share",
                str(args.min_coverage_share),
                "--target-utilization-floor",
                str(args.target_utilization_floor),
                "--max-shortage-increase-share",
                str(args.max_shortage_increase_share),
                "--max-provider-hours-change-share",
                str(args.max_provider_hours_change_share),
                "--monthly-provider-hour-budget-mode",
                args.monthly_provider_hour_budget_mode,
                "--emergency-hour-bank-share",
                str(args.emergency_hour_bank_share),
                "--overflow-role-policy",
                args.overflow_role_policy,
                "--target-utilization-lift",
                str(args.target_utilization_lift),
                "--max-utilization-target",
                str(args.max_utilization_target),
                "--print-top-n",
                str(args.print_top_n),
            ]
            + (["--stop-on-fail"] if args.rollout_stop_on_fail else [])
            + ([] if args.allow_float_pool else ["--no-float-pool"])
            + ([] if args.allow_overtime_extensions else ["--no-overtime-extensions"])
        )
        print_top_for_paths(
            [
                str(rollout_out / "rollout_summary.json"),
                str(rollout_out / "phase_summary.csv"),
            ],
            top_n=args.print_top_n,
        )


    if args.run_scenarios:
        scenarios_out = results_root / "scenarios"
        _run(
            [
                sys.executable,
                str(repo_dir / "run_forecast_scenarios.py"),
                "--in",
                str(model_path),
                "--tabfm-forecast",
                str(forecast_csv),
                "--out",
                str(scenarios_out),
                "--seed",
                str(args.seed),
                "--print-top-n",
                str(args.print_top_n),
            ]
        )
        scenario_rows = []
        budget_rows = []
        for scenario_name in ["tabfm_guarded", "statistical_ensemble", "conservative_peak"]:
            scenario_csv = scenarios_out / f"scenario_{scenario_name}.csv"
            scenario_out = results_root / f"scenario_eval_{scenario_name}"
            audit_out = scenario_out / "daily_audit"
            _run(
                [
                    sys.executable,
                    str(repo_dir / "run_daily_audit_simulation.py"),
                    "--forecast-csv",
                    str(scenario_csv),
                    "--actual-csv",
                    str(model_path),
                    "--provider-shifts",
                    str(shifts_path),
                    "--out",
                    str(audit_out),
                    "--optimizer",
                    args.optimizer,
                    "--audit-days",
                    str(args.audit_days),
                    "--plan-horizon-hours",
                    str(args.plan_horizon_hours),
                    "--shift-hours",
                    str(args.shift_hours),
                    "--min-shift-hours",
                    str(args.min_shift_hours),
                    "--print-top-n",
                    str(args.print_top_n),
                    "--min-coverage-share",
                    str(args.min_coverage_share),
                    "--target-utilization-floor",
                    str(args.target_utilization_floor),
                    "--max-shortage-increase-share",
                    str(args.max_shortage_increase_share),
                    "--max-provider-hours-change-share",
                    str(args.max_provider_hours_change_share),
                    "--monthly-provider-hour-budget-mode",
                    args.monthly_provider_hour_budget_mode,
                    "--emergency-hour-bank-share",
                    str(args.emergency_hour_bank_share),
                    "--overflow-role-policy",
                    args.overflow_role_policy,
                    "--target-utilization-lift",
                    str(args.target_utilization_lift),
                    "--max-utilization-target",
                    str(args.max_utilization_target),
                    "--open-loop",
                ]
                + ([] if args.allow_float_pool else ["--no-float-pool"])
                + ([] if args.allow_overtime_extensions else ["--no-overtime-extensions"])
            )
            with open(audit_out / "daily_audit_summary.json", "r", encoding="utf-8") as f:
                audit_summary = json.load(f)
            row = {"scenario": scenario_name, **audit_summary, "audit_summary_path": str(audit_out / "daily_audit_summary.json")}
            scenario_rows.append(row)
            budget_rows.append(
                {
                    "scenario": scenario_name,
                    "monthly_provider_hour_budget": audit_summary.get("monthly_provider_hour_budget", 0.0),
                    "normal_provider_hours_used_total": audit_summary.get("normal_provider_hours_used_total", 0.0),
                    "emergency_hour_bank_total": audit_summary.get("emergency_hour_bank_total", 0.0),
                    "emergency_bank_used_total": audit_summary.get("emergency_bank_used_total", 0.0),
                    "emergency_bank_used_share": audit_summary.get("emergency_bank_used_share", 0.0),
                    "overflow_provider_hours_total": audit_summary.get("overflow_provider_hours_total", 0.0),
                    "provider_hours_added_outside_budget_total": audit_summary.get("provider_hours_added_outside_budget_total", 0.0),
                    "budget_exceeded": audit_summary.get("budget_exceeded", False),
                }
            )
            if args.rollout:
                rollout_out = scenario_out / "rollout"
                _run(
                    [
                        sys.executable,
                        str(repo_dir / "run_rollout_evaluation.py"),
                        "--forecast-csv",
                        str(scenario_csv),
                        "--actual-csv",
                        str(model_path),
                        "--provider-shifts",
                        str(shifts_path),
                        "--out",
                        str(rollout_out),
                        "--optimizer",
                        args.optimizer,
                        "--audit-days",
                        str(args.audit_days),
                        "--plan-horizon-hours",
                        str(args.plan_horizon_hours),
                        "--shift-hours",
                        str(args.shift_hours),
                        "--min-shift-hours",
                        str(args.min_shift_hours),
                        "--canary-share",
                        str(args.rollout_canary_share),
                        "--canary-min-providers",
                        str(args.rollout_canary_min),
                        "--min-coverage-share",
                        str(args.min_coverage_share),
                        "--target-utilization-floor",
                        str(args.target_utilization_floor),
                        "--max-shortage-increase-share",
                        str(args.max_shortage_increase_share),
                        "--max-provider-hours-change-share",
                        str(args.max_provider_hours_change_share),
                        "--monthly-provider-hour-budget-mode",
                        args.monthly_provider_hour_budget_mode,
                        "--emergency-hour-bank-share",
                        str(args.emergency_hour_bank_share),
                        "--overflow-role-policy",
                        args.overflow_role_policy,
                        "--target-utilization-lift",
                        str(args.target_utilization_lift),
                        "--max-utilization-target",
                        str(args.max_utilization_target),
                        "--print-top-n",
                        str(args.print_top_n),
                    ]
                    + ([] if args.allow_float_pool else ["--no-float-pool"])
                    + ([] if args.allow_overtime_extensions else ["--no-overtime-extensions"])
                )
        scenario_df = pd.DataFrame(scenario_rows)
        budget_df = pd.DataFrame(budget_rows)
        scenario_df.to_csv(results_root / "scenario_comparison.csv", index=False)
        budget_df.to_csv(results_root / "budget_reallocation_summary.csv", index=False)
        print_top_for_paths([str(results_root / "scenario_comparison.csv"), str(results_root / "budget_reallocation_summary.csv")], top_n=args.print_top_n)
        _run(
            [
                sys.executable,
                str(repo_dir / "generate_run_report.py"),
                "--results-root",
                str(results_root),
                "--print-top-n",
                str(args.print_top_n),
            ]
        )

    with open(schedule_out, "r", encoding="utf-8") as f:
        schedule = json.load(f)
    print("SCHEDULING SUMMARY")
    print(json.dumps(schedule[0] if isinstance(schedule, list) and schedule else schedule, indent=2))
    safe_json_dump(
        {
            "seed": args.seed,
            "facilities": args.facilities,
            "days": args.days,
            "backend": args.backend,
            "full_model_run": args.full_model_run,
            "target_transform": args.target_transform,
            "selected_forecast_csv": str(forecast_csv),
            "best_regression_rmse": float(best_rmse),
            "forecast_model_selected": selected_forecast_metrics.get("forecast_model_selected", _get_regression_label(forecast_csv)),
            "forecast_guardrail_reason": selected_forecast_metrics.get("forecast_guardrail_reason", ""),
            "raw_tabfm_flat_prediction_warning": bool(selected_forecast_metrics.get("raw_tabfm_flat_prediction_warning", False)),
            "selected_forecast_rmse": selected_forecast_metrics.get("selected_forecast_rmse", selected_forecast_metrics.get("rmse_test")),
            "baseline_forecast_rmse": selected_forecast_metrics.get("baseline_forecast_rmse", selected_forecast_metrics.get("source_baseline_rmse_test")),
            "selected_classifier_metrics": str(best_cls) if best_cls else "",
            "best_classifier_roc": float(best_cls_score) if best_cls else None,
            "schedule_summary_path": str(schedule_out),
        },
        str(results_root / "run_summary.json"),
    )


if __name__ == "__main__":
    main()
