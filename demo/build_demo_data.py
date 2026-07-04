from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROLES = ["md", "np", "rn", "hcw"]
ROLE_LABELS = {"md": "MD", "np": "NP", "rn": "RN", "hcw": "HCW"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the GitHub Pages demo data bundle.")
    parser.add_argument("--results-root", required=True, help="Fixed-rotation workflow output directory.")
    parser.add_argument("--out", default="demo/data/demo_case.json")
    parser.add_argument("--scenario", default="tabfm_guarded")
    parser.add_argument("--top-actions", type=int, default=8)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Parquet: {path}")
    return pd.read_parquet(path)


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    return int(round(_safe_float(value)))


def _pick_scenario(summary: pd.DataFrame, preferred: str) -> str:
    if preferred in set(summary["scenario_name"].astype(str)):
        row = summary[summary["scenario_name"] == preferred].iloc[0]
        if bool(row.get("rollout_pass", False)) and not bool(row.get("budget_exceeded", True)):
            return preferred
    viable = summary[(summary["rollout_pass"] == True) & (summary["budget_exceeded"] == False)].copy()
    if viable.empty:
        viable = summary.copy()
    viable = viable.sort_values(
        ["shortage_reduction_vs_manual", "coverage_risk_reduction_vs_manual", "optimized_overflow_hit_rate"],
        ascending=[False, False, False],
    )
    return str(viable.iloc[0]["scenario_name"])


def _format_time(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)
    return ts.strftime("%b %-d, %-I %p")


def _role_budget(row: pd.Series) -> list[dict[str, Any]]:
    items = []
    for role in ROLES:
        budget = _safe_float(row[f"overflow_budget_hours_{role}"])
        used = _safe_float(row[f"optimized_{role}_overflow_hours_used"])
        remaining = _safe_float(row[f"overflow_hours_remaining_{role}"])
        items.append(
            {
                "role": ROLE_LABELS[role],
                "budgetHours": round(budget, 1),
                "usedHours": round(used, 1),
                "remainingHours": round(remaining, 1),
                "usedPct": round((used / budget) * 100, 1) if budget else 0.0,
                "overBudget": used > budget + 1e-9,
            }
        )
    return items


def _remaining_gaps(need_detail: pd.DataFrame, scenario: str) -> list[dict[str, Any]]:
    data = need_detail[need_detail["scenario_name"] == scenario].copy()
    out = []
    for role in ROLES:
        gap_col = f"remaining_{role}_gap"
        if gap_col not in data.columns:
            continue
        gap_hours = _safe_float(data[gap_col].sum())
        pressure_rows = int((pd.to_numeric(data[gap_col], errors="coerce").fillna(0.0) > 0).sum())
        out.append({"role": ROLE_LABELS[role], "remainingGapHours": round(gap_hours, 1), "pressureHours": pressure_rows})
    return out


def _comparison(comparison: pd.DataFrame, scenario: str) -> list[dict[str, Any]]:
    labels = {
        "static_fixed_rotation": "Fixed rotation only",
        "manual_threshold_overflow": "Manual overflow",
        "optimized_overflow_reallocation": "Optimized overflow",
    }
    data = comparison[comparison["scenario_name"] == scenario].copy()
    data["order"] = data["mode"].map({name: i for i, name in enumerate(labels)})
    data = data.sort_values("order")
    return [
        {
            "mode": labels.get(str(row["mode"]), str(row["mode"])),
            "shortageHours": round(_safe_float(row["shortage_hours"]), 1),
            "coverageRiskHours": _safe_int(row["coverage_risk_hours"]),
            "waitRiskProxy": round(_safe_float(row["wait_risk_proxy"]), 1),
        }
        for _, row in data.iterrows()
    ]


def _top_actions(psp: pd.DataFrame, scenario: str, limit: int) -> list[dict[str, Any]]:
    data = psp[psp["scenario_name"] == scenario].copy()
    if data.empty:
        return []
    data = data.sort_values(["priority_tier", "priority_score"], ascending=[True, False]).head(limit)
    actions = []
    for _, row in data.iterrows():
        role = str(row["role"]).replace("HOSPITALIST_OVERFLOW", "Hospitalist overflow")
        score = _safe_float(row["priority_score"])
        actions.append(
            {
                "facility": str(row["facility_id"]),
                "role": role,
                "window": f"{_format_time(row['planned_activation_start'])} to {_format_time(row['planned_activation_end'])}",
                "priority": str(row["priority_tier"]),
                "expectedHit": bool(row["optimizer_expected_hit"]),
                "score": round(score, 2),
            }
        )
    return actions


def _schedule_rows(df: pd.DataFrame, scenario: str | None, limit: int = 16) -> list[dict[str, Any]]:
    data = df.copy()
    if scenario is not None and "scenario_name" in data.columns:
        data = data[data["scenario_name"] == scenario].copy()
    start_col = "planned_activation_start" if "planned_activation_start" in data.columns else "shift_start"
    end_col = "planned_activation_end" if "planned_activation_end" in data.columns else "shift_end"
    priority_col = "priority_tier" if "priority_tier" in data.columns else None
    score_col = "priority_score" if "priority_score" in data.columns else None
    data[start_col] = pd.to_datetime(data[start_col], errors="coerce")
    data[end_col] = pd.to_datetime(data[end_col], errors="coerce")
    if "is_core_rotation" in data.columns:
        data = data[data["is_core_rotation"].astype(bool) == (scenario is None)].copy()
    data = data.sort_values([start_col, "facility_id", "role"]).head(limit)
    rows = []
    for _, row in data.iterrows():
        role = str(row["role"]).replace("HOSPITALIST_OVERFLOW", "Hospitalist")
        facility = str(row["facility_id"])
        department = str(row.get("department_id", "ER_MAIN"))
        start = pd.to_datetime(row[start_col], errors="coerce")
        end = pd.to_datetime(row[end_col], errors="coerce")
        hours = max(0.0, (end - start).total_seconds() / 3600.0) if not pd.isna(start) and not pd.isna(end) else 0.0
        rows.append(
            {
                "facility": facility,
                "department": department,
                "role": role,
                "line": str(row.get("rotation_line_id", row.get("plan_id", ""))).replace("_", " "),
                "start": _format_time(start),
                "end": _format_time(end),
                "hours": round(hours, 1),
                "priority": str(row[priority_col]) if priority_col else "Core",
                "score": round(_safe_float(row[score_col]), 2) if score_col else 0.0,
                "status": "Add overflow" if scenario is not None else "Protected core",
            }
        )
    return rows


def _parameter_defaults(row: pd.Series) -> dict[str, Any]:
    return {
        "overflowBudgetPct": 10,
        "surgeMultiplier": 135,
        "minimumCoveredGapHours": 4,
        "overflowShiftHours": 10,
        "baselineManualShortageHours": round(_safe_float(row["manual_shortage_hours"]), 1),
        "baseOptimizedShortageHours": round(_safe_float(row["optimized_shortage_hours"]), 1),
        "baseBudgetHours": round(_safe_float(row["total_overflow_budget_hours"]), 1),
        "baseUsedHours": round(_safe_float(row["total_optimized_overflow_hours_used"]), 1),
    }


def build_payload(results_root: Path, preferred_scenario: str, top_actions: int) -> dict[str, Any]:
    summary = _read_csv(results_root / "overflow_reallocation_summary.csv")
    comparison = _read_csv(results_root / "manual_vs_optimized_overflow.csv")
    need_detail = _read_csv(results_root / "recommended_overflow_capacity_needed_by_hour.csv")
    psp = _read_parquet(results_root / "planned_schedule_table.parquet")
    psh = _read_parquet(results_root / "provider_shifts_historical.parquet")
    meta_path = results_root / "fixed_rotation_overflow.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    scenario = _pick_scenario(summary, preferred_scenario)
    row = summary[summary["scenario_name"] == scenario].iloc[0]

    manual_shortage = _safe_float(row["manual_shortage_hours"])
    optimized_shortage = _safe_float(row["optimized_shortage_hours"])
    shortage_reduction = _safe_float(row["shortage_reduction_vs_manual"])
    coverage_reduction = _safe_float(row["coverage_risk_reduction_vs_manual"])
    wait_risk_reduction = _safe_float(row["wait_risk_proxy_reduction_vs_manual"])

    return {
        "generatedFrom": str(results_root),
        "scenario": scenario,
        "case": {
            "title": "Funded overflow, placed before the surge",
            "subtitle": "Synthetic 4-facility, 30-day ER network using protected fixed rotations and role-specific overflow banks.",
            "syntheticData": True,
            "noPhi": True,
            "seed": meta.get("seed"),
            "facilities": meta.get("facilities"),
            "days": meta.get("days"),
        },
        "kpis": [
            {
                "label": "Shortage hours avoided",
                "value": _safe_int(shortage_reduction),
                "unit": "hours",
                "note": f"Manual {manual_shortage:.0f} -> optimized {optimized_shortage:.0f}",
            },
            {
                "label": "Coverage-risk hours avoided",
                "value": _safe_int(coverage_reduction),
                "unit": "hours",
                "note": "Hours with any remaining role shortage",
            },
            {
                "label": "Wait-risk proxy reduced",
                "value": round(wait_risk_reduction, 1),
                "unit": "points",
                "note": "Only counted during shortage periods",
            },
            {
                "label": "Budget status",
                "value": "Pass" if not bool(row["budget_exceeded"]) else "Review",
                "unit": "",
                "note": f"{_safe_float(row['total_optimized_overflow_hours_used']):.0f} of {_safe_float(row['total_overflow_budget_hours']):.0f} role-hours used",
            },
        ],
        "comparison": _comparison(comparison, scenario),
        "roleBudgets": _role_budget(row),
        "remainingGaps": _remaining_gaps(need_detail, scenario),
        "topActions": _top_actions(psp, scenario, top_actions),
        "initialSchedule": _schedule_rows(psh, None, limit=16),
        "optimizedSchedule": _schedule_rows(psp, scenario, limit=16),
        "parameters": _parameter_defaults(row),
        "plainLanguage": {
            "problem": "The standing schedule is protected. The business case is whether funded overflow can be placed earlier and more precisely.",
            "manual": "Manual call-outs react after visible pressure appears, which can miss the right role, site, or hour.",
            "optimizer": "Adjust the levers, regenerate the recommendation, and compare the added overflow schedule against the protected core.",
            "remaining": "Remaining gaps stay visible so schedulers see what the funded bank could not solve.",
        },
    }


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    out = Path(args.out)
    payload = build_payload(results_root, args.scenario, args.top_actions)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
