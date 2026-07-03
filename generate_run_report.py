from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

from scripts.show_top_tables import print_top_for_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ER scheduling run report")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--print-top-n", type=int, default=5)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _to_markdown_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    small = df.head(max_rows).copy()
    if small.empty:
        return "No rows."
    cols = [str(c) for c in small.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in small.iterrows():
        vals = []
        for c in small.columns:
            value = row[c]
            if isinstance(value, float):
                vals.append(f"{value:.4g}")
            else:
                text = str(value).replace("|", "/")
                vals.append(text[:120])
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    out_dir = Path(args.out) if args.out else root
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_comparison = _read_csv(root / "scenario_comparison.csv")
    budget = _read_csv(root / "budget_reallocation_summary.csv")
    forecast = _read_csv(root / "scenarios" / "forecast_scenario_summary.csv")

    lines = [
        "# Budget-Constrained ER Scheduling Run Report",
        "",
        "## Executive summary",
    ]
    if not scenario_comparison.empty:
        passed = scenario_comparison[(scenario_comparison["budget_exceeded"] == False) & (scenario_comparison["overall_safety_gate_ok"] == True)]
        lines.append(f"- Scenarios evaluated: {len(scenario_comparison)}")
        lines.append(f"- Scenarios passing safety and budget gates: {len(passed)}")
        best = scenario_comparison.sort_values(["budget_exceeded", "total_shortage_hours", "mean_utilization"], ascending=[True, True, False]).head(1)
        if not best.empty:
            row = best.iloc[0]
            lines.append(f"- Best operational candidate: `{row['scenario']}` with utilization {row['mean_utilization']:.1%}, shortage hours {row['total_shortage_hours']:.1f}, emergency bank use {row['emergency_bank_used_share']:.1%}.")
    else:
        lines.append("- Scenario comparison was not available.")

    for title, df in [("Forecast scenarios", forecast), ("Optimization scenario comparison", scenario_comparison), ("Budget reallocation", budget)]:
        lines.extend(["", f"## {title}", ""])
        if df.empty:
            lines.append("No data available.")
        else:
            lines.append(_to_markdown_table(df, max_rows=10))

    md = "\n".join(lines) + "\n"
    md_path = out_dir / "run_report.md"
    html_path = out_dir / "run_report.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text("<html><body><pre>" + html.escape(md) + "</pre></body></html>", encoding="utf-8")
    print(f"Wrote report: {md_path}")
    print(f"Wrote report: {html_path}")
    print_top_for_paths([str(root / "scenario_comparison.csv"), str(root / "budget_reallocation_summary.csv")], top_n=args.print_top_n)


if __name__ == "__main__":
    main()
