from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pandas.api.types as ptypes

from tabfm_healthcare_eval.data import facility_time_split
from tabfm_healthcare_eval.models import build_model, evaluate_model
from tabfm_healthcare_eval.utils import set_deterministic_seed, safe_json_dump


def parse_args():
    parser = argparse.ArgumentParser(description="Run TabFM classifier risk stress test")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument(
        "--backend",
        choices=["auto", "jax", "pytorch", "tabfm", "xgboost", "xgb", "hgb", "hist", "histgb", "baseline", "skip"],
        default="auto",
    )
    parser.add_argument("--run-baseline", action="store_true", default=False)
    parser.add_argument("--no-tabfm", action="store_true", default=False)
    parser.add_argument("--results-dir", default="outputs/classification")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _is_categorical_like(series: pd.Series) -> bool:
    return ptypes.is_string_dtype(series) or ptypes.is_object_dtype(series) or ptypes.is_categorical_dtype(series)


def _backend_label(backend: str, fitted_model) -> str:
    if backend in {"xgboost", "xgb"}:
        return "xgboost"
    if backend in {"auto", "jax", "pytorch", "tabfm"}:
        return "tabfm" if "TabFM" in fitted_model.__class__.__name__ else "hgb"
    if backend in {"hgb", "hist", "histgb", "baseline"}:
        return "baseline"
    return backend


def _write_output_artifacts(out_dir: Path, label: str, test_df: pd.DataFrame, target_col: str, metrics: dict) -> None:
    pred_df = test_df[["facility_id", "timestamp", target_col]].copy()
    pred_df["y_pred"] = test_df["_pred"]
    pred_df["y_true"] = test_df[target_col].values
    predictions_path = out_dir / f"predictions_{label}.csv"
    metrics_path = out_dir / f"metrics_{label}.json"
    pred_df.to_csv(predictions_path, index=False)
    safe_json_dump(metrics, str(metrics_path))

    if label == "tabfm":
        pred_df.to_csv(out_dir / "predictions_tabfm.csv", index=False)
        safe_json_dump(metrics, str(out_dir / "metrics_tabfm.json"))
    if label == "baseline":
        pred_df.to_csv(out_dir / "predictions_baseline.csv", index=False)
        safe_json_dump(metrics, str(out_dir / "metrics_baseline.json"))


def main():
    args = parse_args()
    set_deterministic_seed(args.seed)

    if args.in_path.endswith(".parquet"):
        df = pd.read_parquet(args.in_path)
    else:
        df = pd.read_csv(args.in_path)
    if args.rows:
        df = df.head(args.rows).copy()

    target = "shortage_risk_next_shift"
    if target not in df.columns:
        raise ValueError(f"missing target {target}")

    feature_cols = [c for c in df.columns if c not in {target, "expected_arrivals_next_hour"}]
    dt_features = ["timestamp"]
    feature_cols = [c for c in feature_cols if c != "timestamp"]
    cat_features = [c for c in feature_cols if _is_categorical_like(df[c])]

    train, val, test = facility_time_split(
        df,
        facility_col="facility_id",
        timestamp_col="timestamp",
        test_size=0.2,
        val_size=0.2,
    )

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    if args.no_tabfm and not args.run_baseline:
        raise ValueError("no models selected; pass --run-baseline or remove --no-tabfm")

    if not args.no_tabfm:
        model = build_model(
            kind="classification",
            backend=args.backend,
            random_state=args.seed,
            cat_features=cat_features,
            dt_features=dt_features,
        )
        metrics = evaluate_model(model, train, val, test, feature_cols=feature_cols, target_col=target, kind="classification")
        test = test.copy()
        test["_pred"] = model.predict(test[feature_cols]).astype(int)
        label = _backend_label(args.backend, model)
        _write_output_artifacts(out_dir, label, test, target, metrics)
        metrics["backend"] = args.backend
        metrics["label"] = label
        metrics["kind"] = model.__class__.__name__
        artifacts[label] = metrics

    if args.run_baseline:
        baseline = build_model(
            kind="classification",
            backend="baseline",
            random_state=args.seed,
            cat_features=cat_features,
            dt_features=dt_features,
        )
        metrics_b = evaluate_model(
            baseline,
            train,
            val,
            test,
            feature_cols=feature_cols,
            target_col=target,
            kind="classification",
        )
        test_b = test.copy()
        test_b["_pred"] = baseline.predict(test_b[feature_cols]).astype(int)
        _write_output_artifacts(out_dir, "baseline", test_b, target, metrics_b)
        metrics_b["backend"] = "baseline"
        metrics_b["label"] = "baseline"
        metrics_b["kind"] = baseline.__class__.__name__
        artifacts["baseline"] = metrics_b

    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
