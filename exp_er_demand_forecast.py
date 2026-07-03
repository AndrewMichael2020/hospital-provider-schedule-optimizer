from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple
import numpy as np

import pandas as pd
import pandas.api.types as ptypes

from tabfm_healthcare_eval.data import facility_time_split
from tabfm_healthcare_eval.models import build_model, evaluate_model
from tabfm_healthcare_eval.utils import set_deterministic_seed, safe_json_dump


def parse_args():
    parser = argparse.ArgumentParser(description="Run TabFM regressor stress test")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument(
        "--backend",
        choices=["auto", "jax", "pytorch", "tabfm", "xgboost", "xgb", "hgb", "hist", "histgb", "baseline", "extratrees", "extra_trees", "et", "randomforest", "random_forest", "rf", "skip"],
        default="auto",
    )
    parser.add_argument("--run-baseline", action="store_true", default=False)
    parser.add_argument("--no-tabfm", action="store_true", default=False)
    parser.add_argument("--results-dir", default="outputs/regression")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fit-mode",
        choices=["split", "full"],
        default="split",
        help="Use split for evaluation or full for planner-only full-horizon inference.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["none", "log1p", "sqrt", "auto"],
        default="auto",
        help="Demand target transform for model fit (improves variance control).",
    )
    parser.add_argument("--target-transform-offset", type=float, default=1e-6)
    parser.add_argument("--tabfm-n-estimators", type=int, default=64)
    parser.add_argument("--tabfm-num-folds-for-cv", type=int, default=3)
    parser.add_argument("--max-categorical-cardinality", type=int, default=512)
    return parser.parse_args()


def _is_categorical_like(series: pd.Series) -> bool:
    return ptypes.is_string_dtype(series) or ptypes.is_object_dtype(series) or ptypes.is_categorical_dtype(series)


def _build_feature_lists(
    df: pd.DataFrame,
    target_col: str,
    max_categorical_cardinality: int,
) -> Tuple[list[str], list[str], list[str]]:
    feature_cols = [c for c in df.columns if c not in {target_col, "shortage_risk_next_shift"}]
    dt_features = []
    for col in feature_cols:
        if col == "timestamp" or ptypes.is_datetime64_any_dtype(df[col]):
            dt_features.append(col)
    model_features = [c for c in feature_cols if c not in dt_features]
    cat_features = []
    for col in model_features:
        if _is_categorical_like(df[col]):
            cat_features.append(col)
            continue
        if (ptypes.is_integer_dtype(df[col]) or ptypes.is_unsigned_integer_dtype(df[col])):
            if df[col].nunique(dropna=True) <= max_categorical_cardinality:
                cat_features.append(col)
    return model_features, dt_features, cat_features


def _inverse_target_transform(values: np.ndarray, mode: str, offset: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if mode == "none":
        return np.maximum(values, 0.0)
    if mode == "log1p":
        inv = np.expm1(values) - offset
        return np.maximum(inv, 0.0)
    if mode == "sqrt":
        inv = np.square(values)
        return np.maximum(inv, 0.0)
    raise ValueError(f"unsupported target transform: {mode}")


def _backend_label(backend: str, fitted_model) -> str:
    if backend in {"xgboost", "xgb"}:
        return "xgboost"
    if backend in {"extratrees", "extra_trees", "et"}:
        return "extratrees"
    if backend in {"randomforest", "random_forest", "rf"}:
        return "randomforest"
    if backend in {"auto", "jax", "pytorch", "tabfm"}:
        return "tabfm" if "TabFM" in fitted_model.__class__.__name__ else "hgb"
    if backend in {"hgb", "hist", "histgb", "baseline"}:
        return "baseline"
    return backend


def _write_output_artifacts(out_dir: Path, label: str, test_df: pd.DataFrame, target_col: str, metrics: dict) -> Path:
    pred_df = test_df[[ "facility_id", "timestamp", target_col]].copy()
    pred_df["y_pred"] = test_df["_pred"]
    pred_df["y_true"] = test_df[target_col].values
    predictions_path = out_dir / f"predictions_{label}.csv"
    metrics_path = out_dir / f"metrics_{label}.json"
    pred_df.to_csv(predictions_path, index=False)
    safe_json_dump(metrics, str(metrics_path))

    # Backward-compatible aliases for downstream scripts.
    if label == "tabfm":
        pred_df.to_csv(out_dir / "predictions_tabfm.csv", index=False)
        safe_json_dump(metrics, str(out_dir / "metrics_tabfm.json"))
    if label == "baseline":
        pred_df.to_csv(out_dir / "predictions_baseline.csv", index=False)
        safe_json_dump(metrics, str(out_dir / "metrics_baseline.json"))
    return predictions_path


def _prediction_quality_flags(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    true_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
    std_ratio = float(pred_std / max(true_std, 1e-9))
    return {
        "prediction_std": pred_std,
        "target_std": true_std,
        "prediction_std_ratio": std_ratio,
        "flat_prediction_warning": bool(std_ratio < 0.20),
    }


def _regression_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "rmse_test": float(np.sqrt(np.mean(err ** 2))),
        "mae_test": float(np.mean(np.abs(err))),
        "r2_test": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def _train_and_score(
    backend: str,
    target_transform: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    args,
    dt_features: list[str],
    cat_features: list[str],
):
    model = build_model(
        kind="regression",
        backend=backend,
        random_state=args.seed,
        cat_features=cat_features,
        dt_features=dt_features,
        tabfm_n_estimators=args.tabfm_n_estimators,
        tabfm_num_folds_for_cv=args.tabfm_num_folds_for_cv,
    )
    metrics = evaluate_model(
        model,
        train,
        val,
        test,
        feature_cols=feature_cols,
        target_col=target,
        kind="regression",
        target_transform=target_transform,
        target_transform_offset=args.target_transform_offset,
        return_predictions=True,
    )
    pred_test = np.asarray(metrics.pop("pred_test_orig"), dtype=float)
    metrics.pop("pred_train_orig", None)
    metrics.pop("pred_val_orig", None)
    metrics.pop("y_val_true", None)
    metrics.pop("y_test_true", None)
    scored_test = test.copy()
    scored_test["_pred"] = pred_test
    metrics["target_transform"] = target_transform
    return model, metrics, scored_test


def main():
    args = parse_args()
    set_deterministic_seed(args.seed)

    if args.in_path.endswith(".parquet"):
        df = pd.read_parquet(args.in_path)
    else:
        df = pd.read_csv(args.in_path)
    if args.rows:
        df = df.head(args.rows).copy()

    target = "expected_arrivals_next_hour"
    if target not in df.columns:
        raise ValueError(f"missing target {target}")

    feature_cols, dt_features, cat_features = _build_feature_lists(
        df, target_col=target, max_categorical_cardinality=args.max_categorical_cardinality
    )

    if args.fit_mode == "full":
        train = df.copy()
        val = df.copy()
        test = df.copy()
    else:
        train, val, test = facility_time_split(df, facility_col="facility_id", timestamp_col="timestamp", test_size=0.2, val_size=0.2)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    tabfm_scored_test = None
    if args.no_tabfm and not args.run_baseline:
        raise ValueError("no models selected; pass --run-baseline or remove --no-tabfm")

    selected_transform = args.target_transform
    if not args.no_tabfm:
        candidate_transforms = [args.target_transform] if args.target_transform != "auto" else ["log1p", "sqrt", "none"]
        best_model = None
        best_metrics = None
        best_test = None
        best_val = float("inf")
        for candidate in candidate_transforms:
            model, candidate_metrics, scored_test = _train_and_score(
                backend=args.backend,
                target_transform=candidate,
                train=train,
                val=val,
                test=test,
                feature_cols=feature_cols,
                target=target,
                args=args,
                dt_features=dt_features,
                cat_features=cat_features,
            )
            candidate_val = float(candidate_metrics.get("rmse_val", float("inf")))
            if np.isfinite(candidate_val) and candidate_val < best_val:
                best_val = candidate_val
                best_model = model
                best_metrics = candidate_metrics
                best_test = scored_test
                selected_transform = candidate
        if best_model is None or best_metrics is None or best_test is None:
            raise RuntimeError("no valid regression candidate produced metrics")
        model = best_model
        metrics = best_metrics
        test = best_test
        metrics["fit_seconds"] = float(metrics.get("fit_seconds", 0.0))
        metrics["pred_seconds"] = float(metrics.get("pred_seconds", 0.0))
        label = _backend_label(args.backend, model)
        metrics["backend"] = args.backend
        metrics["label"] = label
        metrics["kind"] = model.__class__.__name__
        metrics.update(_prediction_quality_flags(test[target].values, test["_pred"].values))
        _write_output_artifacts(out_dir, label, test, target, metrics)
        artifacts[label] = metrics
        if label == "tabfm":
            tabfm_scored_test = test.copy()
    else:
        label = "baseline"

    if args.run_baseline:
        # run a robust sklearn fallback for comparison
        baseline_transform = selected_transform if selected_transform != "auto" else "none"
        baseline = build_model("regression", backend="baseline", random_state=args.seed, cat_features=cat_features, dt_features=dt_features)
        metrics_b = evaluate_model(
            baseline,
            train,
            val,
            test,
            feature_cols=feature_cols,
            target_col=target,
            kind="regression",
            target_transform=baseline_transform,
            target_transform_offset=args.target_transform_offset,
            return_predictions=True,
        )
        test_base = test.copy()
        pred_base = np.asarray(metrics_b.pop("pred_test_orig"), dtype=float)
        metrics_b.pop("pred_train_orig", None)
        metrics_b.pop("pred_val_orig", None)
        metrics_b.pop("y_val_true", None)
        metrics_b.pop("y_test_true", None)
        test_base["_pred"] = pred_base
        metrics_b["kind"] = baseline.__class__.__name__
        metrics_b["target_transform"] = baseline_transform
        metrics_b.update(_prediction_quality_flags(test_base[target].values, test_base["_pred"].values))
        _write_output_artifacts(out_dir, "baseline", test_base, target, metrics_b)
        metrics_b["backend"] = "baseline"
        metrics_b["label"] = "baseline"
        artifacts["baseline"] = metrics_b

        if tabfm_scored_test is not None:
            guarded = tabfm_scored_test[[ "facility_id", "timestamp", target]].copy()
            tabfm_pred = np.asarray(tabfm_scored_test["_pred"], dtype=float)
            baseline_pred = np.asarray(test_base["_pred"], dtype=float)
            guarded["_pred"] = np.maximum(tabfm_pred, baseline_pred)
            guarded_metrics = _regression_metrics_from_predictions(
                guarded[target].values,
                guarded["_pred"].values,
            )
            guarded_metrics.update(_prediction_quality_flags(guarded[target].values, guarded["_pred"].values))
            guarded_metrics.update(
                {
                    "kind": "TabFMGuardedForecast",
                    "backend": "tabfm_guarded",
                    "label": "tabfm_guarded",
                    "target_transform": selected_transform,
                    "guardrail": "max(tabfm_prediction, baseline_prediction)",
                    "source_tabfm_rmse_test": float(artifacts["tabfm"].get("rmse_test", float("nan"))),
                    "source_baseline_rmse_test": float(metrics_b.get("rmse_test", float("nan"))),
                    "forecast_model_selected": "tabfm_guarded",
                    "forecast_guardrail_reason": "raw_tabfm_flat_or_lower_than_baseline",
                    "raw_tabfm_flat_prediction_warning": bool(artifacts["tabfm"].get("flat_prediction_warning", False)),
                    "selected_forecast_rmse": float(guarded_metrics.get("rmse_test", float("nan"))),
                    "baseline_forecast_rmse": float(metrics_b.get("rmse_test", float("nan"))),
                }
            )
            _write_output_artifacts(out_dir, "tabfm_guarded", guarded, target, guarded_metrics)
            artifacts["tabfm_guarded"] = guarded_metrics

    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
