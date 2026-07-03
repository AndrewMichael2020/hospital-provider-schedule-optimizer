from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from tabfm_healthcare_eval.data import facility_time_split
from tabfm_healthcare_eval.models import build_model, evaluate_model
from tabfm_healthcare_eval.utils import safe_json_dump, set_deterministic_seed
from scripts.show_top_tables import print_top_for_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build three forecast scenarios for budget-constrained ER scheduling")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--tabfm-forecast", required=True)
    parser.add_argument("--out", default="outputs/scenarios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-transform", choices=["none", "log1p", "sqrt"], default="sqrt")
    parser.add_argument("--target-transform-offset", type=float, default=1e-6)
    parser.add_argument("--print-top-n", type=int, default=5)
    return parser.parse_args()


def _is_categorical_like(series: pd.Series) -> bool:
    return ptypes.is_string_dtype(series) or ptypes.is_object_dtype(series) or ptypes.is_categorical_dtype(series)


def _feature_lists(df: pd.DataFrame, target: str) -> tuple[list[str], list[str], list[str]]:
    feature_cols = [c for c in df.columns if c not in {target, "shortage_risk_next_shift"}]
    dt_features = [c for c in feature_cols if c == "timestamp" or ptypes.is_datetime64_any_dtype(df[c])]
    model_features = [c for c in feature_cols if c not in dt_features]
    cat_features = [c for c in model_features if _is_categorical_like(df[c])]
    return model_features, dt_features, cat_features


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    pred_std = float(np.std(y_pred))
    target_std = float(np.std(y_true))
    return {
        "rmse_test": float(np.sqrt(np.mean(err ** 2))),
        "mae_test": float(np.mean(np.abs(err))),
        "r2_test": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "prediction_std": pred_std,
        "target_std": target_std,
        "prediction_std_ratio": float(pred_std / max(target_std, 1e-9)),
        "flat_prediction_warning": bool(pred_std / max(target_std, 1e-9) < 0.20),
    }


def _fit_predict_backend(name: str, df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], dt_features: list[str], cat_features: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    target = "expected_arrivals_next_hour"
    model = build_model("regression", backend=name, random_state=args.seed, cat_features=cat_features, dt_features=dt_features)
    split_metrics = evaluate_model(
        model,
        train,
        val,
        test,
        feature_cols=feature_cols,
        target_col=target,
        kind="regression",
        target_transform=args.target_transform,
        target_transform_offset=args.target_transform_offset,
        return_predictions=True,
    )
    test_pred = np.asarray(split_metrics.pop("pred_test_orig"), dtype=float)
    y_test = np.asarray(test[target], dtype=float)
    split_score = _metrics(y_test, test_pred)
    split_score.update({"model": name, "fit_seconds": float(split_metrics.get("fit_seconds", 0.0)), "pred_seconds": float(split_metrics.get("pred_seconds", 0.0))})

    full_model = build_model("regression", backend=name, random_state=args.seed, cat_features=cat_features, dt_features=dt_features)
    full_metrics = evaluate_model(
        full_model,
        df,
        df,
        df,
        feature_cols=feature_cols,
        target_col=target,
        kind="regression",
        target_transform=args.target_transform,
        target_transform_offset=args.target_transform_offset,
        return_predictions=True,
    )
    full_pred = np.asarray(full_metrics.pop("pred_test_orig"), dtype=float)
    out = df[["facility_id", "timestamp", target]].copy()
    out["y_pred"] = np.maximum(full_pred, 0.0)
    out["y_true"] = df[target].astype(float).values
    return out, split_score


def main() -> None:
    args = parse_args()
    set_deterministic_seed(args.seed)
    in_path = Path(args.in_path)
    df = pd.read_parquet(in_path) if in_path.suffix.lower() == ".parquet" else pd.read_csv(in_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    target = "expected_arrivals_next_hour"
    feature_cols, dt_features, cat_features = _feature_lists(df, target)
    train, val, test = facility_time_split(df, facility_col="facility_id", timestamp_col="timestamp", test_size=0.2, val_size=0.2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tabfm = pd.read_csv(args.tabfm_forecast)
    tabfm["timestamp"] = pd.to_datetime(tabfm["timestamp"])
    tabfm = tabfm[["facility_id", "timestamp", target, "y_pred", "y_true"]].copy()
    tabfm_metrics = _metrics(tabfm["y_true"].values, tabfm["y_pred"].values)
    tabfm_metrics.update({"scenario": "tabfm_guarded", "model_components": "TabFM guarded by baseline", "scheduling_safe": bool(not tabfm_metrics["flat_prediction_warning"])})

    hgb_pred, hgb_metrics = _fit_predict_backend("baseline", df, train, val, test, feature_cols, dt_features, cat_features, args)
    et_pred, et_metrics = _fit_predict_backend("extratrees", df, train, val, test, feature_cols, dt_features, cat_features, args)
    weights_raw = np.array([1.0 / max(hgb_metrics["rmse_test"], 1e-6), 1.0 / max(et_metrics["rmse_test"], 1e-6)], dtype=float)
    weights = weights_raw / weights_raw.sum()
    ensemble = hgb_pred.copy()
    ensemble["y_pred"] = weights[0] * hgb_pred["y_pred"].values + weights[1] * et_pred["y_pred"].values
    ensemble_metrics = _metrics(ensemble["y_true"].values, ensemble["y_pred"].values)
    ensemble_metrics.update({
        "scenario": "statistical_ensemble",
        "model_components": f"HistGradientBoosting weight={weights[0]:.3f}; ExtraTrees weight={weights[1]:.3f}",
        "hgb_rmse_test": hgb_metrics["rmse_test"],
        "extratrees_rmse_test": et_metrics["rmse_test"],
        "scheduling_safe": bool(not ensemble_metrics["flat_prediction_warning"]),
    })

    residual = np.asarray(test[target], dtype=float) - np.asarray(hgb_pred.loc[test.index.intersection(hgb_pred.index), "y_pred"] if len(test.index.intersection(hgb_pred.index)) == len(test) else np.zeros(len(test)), dtype=float)
    uplift = float(max(0.5, np.nanpercentile(np.abs(residual), 75))) if len(residual) else 1.0
    conservative = ensemble.copy()
    conservative["y_pred"] = np.maximum.reduce([ensemble["y_pred"].values, tabfm["y_pred"].values]) + uplift
    conservative_metrics = _metrics(conservative["y_true"].values, conservative["y_pred"].values)
    conservative_metrics.update({
        "scenario": "conservative_peak",
        "model_components": "max(tabfm_guarded, statistical_ensemble) + p75 residual uplift",
        "residual_uplift": uplift,
        "scheduling_safe": bool(not conservative_metrics["flat_prediction_warning"]),
    })

    artifacts = [
        ("tabfm_guarded", tabfm, tabfm_metrics),
        ("statistical_ensemble", ensemble, ensemble_metrics),
        ("conservative_peak", conservative, conservative_metrics),
    ]
    rows = []
    for name, pred, metrics in artifacts:
        pred = pred.sort_values(["facility_id", "timestamp"]).reset_index(drop=True)
        pred.to_csv(out_dir / f"scenario_{name}.csv", index=False)
        safe_json_dump(metrics, str(out_dir / f"scenario_{name}.metrics.json"))
        row = {"scenario": name, **metrics, "forecast_csv": str(out_dir / f"scenario_{name}.csv")}
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "forecast_scenario_summary.csv", index=False)
    safe_json_dump({"scenarios": rows}, str(out_dir / "forecast_scenario_summary.json"))
    print_top_for_paths([str(out_dir / "forecast_scenario_summary.csv")] + [str(out_dir / f"scenario_{n}.csv") for n, _, _ in artifacts], top_n=args.print_top_n)


if __name__ == "__main__":
    main()
