from __future__ import annotations

import inspect
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _build_tabfm_regressor_or_classifier(
    kind: str, random_state: int, strict_backend: str | None = None
):
    """Instantiate a TabFM regressor or classifier model and the requested base model."""
    # Prefer PyTorch first for reliability in this environment.
    model_builder_order = ("pytorch", "jax") if strict_backend is None else (strict_backend,)

    last_error = None
    for backend in model_builder_order:
        if backend == "pytorch":
            try:
                from tabfm.src.pytorch import tabfm_v1_0_0 as pytorch_tabfm
                from tabfm import TabFMRegressor, TabFMClassifier

                if kind == "regression":
                    base_model = pytorch_tabfm.TabFM(is_classifier=False)
                    return TabFMRegressor(model=base_model, random_state=random_state)
                return TabFMClassifier(model=pytorch_tabfm.TabFM(is_classifier=True), random_state=random_state)
            except Exception as exc:
                last_error = exc
                if strict_backend == "pytorch":
                    raise
                continue

        if backend == "jax":
            try:
                from tabfm.src.jax import tabfm_v1_0_0 as jax_tabfm
                from tabfm import TabFMRegressor, TabFMClassifier

                if kind == "regression":
                    base_model = jax_tabfm.TabFM()
                    return TabFMRegressor(model=base_model, random_state=random_state)
                return TabFMClassifier(model=jax_tabfm.TabFM(), random_state=random_state)
            except Exception as exc:
                last_error = exc
                if strict_backend == "jax":
                    raise
                continue

    if strict_backend is not None:
        raise RuntimeError(f"failed to initialize tabfm backend {strict_backend}") from last_error
    return None


def _filter_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    params = inspect.signature(cls.__init__)
    return {k: v for k, v in kwargs.items() if k in params.parameters}


def build_model(
    kind: str,
    backend: str = "auto",
    random_state: int = 42,
    cat_features=None,
    dt_features=None,
    tabfm_n_estimators: int = 64,
    tabfm_num_folds_for_cv: int = 3,
):
    if cat_features is None:
        cat_features = []
    if dt_features is None:
        dt_features = []
    if backend in {"skip", "none"}:
        raise RuntimeError("backend set to skip")
    # TabFM candidates (when installed)
    if backend in {"auto", "tabfm", "jax", "pytorch"}:
        try:
            model = None
            if backend in {"tabfm", "jax"}:
                model = _build_tabfm_regressor_or_classifier(
                    kind=kind,
                    random_state=random_state,
                    strict_backend="jax" if backend == "jax" else "pytorch",
                )
            elif backend == "pytorch":
                model = _build_tabfm_regressor_or_classifier(
                    kind=kind, random_state=random_state, strict_backend="pytorch"
                )
            else:
                model = _build_tabfm_regressor_or_classifier(
                    kind=kind, random_state=random_state, strict_backend=None
                )
            if model is not None:
                if hasattr(model, "set_params"):
                    try:
                        model.set_params(
                            n_estimators=tabfm_n_estimators,
                            num_folds_for_cv=tabfm_num_folds_for_cv,
                        )
                    except Exception:
                        pass
                return model
        except Exception:
            if backend == "tabfm":
                raise
            if backend == "jax":
                if backend == "jax":
                    raise

    # Explicitly force XGBoost if requested.
    if backend in {"xgboost", "xgb"}:
        try:
            import xgboost as xgb

            if kind == "classification":
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                return Pipeline(
                    steps=[
                        ("encode", ColumnTransformer([("cat", enc, cat_features)], remainder="passthrough")),
                        ("clf", xgb.XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=-1)),
                    ]
                )
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            return Pipeline(
                steps=[
                    ("encode", ColumnTransformer([("cat", enc, cat_features)], remainder="passthrough")),
                    (
                        "reg",
                        xgb.XGBRegressor(
                            random_state=random_state,
                            objective="reg:squarederror",
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        except Exception:
            if backend == "xgboost":
                raise

    if backend in {"extratrees", "extra_trees", "et"} and kind == "regression":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return Pipeline(
            steps=[
                ("encode", ColumnTransformer([("cat", enc, cat_features)], remainder="passthrough")),
                ("reg", ExtraTreesRegressor(random_state=random_state, n_estimators=80, n_jobs=-1, min_samples_leaf=2)),
            ]
        )

    if backend in {"randomforest", "random_forest", "rf"} and kind == "regression":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return Pipeline(
            steps=[
                ("encode", ColumnTransformer([("cat", enc, cat_features)], remainder="passthrough")),
                ("reg", RandomForestRegressor(random_state=random_state, n_estimators=80, n_jobs=-1, min_samples_leaf=2)),
            ]
        )

    # Baseline model family
    if backend in {"auto", "hgb", "hist", "histgb", "baseline"}:
        pass

    # default fallback
    if kind == "classification":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        model = Pipeline(
            steps=[
                ("encode", ColumnTransformer([("cat", enc, cat_features)], remainder="passthrough")),
                ("clf", HistGradientBoostingClassifier(random_state=random_state)),
            ]
        )
    else:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        model = Pipeline(
            steps=[
                ("encode", ColumnTransformer([("cat", enc, cat_features)], remainder="passthrough")),
                (
                    "reg",
                    HistGradientBoostingRegressor(random_state=random_state),
                ),
            ]
        )
    return model


def _apply_target_transform(y: pd.Series, mode: str, offset: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if mode == "none":
        return y
    if mode == "log1p":
        return np.log1p(np.maximum(y, 0.0) + offset)
    if mode == "sqrt":
        return np.sqrt(np.maximum(y, 0.0))
    raise ValueError(f"unsupported target transform: {mode}")


def _inverse_target_transform(y: np.ndarray, mode: str, offset: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if mode == "none":
        return np.maximum(y, 0.0)
    if mode == "log1p":
        return np.maximum(np.expm1(y) - offset, 0.0)
    if mode == "sqrt":
        return np.maximum(np.square(y), 0.0)
    raise ValueError(f"unsupported target transform: {mode}")


def evaluate_model(
    model,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols,
    target_col: str,
    kind: str,
    target_transform: str = "none",
    target_transform_offset: float = 1e-6,
    return_predictions: bool = False,
) -> dict:
    X_train = train[feature_cols]
    y_train = _apply_target_transform(train[target_col], target_transform, target_transform_offset)
    X_val = val[feature_cols]
    y_val = _apply_target_transform(val[target_col], target_transform, target_transform_offset)
    X_test = test[feature_cols]
    y_test = _apply_target_transform(test[target_col], target_transform, target_transform_offset)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_train = np.asarray(model.predict(X_train), dtype=float)
    pred_val = np.asarray(model.predict(X_val), dtype=float)
    pred_test = np.asarray(model.predict(X_test), dtype=float)
    pred_seconds = time.perf_counter() - t1

    y_train_true = _inverse_target_transform(y_train, target_transform, target_transform_offset)
    y_val_true = _inverse_target_transform(y_val, target_transform, target_transform_offset)
    y_test_true = _inverse_target_transform(y_test, target_transform, target_transform_offset)
    pred_train_orig = _inverse_target_transform(pred_train, target_transform, target_transform_offset)
    pred_val_orig = _inverse_target_transform(pred_val, target_transform, target_transform_offset)
    pred_test_orig = _inverse_target_transform(pred_test, target_transform, target_transform_offset)

    if kind == "classification":
        try:
            prob_val = model.predict_proba(X_val)[:, 1]
            prob_test = model.predict_proba(X_test)[:, 1]
        except Exception:
            prob_val = pred_val.astype(float)
            prob_test = pred_test.astype(float)
        roc_val = roc_auc_score(y_val_true, prob_val) if len(np.unique(y_val_true)) > 1 else float("nan")
        roc_test = roc_auc_score(y_test_true, prob_test) if len(np.unique(y_test_true)) > 1 else float("nan")
        precision, recall, _ = (
            precision_recall_curve(y_test_true, prob_test)
            if len(np.unique(y_test_true)) > 1
            else (np.array([0.0]), np.array([0.0]), np.array([0.0]))
        )
        pr_auc = np.trapezoid(recall, precision) if len(recall) > 1 else float("nan")
        ll_val = log_loss(y_val_true, prob_val) if len(np.unique(y_val_true)) > 1 else float("nan")
        ll_test = log_loss(y_test_true, prob_test) if len(np.unique(y_test_true)) > 1 else float("nan")
        acc = accuracy_score(y_test_true, pred_test)
        f1 = f1_score(y_test_true, pred_test, zero_division=0)
        results = {
            "roc_auc_val": float(roc_val),
            "roc_auc_test": float(roc_test),
            "pr_auc_test": float(pr_auc),
            "log_loss_val": float(ll_val),
            "log_loss_test": float(ll_test),
            "accuracy_test": float(acc),
            "f1_test": float(f1),
            "pred_seconds": pred_seconds,
            "fit_seconds": fit_seconds,
            "pred_sample": pred_test_orig[:5].tolist() if len(pred_test_orig) else [],
        }
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test_true, pred_test_orig)))
        rmse_val = float(np.sqrt(mean_squared_error(y_val_true, pred_val_orig)))
        mae = float(mean_absolute_error(y_test_true, pred_test_orig))
        mae_val = float(mean_absolute_error(y_val_true, pred_val_orig))
        r2 = float(r2_score(y_test_true, pred_test_orig)) if np.isfinite(pred_test_orig).all() else float("nan")
        r2_val = float(r2_score(y_val_true, pred_val_orig)) if np.isfinite(pred_val_orig).all() else float("nan")
        results = {
            "rmse_val": rmse_val,
            "rmse_test": rmse,
            "mae_val": mae_val,
            "mae_test": mae,
            "r2_val": r2_val,
            "r2_test": r2,
            "pred_seconds": pred_seconds,
            "fit_seconds": fit_seconds,
            "pred_sample": pred_test_orig[:5].tolist() if len(pred_test_orig) else [],
        }

    if return_predictions:
        results["pred_train_orig"] = pred_train_orig
        results["pred_val_orig"] = pred_val_orig
        results["pred_test_orig"] = pred_test_orig
        results["y_val_true"] = y_val_true
        results["y_test_true"] = y_test_true

    return results
