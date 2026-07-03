from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pandas.api.types as ptypes


def identify_feature_columns(
    df: pd.DataFrame, targets: tuple[str, str], datetime_col: str = "timestamp"
) -> tuple[list[str], list[str]]:
    target_set = set(targets)
    feature_cols = [c for c in df.columns if c not in target_set]
    dt_features = [c for c in feature_cols if c == datetime_col]
    cat_features = []
    for c in feature_cols:
        if c == datetime_col or c in dt_features:
            continue
        if pd.api.types.is_categorical_dtype(df[c]) or df[c].dtype == "object":
            cat_features.append(c)
    num_features = [c for c in feature_cols if c not in cat_features and c not in dt_features]
    return num_features + [c for c in cat_features], num_features, dt_features, cat_features


def temporal_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df)
    if n < 10:
        raise ValueError("Need at least 10 rows for split.")
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def facility_time_split(
    df: pd.DataFrame,
    facility_col: str = "facility_id",
    timestamp_col: str = "timestamp",
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Split each facility separately by timestamp to avoid facility-time leakage.
    chunks = []
    val_chunks = []
    test_chunks = []
    train_chunks = []
    for _, g in df.groupby(facility_col, group_keys=False):
        g = g.sort_values(timestamp_col)
        g_train, g_val, g_test = temporal_split(
            g,
            timestamp_col=timestamp_col,
            test_size=test_size,
            val_size=val_size,
        )
        train_chunks.append(g_train)
        val_chunks.append(g_val)
        test_chunks.append(g_test)
    train = pd.concat(train_chunks).sort_values(timestamp_col).reset_index(drop=True)
    val = pd.concat(val_chunks).sort_values(timestamp_col).reset_index(drop=True)
    test = pd.concat(test_chunks).sort_values(timestamp_col).reset_index(drop=True)
    return train, val, test


def evaluate_dataset_contract(df: pd.DataFrame, timestamp_col: str = "timestamp") -> dict:
    issues = []
    facility_issues = []
    if "facility_id" in df.columns and timestamp_col in df.columns:
        for facility, g in df.groupby("facility_id"):
            ts = pd.to_datetime(g[timestamp_col])
            if not ts.is_monotonic_increasing:
                facility_issues.append(facility)
        if facility_issues:
            issues.append(f"non-monotonic timestamps in facilities: {facility_issues[:5]}")
    if not pd.to_datetime(df[timestamp_col]).is_monotonic_increasing:
        issues.append("global timestamps not monotonic")

    target_cols = [c for c in ["expected_arrivals_next_hour", "shortage_risk_next_shift"] if c in df.columns]
    for t in target_cols:
        if t == "expected_arrivals_next_hour":
            if (df[t] < 0).any():
                issues.append("negative arrivals target found")
        if t == "shortage_risk_next_shift":
            vals = df[t].dropna()
            if not set(vals.unique()).issubset({0, 1}):
                issues.append("shortage target non-binary")

    cat_cols = [
        c
        for c in df.columns
        if ptypes.is_string_dtype(df[c]) or ptypes.is_categorical_dtype(df[c]) or ptypes.is_object_dtype(df[c])
    ]
    numeric_cols = [c for c in df.columns if c not in cat_cols + [timestamp_col] and ptypes.is_numeric_dtype(df[c])]
    missing_frac = float(df.isna().mean().mean())
    outlier_rate = float((df[numeric_cols].abs() > 1e6).mean().mean()) if numeric_cols else 0.0

    result = {
        "n_rows": int(len(df)),
        "n_facilities": int(df["facility_id"].nunique()) if "facility_id" in df.columns else 0,
        "missing_fraction": missing_frac,
        "outlier_rate": outlier_rate,
        "facility_monotonicity_violations": facility_issues,
        "issues": issues,
    }
    return result


def assert_no_label_leakage(
    df: pd.DataFrame,
    facility_col: str = "facility_id",
    timestamp_col: str = "timestamp",
    min_gap_steps: int = 1,
) -> None:
    ts = pd.to_datetime(df[timestamp_col])
    for facility, g in df.groupby(facility_col):
        idx = g.sort_values(timestamp_col).index.to_list()
        diffs = np.diff(np.array(idx))
        if (diffs < 0).any():
            raise AssertionError("facility rows are not contiguous/ordered")
    if min_gap_steps <= 0:
        return
    sorted_df = df.sort_values([facility_col, timestamp_col])
    duplicates = sorted_df.duplicated(subset=[facility_col, timestamp_col])
    if duplicates.any():
        raise AssertionError("duplicate timestamp within facility found")
