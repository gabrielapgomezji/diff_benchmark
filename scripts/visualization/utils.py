from __future__ import annotations

from typing import Iterable, List, Tuple
import re

import numpy as np
import pandas as pd


DEFAULT_COMBOS = [
    ("hcp", "Gender", "binary_classification"),
    ("camcan", "Age", "regression"),
    ("camcan", "Gender", "binary_classification"),
]


def clean_target(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    s = re.sub(r"[\[\]']", "", s).strip()
    s = s.replace("Age_in_Yrs", "Age")
    s = s.replace("dx_group", "DX_GROUP")
    s = s.replace("DX_GROUP", "DX_GROUP")
    s = s.replace("Gender", "Gender")
    return s


def format_label(text: str) -> str:
    s = text.replace("|", " - ")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def filter_combos(df: pd.DataFrame, combos: Iterable[Tuple[str, str, str]]) -> pd.DataFrame:
    combos = list(combos)
    if not combos:
        return df
    mask = pd.Series(False, index=df.index)
    for dataset, target, task in combos:
        mask |= (
            (df["dataset"] == dataset)
            & (df["target_clean"] == target)
            & (df["prediction_task"] == task)
        )
    return df[mask].copy()


def choose_fold_metric(group: pd.DataFrame, prediction_task: str) -> Tuple[str, str, bool]:
    if prediction_task == "binary_classification":
        if any(col.startswith("accuracy_weighted_test_fold") for col in group.columns):
            return "accuracy_weighted_test_fold", "Balanced Accuracy", True
        return "accuracy_test_fold", "Accuracy", True
    if any(col.startswith("r2_test_fold") for col in group.columns):
        return "r2_test_fold", "R2", True
    return "mae_test_fold", "MAE", False


def fold_columns(group: pd.DataFrame, prefix: str, max_folds: int = 5) -> List[str]:
    cols = [f"{prefix}{i}" for i in range(max_folds) if f"{prefix}{i}" in group.columns]
    return cols


def model_label(row: pd.Series) -> str:
    name = str(row.get("model_name", ""))
    if name.startswith("dummy"):
        return "Dummy Baseline"
    parts = [name, str(row.get("primary_metric", "")), str(row.get("tissue_type", ""))]
    return format_label("|".join(parts))


def is_dummy_model(name: str) -> bool:
    return str(name).startswith("dummy")


def score_from_metric(value: float, higher_is_better: bool) -> float:
    return float(value) if higher_is_better else -float(value)


def zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0.0 or np.isnan(std):
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def select_best_runs(
    df: pd.DataFrame,
    fold_prefix: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    fold_cols = fold_columns(df, fold_prefix)
    if not fold_cols:
        return df.copy()

    df = df.copy()
    df["_fold_mean"] = df[fold_cols].mean(axis=1, skipna=True)
    df = df[~df["_fold_mean"].isna()].copy()

    group_cols = ["model_name", "primary_metric", "tissue_type"]

    best_rows = []
    for _, group in df.groupby(group_cols, dropna=False):
        if group.empty:
            continue
        if higher_is_better:
            idx = group["_fold_mean"].idxmax()
        else:
            idx = group["_fold_mean"].idxmin()
        best_rows.append(group.loc[idx])

    if not best_rows:
        return df.copy()

    selected = pd.DataFrame(best_rows)
    dummy = selected[selected["model_name"].astype(str).str.startswith("dummy")]
    non_dummy = selected[~selected["model_name"].astype(str).str.startswith("dummy")]

    if not dummy.empty:
        if higher_is_better:
            best_dummy = dummy.loc[dummy["_fold_mean"].idxmax()]
        else:
            best_dummy = dummy.loc[dummy["_fold_mean"].idxmin()]
        selected = pd.concat([non_dummy, pd.DataFrame([best_dummy])], ignore_index=True)

    return selected


def build_strip_data(
    df: pd.DataFrame,
    prediction_task: str,
    fold_prefix: str,
    higher_is_better: bool,
    best_run: bool,
) -> pd.DataFrame:
    group = df.copy()
    if best_run:
        group = select_best_runs(group, fold_prefix, higher_is_better)

    fold_cols = fold_columns(group, fold_prefix)
    if not fold_cols:
        return pd.DataFrame()

    rows = []
    for _, row in group.iterrows():
        label = model_label(row)
        for fold_idx, col in enumerate(fold_cols):
            val = row.get(col)
            if pd.notna(val):
                rows.append({"label": label, "fold": fold_idx, "value": float(val)})

    return pd.DataFrame(rows)
