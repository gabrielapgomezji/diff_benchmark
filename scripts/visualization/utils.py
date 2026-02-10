from __future__ import annotations

from typing import Iterable, List, Tuple
import re

import numpy as np
import pandas as pd


DEFAULT_COMBOS = [
    ("hcp", "Gender", "binary_classification"),
    ("camcan", "Age", "regression"),
    ("camcan", "Gender", "binary_classification"),
    ("abide", "DX_GROUP", "binary_classification"),
    ("abide", "fiq", "regression"),
    ("abide", "viq", "regression"),
    ("abide", "piq", "regression"),
    ("abide", "srs_total_t", "regression"),
    ("abide", "ados_g_total", "regression"),
    ("abide", "ados_g_stereo_behav", "regression"),
    ("abide", "ados_g_social", "regression"),
    ("abide", "ados_g_comm", "regression"),
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
    # We used to split dummy models here, but technically a dummy model is just a model.
    # The caller can filter dummies if they want.
    # However, existing logic (strip_plots) might rely on this, but it seems to just take 'selected'
    # Actually the logic below "If not dummy.empty ... concat" seems to try to pick ONLY the best dummy? 
    # But groupby already does that per model_name. "dummy" is a model_name prefix.
    # If we have "dummy_mean" and "dummy_median", they are different models.
    # The previous code logic:
    # dummy = selected... startswith("dummy")
    # non_dummy = selected...
    # if not dummy.empty: keep only ONE best dummy.
    # This might be desired. I will keep it for compatibility.

    dummy = selected[selected["model_name"].astype(str).str.startswith("dummy")]
    non_dummy = selected[~selected["model_name"].astype(str).str.startswith("dummy")]

    if not dummy.empty:
        if higher_is_better:
            best_dummy = dummy.loc[dummy["_fold_mean"].idxmax()]
        else:
            best_dummy = dummy.loc[dummy["_fold_mean"].idxmin()]
        selected = pd.concat([non_dummy, pd.DataFrame([best_dummy])], ignore_index=True)

    return selected


def get_display_label(dataset: str, target: str, task: str, metric_label: str = None) -> str:
    # We want to format as "Dataset - Target"
    # User Request: "reduce the dataset -task description: no need to display whether it's a binary classificaiton or a regression. For instance camcan- Age is enough"
    # But later: "on the y axis replace the task name by the metric used (for regression show "R^2", for binary classification show "Acc")"
    
    # So we want: "Dataset - Target (Metric)"
    
    base = format_label(f"{dataset}|{target}")
    
    # Check metric
    metric_short = ""
    if metric_label:
        if "Accuracy" in metric_label or "Acc" in metric_label:
            metric_short = "Acc"
        elif "R2" in metric_label:
            metric_short = "R²"
        elif "MAE" in metric_label:
            metric_short = "MAE"
        elif "RMSE" in metric_label:
            metric_short = "RMSE"
    
    # Fallback if metric_label not provided but task is known?
    if not metric_short and task:
        if "classification" in task:
            metric_short = "Acc"
        elif "regression" in task:
            metric_short = "R²" # Default assumption? Or MAE? 
            # In choose_fold_metric we see MAE is default for regression unless R2 exists.
            # But user explicitly asked for "R^2" for regression in the prompt example.
            
    if metric_short:
        return f"{base} ({metric_short})"
    
    return base


def calculate_paired_ttest(
    vals_main: np.ndarray,
    vals_other: np.ndarray,
) -> float:
    from scipy.stats import ttest_rel
    # Ensure they are valid
    if len(vals_main) != len(vals_other):
        return 1.0
    
    # ttest_rel
    res = ttest_rel(vals_main, vals_other)
    if np.isnan(res.pvalue):
        return 1.0
    return float(res.pvalue)


def calculate_paired_stats(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    higher_is_better: bool,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Calculates statistics for paired comparison (A vs B).
    Returns (t_score, mean_diff, std_diff, normalized_diffs)
    """
    # Ensure floats
    vals_a = vals_a.astype(float)
    vals_b = vals_b.astype(float)

    # Calculate difference
    diffs = vals_a - vals_b
    
    # Adjust sign so positive always means A is better (if that is the intent)
    # Actually, usually "Diff" means A - B.
    # If Higher is Better: A=0.9, B=0.8 => A is better. Diff = 0.1 (Pos). Correct.
    # If Lower is Better (MAE): A=2, B=5 => A is better. Diff = -3 (Neg). 
    # To make "Positive t-score" mean "A is better", we must invert diffs if lower is better.
    if not higher_is_better:
        diffs = -diffs

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) # Sample std dev

    if std_diff == 0:
        # Avoid division by zero
        if mean_diff == 0:
            return 0.0, 0.0, 0.0, np.zeros_like(diffs)
        # If mean != 0 but std == 0, it is a constant difference.
        # T-score is infinite. Return a large number or 0?
        # For visualization purposes, let's return 0 to avoid breaking plots, 
        # or handle it upstream.
        return 0.0, mean_diff, 0.0, np.zeros_like(diffs)

    t_score = mean_diff / std_diff
    normalized_diffs = diffs / std_diff
    
    return t_score, mean_diff, std_diff, normalized_diffs



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
