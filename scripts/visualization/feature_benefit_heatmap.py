"""
feature_benefit_heatmap.py
==========================

Heatmap answering: "For each dataset-task and feature, how many models exhibit
a consistent improvement vs b0?"

Two side-by-side panels:
  Left  — p-value method  (one-sample t-test on cluster means, p < 0.05 & mean > 0)
  Right — robustness rule (median Δ ≥ epsilon AND frac_positive ≥ tau)

See docstring of `plot_feature_benefit_heatmap` for full pipeline description.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import MICCAI_DOUBLE_COLUMN_FIGSIZE, apply_miccai_style
from utils import (
    choose_spread_metric,
    clean_target,
    filter_combos,
    fold_columns,
    map_model_family,
    normalize_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_DELTA_COMBOS = [
    ("hcp", "Gender", "binary_classification"),
    ("camcan", "Gender", "binary_classification"),
    ("camcan", "Age", "regression"),
    ("abide", "DX_GROUP", "binary_classification"),
]
FEATURE_DELTA_TISSUES = {"white", "gray"}
FEATURE_PREFERRED_ORDER = ["md", "mk", "sh"]
FEATURE_EXCLUDE = {"rtop"}

# Robustness-rule thresholds
EPSILON: float = 0.03   # minimum median delta to count as benefit
TAU: float = 0.80       # minimum fraction of positive-delta folds

PVALUE_THRESHOLD: float = 0.05

HEATMAP_CMAP = "Blues"

TITLE_MAIN = "Proportion of models consistently improving over b0, per feature and dataset-task"
TITLE_LEFT = "Statistical criterion\n(t-test, p<0.05 & mean\u0394>0)"
TITLE_RIGHT = "Practical criterion\n(med.\u0394\u2265{EPSILON}, \u2265{TAU:.0%} folds positive)".format(EPSILON=EPSILON, TAU=TAU)


# ---------------------------------------------------------------------------
# Step 0: data loading (individual models, no family grouping)
# ---------------------------------------------------------------------------

def _load_scope(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["target_clean"] = df["target"].map(clean_target)
    df = filter_combos(df, FEATURE_DELTA_COMBOS)
    df = df[df["tissue_type"].isin(FEATURE_DELTA_TISSUES)].copy()
    df["model_family"] = df["model_name"].map(map_model_family)
    df = df[df["model_family"].notna()].copy()  # drop dummies
    if df.empty:
        raise RuntimeError("No rows after scope filters")
    return df


# ---------------------------------------------------------------------------
# Step 1: fold-level paired deltas
# ---------------------------------------------------------------------------

def _extract_fold_scores(df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for (dataset, target, task), group in df.groupby(
        ["dataset", "target_clean", "prediction_task"], dropna=False
    ):
        try:
            fold_prefix, _ = choose_spread_metric(group, task)
        except (RuntimeError, ValueError):
            continue
        f_cols = fold_columns(group, fold_prefix)
        if not f_cols:
            continue
        for col in f_cols:
            fold_idx = int(col.replace(fold_prefix, ""))
            part = group[
                ["dataset", "target_clean", "prediction_task",
                 "tissue_type", "model_name", "primary_metric"]
            ].copy()
            part["fold_index"] = fold_idx
            part["score_raw"] = pd.to_numeric(group[col], errors="coerce")
            parts.append(part)

    if not parts:
        raise RuntimeError("No fold-level rows found")

    fold_df = pd.concat(parts, ignore_index=True)
    fold_df = fold_df[fold_df["score_raw"].notna()].copy()
    fold_df["score_norm"] = fold_df.apply(normalize_score, axis=1)
    return fold_df[fold_df["score_norm"].notna()].copy()


def _build_delta_df(fold_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "dataset", "target_clean", "prediction_task",
        "tissue_type", "model_name", "fold_index", "primary_metric",
    ]
    dedup = (
        fold_df.groupby(key_cols, dropna=False, as_index=False)
        .agg(score_norm=("score_norm", "mean"))
    )

    base_cols = [
        "dataset", "target_clean", "prediction_task",
        "tissue_type", "model_name", "fold_index",
    ]
    wide = (
        dedup.pivot_table(
            index=base_cols,
            columns="primary_metric",
            values="score_norm",
            aggfunc="mean",
        )
        .reset_index()
        .copy()
    )
    if "b0" not in wide.columns:
        raise RuntimeError("Feature 'b0' missing after pivot — cannot build paired deltas")

    wide = wide[wide["b0"].notna()].copy()
    feature_cols = [c for c in wide.columns if c not in set(base_cols + ["b0"])
                    and c not in FEATURE_EXCLUDE]
    feature_cols = _ordered_features(feature_cols)
    if not feature_cols:
        raise RuntimeError("No non-b0 features available after filtering")

    delta_df = wide.melt(
        id_vars=base_cols + ["b0"],
        value_vars=feature_cols,
        var_name="feature",
        value_name="feature_score",
    )
    delta_df = delta_df[delta_df["feature_score"].notna()].copy()
    delta_df["delta"] = delta_df["feature_score"] - delta_df["b0"]
    return delta_df[base_cols + ["feature", "delta"]].copy()


def _ordered_features(features: list[str]) -> list[str]:
    seen = set(features)
    ordered = [f for f in FEATURE_PREFERRED_ORDER if f in seen]
    ordered.extend(sorted(f for f in features if f not in ordered))
    return ordered


# ---------------------------------------------------------------------------
# Step 2: benefit flags per (dataset-task, tissue, model, feature)
# ---------------------------------------------------------------------------

def _benefit_flags(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (dataset, target_clean, prediction_task, tissue_type,
    model_name, feature) with columns benefit_pvalue and benefit_robust."""
    rows: list[dict] = []

    group_cols = [
        "dataset", "target_clean", "prediction_task",
        "tissue_type", "model_name", "feature",
    ]
    for keys, grp in delta_df.groupby(group_cols, dropna=False):
        deltas = grp["delta"].to_numpy(dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size == 0:
            continue

        # --- p-value method ---
        # Collapse folds to one cluster mean then t-test
        # (here each row is already one fold; cluster = the group itself)
        # We treat each fold as an independent observation → ttest_1samp
        benefit_pvalue = 0
        if deltas.size >= 2:
            res = ttest_1samp(deltas, popmean=0.0, nan_policy="omit")
            if np.isfinite(res.pvalue) and res.pvalue < PVALUE_THRESHOLD and deltas.mean() > 0:
                benefit_pvalue = 1

        # --- robustness rule ---
        median_delta = float(np.median(deltas))
        frac_positive = float(np.mean(deltas > 0))
        benefit_robust = int(median_delta >= EPSILON and frac_positive >= TAU)

        row = dict(zip(group_cols, keys))
        row["n_folds"] = int(deltas.size)
        row["mean_delta"] = float(deltas.mean())
        row["median_delta"] = median_delta
        row["frac_positive"] = frac_positive
        row["benefit_pvalue"] = benefit_pvalue
        row["benefit_robust"] = benefit_robust
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: collapse across models
# ---------------------------------------------------------------------------

def _collapse_across_models(flags_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate benefit flags across models for each (dataset-task, feature).

    Tissue types are collapsed (max benefit across tissues per model) before
    counting, so a model is counted once per dataset-task-feature regardless
    of tissue.
    """
    # Per model: take max benefit across tissues (if any tissue shows benefit,
    # count the model as benefiting)
    per_model = (
        flags_df.groupby(
            ["dataset", "target_clean", "prediction_task", "model_name", "feature"],
            dropna=False,
            as_index=False,
        )
        .agg(
            benefit_pvalue=("benefit_pvalue", "max"),
            benefit_robust=("benefit_robust", "max"),
        )
    )

    agg = (
        per_model.groupby(
            ["dataset", "target_clean", "prediction_task", "feature"],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_models_total=("model_name", "nunique"),
            n_models_pvalue=("benefit_pvalue", "sum"),
            n_models_robust=("benefit_robust", "sum"),
        )
    )

    agg["prop_pvalue"] = agg["n_models_pvalue"] / agg["n_models_total"]
    agg["prop_robust"] = agg["n_models_robust"] / agg["n_models_total"]
    agg["row_label"] = (
        agg["dataset"].str.upper() + " – " + agg["target_clean"]
    )
    return agg


def _merge_gender_rows(agg: pd.DataFrame) -> pd.DataFrame:
    """Pool HCP–Gender and CamCAN–Gender into a single combined row.

    Raw counts (n_models_pvalue, n_models_robust, n_models_total) are summed
    across the two datasets so the annotation fractions reflect the full
    combined model pool.  Proportions are then recomputed from the pooled counts.
    """
    gender_mask = agg["target_clean"] == "Gender"
    gender_rows = agg[gender_mask].copy()
    other_rows  = agg[~gender_mask].copy()

    if gender_rows.empty:
        return agg

    pooled = (
        gender_rows
        .groupby("feature", dropna=False, as_index=False)
        .agg(
            n_models_total  =("n_models_total",  "sum"),
            n_models_pvalue =("n_models_pvalue", "sum"),
            n_models_robust =("n_models_robust", "sum"),
        )
    )
    pooled["prop_pvalue"]     = pooled["n_models_pvalue"] / pooled["n_models_total"]
    pooled["prop_robust"]     = pooled["n_models_robust"] / pooled["n_models_total"]
    pooled["row_label"]       = "Gender (HCP+CamCAN)"
    pooled["target_clean"]    = "Gender"
    pooled["dataset"]         = "hcp+camcan"
    pooled["prediction_task"] = "binary_classification"

    return pd.concat([other_rows, pooled], ignore_index=True)


# ---------------------------------------------------------------------------
# Step 4: sort rows and columns
# ---------------------------------------------------------------------------

def _sort_heatmap(agg: pd.DataFrame) -> tuple[list[str], list[str]]:
    # Row order: descending total robust signal
    row_strength = (
        agg.groupby("row_label")["prop_robust"].sum().sort_values(ascending=True)
    )
    row_order = row_strength.index.tolist()  # weakest first → bottom = strongest

    # Column order: descending total robust signal
    col_strength = (
        agg.groupby("feature")["prop_robust"].sum().sort_values(ascending=False)
    )
    col_order = _ordered_features(col_strength.index.tolist())

    return row_order, col_order


# ---------------------------------------------------------------------------
# Step 5: visualisation
# ---------------------------------------------------------------------------

def _pivot_for_heatmap(
    agg: pd.DataFrame, value_col: str, row_order: list[str], col_order: list[str]
) -> pd.DataFrame:
    pivot = agg.pivot_table(
        index="row_label", columns="feature", values=value_col, aggfunc="mean"
    )
    pivot = pivot.reindex(index=row_order, columns=col_order)
    return pivot


def _annot_fraction(agg: pd.DataFrame, row_order: list[str], col_order: list[str]) -> pd.DataFrame:
    """Build annotation matrix showing 'n_benefit / n_total'."""
    num = agg.pivot_table(index="row_label", columns="feature", values="n_models_pvalue", aggfunc="sum")
    den = agg.pivot_table(index="row_label", columns="feature", values="n_models_total", aggfunc="max")
    num = num.reindex(index=row_order, columns=col_order)
    den = den.reindex(index=row_order, columns=col_order)
    annot = num.astype("Int64").astype(str) + "/" + den.astype("Int64").astype(str)
    annot = annot.where(num.notna(), other="")
    return annot


def _annot_fraction_robust(agg: pd.DataFrame, row_order: list[str], col_order: list[str]) -> pd.DataFrame:
    num = agg.pivot_table(index="row_label", columns="feature", values="n_models_robust", aggfunc="sum")
    den = agg.pivot_table(index="row_label", columns="feature", values="n_models_total", aggfunc="max")
    num = num.reindex(index=row_order, columns=col_order)
    den = den.reindex(index=row_order, columns=col_order)
    annot = num.astype("Int64").astype(str) + "/" + den.astype("Int64").astype(str)
    annot = annot.where(num.notna(), other="")
    return annot


def _draw_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    annot: pd.DataFrame,
    title: str,
    show_yticklabels: bool,
    show_cbar: bool,
) -> None:
    import matplotlib as mpl
    fs_base  = mpl.rcParams["font.size"]          # 9
    fs_title = mpl.rcParams["axes.titlesize"]     # 10
    fs_label = mpl.rcParams["axes.labelsize"]     # 9
    fs_tick  = mpl.rcParams["xtick.labelsize"]    # 8

    # Gray for missing cells; "N/A" annotation
    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color="#b0b0b0")
    annot_values = annot.copy()
    annot_values[data.isna()] = "N/A"

    sns.heatmap(
        data,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        annot=annot_values.values,
        fmt="",
        square=False,
        linewidths=0.5,
        linecolor="#cccccc",
        cbar=show_cbar,
        cbar_kws={"shrink": 0.75, "label": "Prop. models benefiting",
                  "format": "%.1f"} if show_cbar else {},
        annot_kws={"size": fs_base},
    )
    ax.set_title(title, pad=2, fontsize=fs_title)
    ax.set_xlabel("", labelpad=4)
    ax.set_ylabel("", labelpad=4)
    ax.tick_params(axis="x", rotation=30, labelsize=fs_tick)
    if show_yticklabels:
        ax.tick_params(axis="y", rotation=0, labelsize=fs_tick)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)
    if show_cbar and ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.ax.tick_params(labelsize=fs_tick)
            cbar.set_label("Prop. models benefiting", fontsize=fs_label)


def _plot_heatmaps(agg: pd.DataFrame, out_file: Path, n_models_total: int) -> None:
    import matplotlib as mpl
    fs_title = mpl.rcParams["figure.titlesize"]   # 10

    row_order, col_order = _sort_heatmap(agg)

    pivot_pvalue = _pivot_for_heatmap(agg, "prop_pvalue", row_order, col_order)
    pivot_robust = _pivot_for_heatmap(agg, "prop_robust", row_order, col_order)

    annot_pvalue = _annot_fraction(agg, row_order, col_order)
    annot_robust = _annot_fraction_robust(agg, row_order, col_order)

    # Enforce publication target size from config (double-column layout).
    fig, axes = plt.subplots(1, 2, figsize=MICCAI_DOUBLE_COLUMN_FIGSIZE)

    _draw_heatmap(axes[0], pivot_pvalue, annot_pvalue,
                  title=TITLE_LEFT, show_yticklabels=True, show_cbar=False)
    _draw_heatmap(axes[1], pivot_robust, annot_robust,
                  title=TITLE_RIGHT, show_yticklabels=False, show_cbar=True)

    fig.suptitle(TITLE_MAIN, y=1.01, fontsize=fs_title, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_feature_benefit_heatmap(
    parquet_path: str,
    out_dir: str = "exp_outputs/summary/plots/features",
    merge_gender: bool = False,
) -> Path:
    """Full pipeline: load → deltas → benefit flags → aggregate → plot."""
    apply_miccai_style()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = _load_scope(parquet_path)
    fold_df = _extract_fold_scores(df)
    delta_df = _build_delta_df(fold_df)

    flags_df = _benefit_flags(delta_df)
    if flags_df.empty:
        raise RuntimeError("No benefit flags computed — check delta pipeline")

    agg = _collapse_across_models(flags_df)

    if merge_gender:
        agg = _merge_gender_rows(agg)

    n_models_total = int(agg["n_models_total"].max())

    # Sanity: warn if n_models_total varies
    model_count_range = agg["n_models_total"].agg(["min", "max"])
    if model_count_range["max"] - model_count_range["min"] > 2:
        warnings.warn(
            f"n_models_total varies: {model_count_range['min']}–{model_count_range['max']}. "
            "Some model/feature combinations may be missing.",
            stacklevel=2,
        )

    out_file = out_path / "feature_benefit_heatmap.png"
    _plot_heatmaps(agg, out_file, n_models_total)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Heatmap: for each dataset-task × feature, proportion of models "
            "that consistently improve over b0."
        )
    )
    parser.add_argument(
        "--input",
        default="exp_outputs/summary/comprehensive_results.parquet",
        help="Input parquet file",
    )
    parser.add_argument(
        "--outdir",
        default="exp_outputs/summary/plots/features",
        help="Output directory",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=EPSILON,
        help=f"Minimum median delta for robustness rule (default {EPSILON})",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=TAU,
        help=f"Minimum fraction of positive-delta folds (default {TAU})",
    )
    parser.add_argument(
        "--merge-gender",
        action="store_true",
        help=(
            "Merge HCP–Gender and CamCAN–Gender into a single row "
            "'Gender (HCP+CamCAN)' by pooling raw model counts."
        ),
    )
    args = parser.parse_args()

    # Allow CLI overrides of thresholds
    import feature_benefit_heatmap as _self
    _self.EPSILON = args.epsilon
    _self.TAU = args.tau

    out_path = plot_feature_benefit_heatmap(
        args.input,
        args.outdir,
        merge_gender=args.merge_gender,
    )
    print("Saved feature benefit heatmap to", out_path)


if __name__ == "__main__":
    main()
