"""
feature_benefit_heatmap_all_tissues_folds.py
============================================

Heatmap answering: "For each dataset-task and feature, how many models improve
over b0 under a robustness criterion, shown separately for gray and white matter?"

Two side-by-side panels:
  Left  — gray matter robustness
  Right — white matter robustness
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import MICCAI_DOUBLE_COLUMN_FIGSIZE, MICCAI_MPL_PARAMS, apply_miccai_style
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

EPSILON: float = 0.05
TAU: float = 0.80

HEATMAP_CMAP = "Blues"

TITLE_MAIN = "Model benefit over b0 per feature and dataset-task by tissue"


# ---------------------------------------------------------------------------
# Step 0: data loading
# ---------------------------------------------------------------------------


def _load_scope(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["target_clean"] = df["target"].map(clean_target)
    df = filter_combos(df, FEATURE_DELTA_COMBOS)
    df = df[df["tissue_type"].isin(FEATURE_DELTA_TISSUES)].copy()
    df["model_family"] = df["model_name"].map(map_model_family)
    df = df[df["model_family"].notna()].copy()
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
                [
                    "dataset",
                    "target_clean",
                    "prediction_task",
                    "tissue_type",
                    "model_name",
                    "primary_metric",
                ]
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


def _ordered_features(features: list[str]) -> list[str]:
    seen = set(features)
    ordered = [f for f in FEATURE_PREFERRED_ORDER if f in seen]
    ordered.extend(sorted(f for f in features if f not in ordered))
    return ordered


def _build_delta_df(fold_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "dataset",
        "target_clean",
        "prediction_task",
        "tissue_type",
        "model_name",
        "fold_index",
        "primary_metric",
    ]
    dedup = fold_df.groupby(key_cols, dropna=False, as_index=False).agg(
        score_norm=("score_norm", "mean")
    )

    base_cols = [
        "dataset",
        "target_clean",
        "prediction_task",
        "tissue_type",
        "model_name",
        "fold_index",
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
        raise RuntimeError("Feature 'b0' missing after pivot")

    wide = wide[wide["b0"].notna()].copy()
    feature_cols = [
        c
        for c in wide.columns
        if c not in set(base_cols + ["b0"]) and c not in FEATURE_EXCLUDE
    ]
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
    return delta_df[
        [
            "dataset",
            "target_clean",
            "prediction_task",
            "tissue_type",
            "model_name",
            "fold_index",
            "feature",
            "delta",
        ]
    ].copy()


# ---------------------------------------------------------------------------
# Step 2: per-model robustness flags within each tissue (across folds)
# ---------------------------------------------------------------------------


def _per_model_robust_stats(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = [
        "dataset",
        "target_clean",
        "prediction_task",
        "tissue_type",
        "model_name",
        "feature",
    ]
    for keys, grp in delta_df.groupby(group_cols, dropna=False):
        deltas = grp["delta"].to_numpy(dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size == 0:
            continue

        median_delta = float(np.median(deltas))
        frac_positive = float(np.mean(deltas > 0))

        row = dict(zip(group_cols, keys))
        row["n_obs"] = int(deltas.size)
        row["median_delta"] = median_delta
        row["frac_positive"] = frac_positive
        row["benefit_robust"] = int(median_delta >= EPSILON and frac_positive >= TAU)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: collapse across models
# ---------------------------------------------------------------------------


def _collapse_across_models(per_model: pd.DataFrame) -> pd.DataFrame:
    agg = per_model.groupby(
        ["dataset", "target_clean", "prediction_task", "tissue_type", "feature"],
        dropna=False,
        as_index=False,
    ).agg(
        n_models_total=("model_name", "nunique"),
        n_models_robust=("benefit_robust", "sum"),
    )

    agg["prop_robust"] = agg["n_models_robust"] / agg["n_models_total"]
    agg["row_label"] = agg["dataset"].str.upper() + " – " + agg["target_clean"]
    return agg


def _merge_gender_rows(agg: pd.DataFrame) -> pd.DataFrame:
    gender_mask = agg["target_clean"] == "Gender"
    gender_rows = agg[gender_mask].copy()
    other_rows = agg[~gender_mask].copy()
    if gender_rows.empty:
        return agg

    pooled = gender_rows.groupby(
        ["tissue_type", "feature"], dropna=False, as_index=False
    ).agg(
        n_models_total=("n_models_total", "sum"),
        n_models_robust=("n_models_robust", "sum"),
    )
    pooled["prop_robust"] = pooled["n_models_robust"] / pooled["n_models_total"]
    pooled["row_label"] = "Gender (HCP+CamCAN)"
    pooled["target_clean"] = "Gender"
    pooled["dataset"] = "hcp+camcan"
    pooled["prediction_task"] = "binary_classification"
    return pd.concat([other_rows, pooled], ignore_index=True)


# ---------------------------------------------------------------------------
# Step 4: ordering and annotations
# ---------------------------------------------------------------------------


def _sort_heatmap(agg: pd.DataFrame) -> tuple[list[str], list[str]]:
    row_strength = (
        agg.groupby("row_label")["prop_robust"].sum().sort_values(ascending=True)
    )
    row_order = row_strength.index.tolist()

    col_strength = (
        agg.groupby("feature")["prop_robust"].sum().sort_values(ascending=False)
    )
    col_order = _ordered_features(col_strength.index.tolist())
    return row_order, col_order


def _pivot_for_heatmap(
    agg: pd.DataFrame, row_order: list[str], col_order: list[str]
) -> pd.DataFrame:
    pivot = agg.pivot_table(
        index="row_label", columns="feature", values="prop_robust", aggfunc="mean"
    )
    return pivot.reindex(index=row_order, columns=col_order)


def _annot_fraction(
    agg: pd.DataFrame,
    row_order: list[str],
    col_order: list[str],
) -> pd.DataFrame:
    num = agg.pivot_table(
        index="row_label", columns="feature", values="n_models_robust", aggfunc="sum"
    )
    den = agg.pivot_table(
        index="row_label", columns="feature", values="n_models_total", aggfunc="max"
    )
    num = num.reindex(index=row_order, columns=col_order)
    den = den.reindex(index=row_order, columns=col_order)
    pct = (num / den * 100).round(0).astype("Int64")
    annot = pct.astype(str) + "%"
    return annot.where(num.notna(), other="")


# ---------------------------------------------------------------------------
# Step 5: plotting
# ---------------------------------------------------------------------------


def _draw_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    annot: pd.DataFrame,
    title: str,
    show_yticklabels: bool,
    show_cbar: bool,
) -> None:
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
        cbar_kws=(
            {"shrink": 0.75, "label": "Prop. models benefiting", "format": "%.1f"}
            if show_cbar
            else {}
        ),
        annot_kws={"size": MICCAI_MPL_PARAMS["font.size"]},
    )
    ax.set_title(title, pad=2, fontsize=MICCAI_MPL_PARAMS["axes.titlesize"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=MICCAI_MPL_PARAMS["xtick.labelsize"])
    if show_yticklabels:
        ax.tick_params(axis="y", rotation=0, labelsize=MICCAI_MPL_PARAMS["ytick.labelsize"])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)
    if show_cbar and ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.set_label("Prop. models benefiting", fontsize=MICCAI_MPL_PARAMS["axes.labelsize"])
            cbar.ax.tick_params(labelsize=MICCAI_MPL_PARAMS["xtick.labelsize"])


def _robust_panel_title(tissue: str) -> str:
    tissue_name = "Gray matter" if tissue == "gray" else "White matter"
    return (
        f"{tissue_name} robustness\n"
        f"(med.\u0394\u2265{EPSILON:g}, \u2265{TAU:.0%} positive)"
    )


def _plot_heatmaps(agg: pd.DataFrame, out_file: Path) -> None:
    row_order, col_order = _sort_heatmap(agg)

    gray = agg[agg["tissue_type"] == "gray"].copy()
    white = agg[agg["tissue_type"] == "white"].copy()

    pivot_gray = _pivot_for_heatmap(gray, row_order, col_order)
    pivot_white = _pivot_for_heatmap(white, row_order, col_order)
    annot_gray = _annot_fraction(gray, row_order, col_order)
    annot_white = _annot_fraction(white, row_order, col_order)

    fig = plt.figure(figsize=MICCAI_DOUBLE_COLUMN_FIGSIZE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.22)
    ax_gray = fig.add_subplot(gs[0, 0])
    ax_white = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    _draw_heatmap(
        ax_gray,
        pivot_gray,
        annot_gray,
        title=_robust_panel_title("gray"),
        show_yticklabels=True,
        show_cbar=False,
    )
    _draw_heatmap(
        ax_white,
        pivot_white,
        annot_white,
        title=_robust_panel_title("white"),
        show_yticklabels=False,
        show_cbar=False,
    )

    if ax_white.collections:
        mappable = ax_white.collections[0]
        cbar = fig.colorbar(mappable, cax=cax, format="%.1f")
        cbar.set_label("Prop. models benefiting", fontsize=MICCAI_MPL_PARAMS["axes.labelsize"])
        cbar.ax.tick_params(labelsize=MICCAI_MPL_PARAMS["xtick.labelsize"])
    else:
        cax.axis("off")

    # fig.subplots_adjust(left=0.09, right=0.97, bottom=0.16, top=0.76, wspace=0.22)
    fig.suptitle(TITLE_MAIN, y=0.99, fontsize=MICCAI_MPL_PARAMS["figure.titlesize"] + 1, fontweight="bold")
    
    # fig.tight_layout(rect=[0, 0, 1, 0.9])
    
    # Manually adjust margins to ensure content fits within figsize without bbox_inches='tight'
    # Adjust top to leave space for title (previously 0.76 might be too low)
    # Adjust bottom to leave space for rotated x-tick labels
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.25, top=0.80, wspace=0.3)
    
    # Save with specific DPI and no bbox_inches to enforce exact figsize
    fig.savefig(out_file, dpi=MICCAI_MPL_PARAMS["savefig.dpi"], bbox_inches=None)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point + CLI
# ---------------------------------------------------------------------------


def plot_feature_benefit_heatmap_robust_by_tissue(
    parquet_path: str,
    out_dir: str = "exp_outputs/summary/plots/features",
    merge_gender: bool = True,
) -> Path:
    apply_miccai_style()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = _load_scope(parquet_path)
    fold_df = _extract_fold_scores(df)
    delta_df = _build_delta_df(fold_df)

    per_model = _per_model_robust_stats(delta_df)
    if per_model.empty:
        raise RuntimeError("No per-model statistics computed")

    agg = _collapse_across_models(per_model)
    if merge_gender:
        agg = _merge_gender_rows(agg)

    model_count_range = agg["n_models_total"].agg(["min", "max"])
    if model_count_range["max"] - model_count_range["min"] > 2:
        warnings.warn(
            f"n_models_total varies: {model_count_range['min']}–{model_count_range['max']}. "
            "Some model/feature combinations may be missing.",
            stacklevel=2,
        )

    out_file = out_path / "feature_benefit_heatmap_robust_by_tissue.pdf"
    _plot_heatmaps(agg, out_file)
    return out_file


def plot_feature_benefit_heatmap_all_tissues_folds(
    parquet_path: str,
    out_dir: str = "exp_outputs/summary/plots/features",
    merge_gender: bool = True,
) -> Path:
    return plot_feature_benefit_heatmap_robust_by_tissue(
        parquet_path=parquet_path,
        out_dir=out_dir,
        merge_gender=merge_gender,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Two-panel robustness heatmap by tissue type "
            "(gray matter left, white matter right)."
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
        help=f"Minimum fraction of positive deltas for robustness rule (default {TAU})",
    )
    parser.add_argument(
        "--no-merge-gender",
        action="store_true",
        help="Keep HCP–Gender and CamCAN–Gender as separate rows.",
    )
    args = parser.parse_args()

    import feature_benefit_heatmap_all_tissues_folds as _self

    _self.EPSILON = args.epsilon
    _self.TAU = args.tau

    out_file = plot_feature_benefit_heatmap_robust_by_tissue(
        parquet_path=args.input,
        out_dir=args.outdir,
        merge_gender=not args.no_merge_gender,
    )
    print("Saved feature benefit heatmap to", out_file)


if __name__ == "__main__":
    main()
