from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MICCAI_DOUBLE_COLUMN_FIGSIZE, apply_miccai_style
from utils import (
    choose_spread_metric,
    clean_target,
    filter_combos,
    fold_columns,
    map_model_family,
    normalize_score,
)

FEATURE_DELTA_COMBOS = [
    ("hcp", "Gender", "binary_classification"),
    ("camcan", "Gender", "binary_classification"),
    ("camcan", "Age", "regression"),
    ("abide", "DX_GROUP", "binary_classification"),
]
FEATURE_DELTA_TISSUES = {"white", "gray"}
FEATURE_PREFERRED_ORDER = ["md", "mk", "sh"]
FEATURE_EXCLUDE = {"rtop"}

PLOT_TITLE = "Feature benefit vs b0 by model (paired Δ normalized score)"
PLOT_SUBTITLE = "Each point: one dataset×task×tissue×fold. Δ = score(feature) − score(b0)."

MODEL_DISPLAY_ORDER = ["Linear", "RandomForest", "Deep Learning"]

FAMILY_COLORS = {
    "Linear": "#4C78A8",
    "RandomForest": "#59A14F",
    "Deep Learning": "#E15759",
}

FAMILY_LABELS = {
    "Linear": "Linear",
    "RandomForest": "Random forest",
    "Deep Learning": "Deep learning",
}

# Individual-model view
INDIVIDUAL_MODEL_ORDER = [
    "linear", "pca_linear", "lasso", "svm", "pca_svm",
    "forest", "pca_forest",
    "medicalnet", "dinov2", "curia",
]

INDIVIDUAL_MODEL_COLORS = {
    "linear":    "#4C78A8",
    "pca_linear": "#6EA6D0",
    "lasso":     "#9DC6E8",
    "svm":       "#2C5F8A",
    "pca_svm":   "#1A3D5C",
    "forest":    "#59A14F",
    "pca_forest": "#8CC97E",
    "medicalnet": "#E15759",
    "dinov2":    "#F28E2B",
    "curia":     "#B07AA1",
}

INDIVIDUAL_MODEL_LABELS = {
    "linear":    "Linear",
    "pca_linear": "PCA+Linear",
    "lasso":     "Lasso",
    "svm":       "SVM",
    "pca_svm":   "PCA+SVM",
    "forest":    "Random forest",
    "pca_forest": "PCA+RF",
    "medicalnet": "MedicalNet",
    "dinov2":    "DINOv2",
    "curia":     "Curia",
}


def _map_model_group(model_name: str) -> str | None:
    """Map a model name to one of three display groups."""
    family = map_model_family(model_name)
    if family is None:
        return None
    if family == "DeepEmbedding+LinearHead":
        return "Deep Learning"
    return family  # "Linear" or "RandomForest"


def _load_scope(parquet_path: str, group_by_family: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["target_clean"] = df["target"].map(clean_target)
    df = filter_combos(df, FEATURE_DELTA_COMBOS)
    df = df[df["tissue_type"].isin(FEATURE_DELTA_TISSUES)].copy()
    df["model_family"] = df["model_name"].map(map_model_family)
    df = df[df["model_family"].notna()].copy()
    if group_by_family:
        df["model_plot"] = df["model_name"].map(_map_model_group)
        df = df[df["model_plot"].isin(MODEL_DISPLAY_ORDER)].copy()
    else:
        df["model_plot"] = df["model_name"].str.strip().str.lower()
        df = df[df["model_plot"].isin(INDIVIDUAL_MODEL_ORDER)].copy()

    if df.empty:
        raise RuntimeError("No rows left after applying feature-vs-b0 scope filters")
    return df


def _extract_fold_level_scores(df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    grouping = ["dataset", "target_clean", "prediction_task"]

    for _, group in df.groupby(grouping, dropna=False):
        task = str(group["prediction_task"].iloc[0])
        fold_prefix, metric_label = choose_spread_metric(group, task)
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
                    "model_plot",
                    "primary_metric",
                ]
            ].copy()
            part["fold_index"] = fold_idx
            part["score_raw"] = pd.to_numeric(group[col], errors="coerce")
            part["metric_label"] = metric_label
            parts.append(part)

    if not parts:
        raise RuntimeError("No fold-level rows available for feature-vs-b0 plotting")

    fold_df = pd.concat(parts, ignore_index=True)
    fold_df = fold_df[fold_df["score_raw"].notna()].copy()
    if fold_df.empty:
        raise RuntimeError("All extracted fold-level values are missing")

    fold_df["score_norm"] = fold_df.apply(normalize_score, axis=1)
    fold_df = fold_df[fold_df["score_norm"].notna()].copy()
    if fold_df.empty:
        raise RuntimeError("No normalized fold-level scores available")
    return fold_df


def _build_paired_delta_df(fold_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "dataset",
        "target_clean",
        "prediction_task",
        "tissue_type",
        "model_name",
        "model_plot",
        "fold_index",
        "primary_metric",
    ]
    dedup = (
        fold_df.groupby(key_cols, dropna=False, as_index=False)
        .agg(
            score_norm=("score_norm", "mean"),
            metric_label=("metric_label", "first"),
        )
        .copy()
    )

    duplicate_count = int(len(fold_df) - len(dedup))
    if duplicate_count > 0:
        warnings.warn(
            f"Collapsed {duplicate_count} duplicate fold rows by averaging score_norm "
            "(same dataset/target/task/tissue/model/fold/feature).",
            stacklevel=2,
        )

    base_cols = [
        "dataset",
        "target_clean",
        "prediction_task",
        "tissue_type",
        "model_name",
        "model_plot",
        "fold_index",
        "metric_label",
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
        raise RuntimeError("Feature 'b0' is missing after pivoting; cannot build paired deltas")

    before = len(wide)
    wide = wide[wide["b0"].notna()].copy()
    dropped_missing_b0 = before - len(wide)
    if dropped_missing_b0 > 0:
        warnings.warn(
            f"Dropped {dropped_missing_b0} cell rows without b0 (cannot form paired deltas).",
            stacklevel=2,
        )

    feature_cols = [c for c in wide.columns if c not in set(base_cols + ["b0"])]
    feature_cols = [c for c in feature_cols if c not in FEATURE_EXCLUDE]
    feature_cols = _ordered_features(feature_cols)
    if not feature_cols:
        raise RuntimeError("No non-b0 features available after pairing")

    delta_df = wide.melt(
        id_vars=base_cols + ["b0"],
        value_vars=feature_cols,
        var_name="feature",
        value_name="feature_score",
    )
    delta_df = delta_df[delta_df["feature_score"].notna()].copy()
    delta_df["delta"] = delta_df["feature_score"] - delta_df["b0"]
    delta_df["primary_metric"] = delta_df["feature"]
    delta_df["cell_id"] = delta_df[
        [
            "dataset",
            "target_clean",
            "prediction_task",
            "tissue_type",
            "model_name",
            "fold_index",
        ]
    ].astype(str).agg("|".join, axis=1)

    keep_cols = [
        "cell_id",
        "feature",
        "primary_metric",
        "model_name",
        "model_plot",
        "dataset",
        "target_clean",
        "prediction_task",
        "tissue_type",
        "fold_index",
        "delta",
        "metric_label",
    ]
    delta_df = delta_df[keep_cols].copy()

    if delta_df.empty:
        raise RuntimeError("No paired feature-vs-b0 deltas available for plotting")
    return delta_df


def _aggregate_over_folds(delta_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "target_clean",
        "prediction_task",
        "tissue_type",
        "model_name",
        "model_plot",
        "feature",
        "primary_metric",
        "metric_label",
    ]
    out = (
        delta_df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            delta=("delta", "mean"),
            n_folds=("delta", "size"),
        )
        .copy()
    )
    out["cell_id"] = out[
        ["dataset", "target_clean", "prediction_task", "tissue_type", "model_name"]
    ].astype(str).agg("|".join, axis=1)
    if out.empty:
        raise RuntimeError("No rows left after averaging deltas over folds")
    return out


def _ordered_features(features: list[str]) -> list[str]:
    seen = set(features)
    ordered = [f for f in FEATURE_PREFERRED_ORDER if f in seen]
    ordered.extend(sorted([f for f in features if f not in ordered]))
    return ordered








# (dataset, target_clean, prediction_task, row label); None = all data
ROW_DEFS = [
    (None, None, None, "All datasets & targets"),
    ("hcp", "Gender", "binary_classification", "HCP – Gender"),
    ("camcan", "Gender", "binary_classification", "CamCAN – Gender"),
    ("camcan", "Age", "regression", "CamCAN – Age"),
    ("abide", "DX_GROUP", "binary_classification", "ABIDE – DX Group"),
]


def _draw_delta_ax(
    ax: plt.Axes,
    row_df: pd.DataFrame,
    hue_order: list[str],
    feature_order: list[str],
    palette: dict,
    label_map: dict,
    group_by_family: bool,
    row_label: str,
    show_legend: bool,
    global_ylim: tuple[float, float],
) -> None:
    """Draw violin + strip for one row on an existing Axes."""
    if row_df.empty:
        ax.set_visible(False)
        return

    sns.violinplot(
        data=row_df,
        x="feature",
        y="delta",
        hue="model_plot",
        order=feature_order,
        hue_order=hue_order,
        palette=palette,
        dodge=True,
        inner=None,
        width=0.8,
        linewidth=0.8,
        cut=0,
        ax=ax,
    )
    for collection in ax.collections:
        collection.set_alpha(0.45)

    sns.stripplot(
        data=row_df,
        x="feature",
        y="delta",
        hue="model_plot",
        order=feature_order,
        hue_order=hue_order,
        palette=palette,
        dodge=True,
        jitter=0.06,
        alpha=0.55,
        size=3.5,
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
        ax=ax,
    )

    ax.axhline(0.0, color="#1f1f1f", linewidth=1.4, alpha=0.95, zorder=1)
    ax.set_ylim(global_ylim)
    ax.set_xlabel("Feature" if show_legend else "")
    ax.set_ylabel("Δ norm. score vs b0", fontsize=8)
    ax.set_title(row_label, pad=6, fontsize=9, loc="left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, top=True, right=True)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle: dict[str, object] = {}
        for handle, label in zip(handles, labels):
            if label in hue_order and label not in label_to_handle:
                label_to_handle[label] = handle
        legend_labels = [label_map.get(n, n) for n in hue_order if n in label_to_handle]
        legend_handles = [label_to_handle[n] for n in hue_order if n in label_to_handle]
        if group_by_family:
            ax.legend(
                legend_handles, legend_labels,
                title="Model", loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                ncol=len(legend_handles),
                frameon=False, borderaxespad=0.3, fontsize=8,
            )
        else:
            ax.legend(
                legend_handles, legend_labels,
                title="Model", loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                ncol=1, frameon=True, borderaxespad=0.3, fontsize=8,
            )
    else:
        ax.legend_.remove() if ax.legend_ else None


def _plot_deltas(delta_df: pd.DataFrame, out_file: Path, group_by_family: bool = True) -> None:
    feature_order = _ordered_features(sorted(delta_df["feature"].unique().tolist()))
    if group_by_family:
        display_order = MODEL_DISPLAY_ORDER
        palette = FAMILY_COLORS
        label_map = FAMILY_LABELS
    else:
        display_order = INDIVIDUAL_MODEL_ORDER
        palette = INDIVIDUAL_MODEL_COLORS
        label_map = INDIVIDUAL_MODEL_LABELS
    hue_order = [m for m in display_order if m in delta_df["model_plot"].unique().tolist()]

    if not feature_order:
        raise RuntimeError("No features available for plotting")
    if not hue_order:
        raise RuntimeError("No model groups available for plotting")

    # Global y limits across all rows
    data_min = float(np.nanmin(delta_df["delta"].to_numpy(dtype=float)))
    data_max = float(np.nanmax(delta_df["delta"].to_numpy(dtype=float)))
    pad = 0.05 * (data_max - data_min)
    global_ylim = (data_min - pad, data_max + pad)

    n_rows = len(ROW_DEFS)
    base_w, base_h = MICCAI_DOUBLE_COLUMN_FIGSIZE
    if group_by_family:
        fig_w = max(base_w, 1.15 * len(feature_order) + 2.4)
    else:
        fig_w = max(14.0, 2.0 * len(feature_order) * len(hue_order) / 5 + 4.0)
    row_h = 3.0
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, row_h * n_rows), squeeze=False)

    for i, (dataset, target, task, row_label) in enumerate(ROW_DEFS):
        if dataset is None:
            row_df = delta_df
        else:
            row_df = delta_df[
                (delta_df["dataset"] == dataset)
                & (delta_df["target_clean"] == target)
                & (delta_df["prediction_task"] == task)
            ].copy()

        _draw_delta_ax(
            ax=axes[i, 0],
            row_df=row_df,
            hue_order=hue_order,
            feature_order=feature_order,
            palette=palette,
            label_map=label_map,
            group_by_family=group_by_family,
            row_label=row_label,
            show_legend=(i == 0),
            global_ylim=global_ylim,
        )

    fig.suptitle(PLOT_TITLE, y=1.01, fontsize=10)

    if group_by_family:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 0.85, 1.0))
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_vs_b0(
    parquet_path: str,
    out_dir: str = "exp_outputs/summary/plots/folds",
    group_by_family: bool = True,
) -> Path:
    apply_miccai_style()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scope_df = _load_scope(parquet_path, group_by_family=group_by_family)
    fold_df = _extract_fold_level_scores(scope_df)
    delta_df = _build_paired_delta_df(fold_df)

    suffix = "family" if group_by_family else "individual"
    _plot_deltas(delta_df, out_path / f"feature_vs_b0_by_model_{suffix}_delta.pdf", group_by_family=group_by_family)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot feature benefit vs b0 by model using paired fold-level deltas"
    )
    parser.add_argument(
        "--input",
        default="exp_outputs/summary/comprehensive_results.parquet",
        help="Input parquet file",
    )
    parser.add_argument(
        "--outdir",
        default="exp_outputs/summary/plots/folds",
        help="Output directory",
    )
    parser.add_argument(
        "--individual-models",
        action="store_true",
        help="Show each model individually instead of grouping by family",
    )
    args = parser.parse_args()

    out_path = plot_feature_vs_b0(args.input, args.outdir, group_by_family=not args.individual_models)
    print("Saved feature-vs-b0 plot to", out_path)


if __name__ == "__main__":
    main()
