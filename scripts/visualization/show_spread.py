from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import (
    MODEL_DISPLAY_ORDER,
    add_score_raw_from_prefix,
    aggregate_run_scores,
    choose_spread_metric,
    clean_target,
    filter_combos,
    format_label,
    map_model_display_group,
    map_model_family,
    minmax_normalize_with_baseline,
)

from config import MICCAI_DOUBLE_COLUMN_FIGSIZE, apply_miccai_style

SPREAD_COMBOS = [
    ("hcp", "Gender", "binary_classification"),
    ("camcan", "Gender", "binary_classification"),
    ("camcan", "Age", "regression"),
    ("abide", "DX_GROUP", "binary_classification"),
]
SPREAD_FEATURES = {"md", "mk", "sh", "b0"}
SPREAD_TISSUES = {"white", "gray"}
SPREAD_TITLE = "Performance Spread Across Dataset-Task Conditions"

FAMILY_PALETTE = {
    "Linear": "#4C78A8",
    "RandomForest": "#59A14F",
    "medicalnet": "#E15759",
    "dinov2": "#F28E2B",
    "curia": "#B07AA1",
}


def _load_spread_scope(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["target_clean"] = df["target"].map(clean_target)
    df = filter_combos(df, SPREAD_COMBOS)

    df = df[df["primary_metric"].isin(SPREAD_FEATURES)]
    df = df[df["tissue_type"].isin(SPREAD_TISSUES)]
    df["model_family"] = df["model_name"].map(map_model_family)
    df = df[df["model_family"].notna()].copy()

    if df.empty:
        raise RuntimeError("No rows left after applying spread plot scope filters")
    return df


def _compute_normalized_scores(df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for (_, _, task), group in df.groupby(
        ["dataset", "target_clean", "prediction_task"], dropna=False
    ):
        fold_prefix, metric_label = choose_spread_metric(group, task)
        part = add_score_raw_from_prefix(group, fold_prefix)
        part = part[part["score_raw"].notna()].copy()
        if part.empty:
            continue
        part["metric_label"] = metric_label
        parts.append(part)

    if not parts:
        raise RuntimeError("No valid task metrics found to compute spread scores")

    score_df = pd.concat(parts, ignore_index=True)
    score_df = minmax_normalize_with_baseline(
        score_df,
        score_col="score_raw",
        group_cols=("dataset", "target_clean", "prediction_task"),
    )
    score_df = score_df[score_df["score_norm"].notna()].copy()

    if score_df.empty:
        raise RuntimeError("No normalized scores available for plotting")
    return score_df


def _ordered_dataset_task_labels(df: pd.DataFrame) -> list[str]:
    seen = set(df["dataset_task"].unique().tolist())
    ordered = []
    for dataset, target, _ in SPREAD_COMBOS:
        label = f"{dataset}::{target}"
        if label in seen:
            ordered.append(label)
    return ordered


def _plot_spread(run_df: pd.DataFrame, output_file: Path) -> None:
    run_df = run_df.copy()
    run_df["dataset_task"] = run_df.apply(
        lambda r: f"{r['dataset']}::{r['target_clean']}",
        axis=1,
    )
    order = _ordered_dataset_task_labels(run_df)
    if not order:
        raise RuntimeError("No dataset-task labels available for plotting")
    pretty_order = [format_label(label) for label in order]
    run_df["dataset_task_label"] = run_df["dataset_task"].map(format_label)

    base_w, base_h = MICCAI_DOUBLE_COLUMN_FIGSIZE
    fig_w = max(base_w, 1.35 * len(order) + 1.6)
    fig_h = max(base_h, 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    violin_start = len(ax.collections)
    sns.violinplot(
        data=run_df,
        x="dataset_task_label",
        y="score_norm_run",
        order=pretty_order,
        hue="model_display",
        hue_order=MODEL_DISPLAY_ORDER,
        palette=FAMILY_PALETTE,
        dodge=True,
        inner=None,
        cut=0,
        linewidth=0.85,
        saturation=0.9,
        ax=ax,
    )
    for violin in ax.collections[violin_start:]:
        violin.set_alpha(0.25)
        violin.set_linewidth(0.85)
        violin.set_edgecolor((0.2, 0.2, 0.2, 0.65))

    sns.stripplot(
        data=run_df,
        x="dataset_task_label",
        y="score_norm_run",
        order=pretty_order,
        hue="model_display",
        hue_order=MODEL_DISPLAY_ORDER,
        palette=FAMILY_PALETTE,
        dodge=True,
        jitter=0.16,
        alpha=0.82,
        size=3.2,
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
        ax=ax,
    )

    ordered_labels = [
        name for name in MODEL_DISPLAY_ORDER if name in run_df["model_display"].unique()
    ]
    legend_handles = [
        Patch(
            facecolor=FAMILY_PALETTE[name],
            edgecolor="#333333",
            linewidth=0.8,
            alpha=0.55,
        )
        for name in ordered_labels
    ]
    ax.legend(
        legend_handles,
        ordered_labels,
        title="Model family\n(dummy-baseline min-max)",
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )

    ax.set_title(SPREAD_TITLE)
    ax.set_xlabel("")
    ax.set_ylabel("Normalized score (0 = dummy, 1 = perfect)")
    ax.set_ylim(0.0, 1.0)
    ax.set_facecolor("#FBFBFD")
    ax.tick_params(axis="x", rotation=25)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_model_family_spread(
    parquet_path: str,
    out_dir: str = "exp_outputs/summary/plots/folds",
) -> Path:
    apply_miccai_style()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = _load_spread_scope(parquet_path)
    score_df = _compute_normalized_scores(df)
    run_df = aggregate_run_scores(score_df, score_col="score_norm")
    run_df["model_display"] = run_df["model_name"].map(map_model_display_group)
    run_df = run_df[run_df["model_display"].isin(MODEL_DISPLAY_ORDER)].copy()

    if run_df.empty:
        raise RuntimeError("No per-run scores available after run-level aggregation")

    _plot_spread(run_df, out_path / "model_family_spread.pdf")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot performance spread by dataset-task and model family"
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
    args = parser.parse_args()

    out_path = plot_model_family_spread(args.input, args.outdir)
    print("Saved spread plot to", out_path)


if __name__ == "__main__":
    main()
