from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import (
    MODEL_DISPLAY_ORDER,
    MODEL_FAMILY_ORDER,
    add_score_raw_from_prefix,
    aggregate_run_scores,
    choose_spread_metric,
    clean_target,
    filter_combos,
    format_label,
    make_dataset_task_label,
    map_model_display_group,
    map_model_family,
    normalize_score,
    ordered_dataset_task_labels_from_combos,
)

from config import MICCAI_DOUBLE_COLUMN_FIGSIZE, apply_miccai_style

DELTA_COMBOS = [
    ("hcp", "Gender", "binary_classification"),
    ("camcan", "Gender", "binary_classification"),
    ("camcan", "Age", "regression"),
    ("abide", "DX_GROUP", "binary_classification"),
]
DELTA_FEATURES = {"md", "mk", "sh", "b0"}
DELTA_TISSUES = {"white", "gray"}

PLOT_TITLE = "Deep features vs linear baseline (score by dataset-task)"
PLOT_SUBTITLE = "Diamond = median per model group · Black segment = median(Linear) within each dataset-task"

FAMILY_COLORS = {
    "Linear": "#4C78A8",
    "RandomForest": "#59A14F",
    "medicalnet": "#E15759",
    "dinov2": "#F28E2B",
    "curia": "#B07AA1",
}
FAMILY_LABELS = {
    "Linear": "Linear",
    "RandomForest": "Random forest",
    "medicalnet": "medicalnet",
    "dinov2": "dinov2",
    "curia": "curia",
}


def _load_scope(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["target_clean"] = df["target"].map(clean_target)
    df = filter_combos(df, DELTA_COMBOS)
    df = df[df["primary_metric"].isin(DELTA_FEATURES)]
    df = df[df["tissue_type"].isin(DELTA_TISSUES)]
    df["model_family"] = df["model_name"].map(map_model_family)
    df = df[df["model_family"].notna()].copy()
    if df.empty:
        raise RuntimeError("No rows left after applying delta-to-linear scope filters")
    return df


def _compute_normalized_run_scores(df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for (_, _, task), group in df.groupby(
        ["dataset", "target_clean", "prediction_task"], dropna=False
    ):
        fold_prefix, _metric_label = choose_spread_metric(group, task)
        part = add_score_raw_from_prefix(group, fold_prefix)
        part = part[part["score_raw"].notna()].copy()
        if not part.empty:
            parts.append(part)
    if not parts:
        raise RuntimeError("No valid task metrics found for delta-to-linear plot")

    score_df = pd.concat(parts, ignore_index=True)
    score_df["score_norm"] = score_df.apply(normalize_score, axis=1)
    score_df = score_df[score_df["score_norm"].notna()].copy()
    if score_df.empty:
        raise RuntimeError("No normalized scores available for delta-to-linear plot")

    run_df = aggregate_run_scores(score_df, score_col="score_norm")
    run_df["dataset_task"] = run_df.apply(
        lambda r: make_dataset_task_label(r["dataset"], r["target_clean"]),
        axis=1,
    )
    run_df = run_df[run_df["model_family"].isin(MODEL_FAMILY_ORDER)].copy()
    run_df["model_display"] = run_df["model_name"].map(map_model_display_group)
    run_df = run_df[run_df["model_display"].isin(MODEL_DISPLAY_ORDER)].copy()
    if run_df.empty:
        raise RuntimeError("No per-run scores available after run-level aggregation")
    return run_df


def _attach_linear_reference(run_df: pd.DataFrame) -> pd.DataFrame:
    linear_ref = (
        run_df[run_df["model_family"] == "Linear"]
        .groupby("dataset_task", dropna=False)["score_norm_run"]
        .median()
    )
    missing = [
        label
        for label in run_df["dataset_task"].unique().tolist()
        if label not in linear_ref.index
    ]
    if missing:
        missing_fmt = ", ".join(sorted(missing))
        warnings.warn(
            f"Dropping dataset-task without Linear rows: {missing_fmt}",
            stacklevel=2,
        )

    kept_labels = set(linear_ref.index.tolist())
    run_df = run_df[run_df["dataset_task"].isin(kept_labels)].copy()
    if run_df.empty:
        raise RuntimeError(
            "No dataset-task left after dropping groups without Linear runs"
        )

    run_df["linear_ref"] = run_df["dataset_task"].map(linear_ref)
    return run_df


def _plot_delta_to_linear(run_df: pd.DataFrame, output_file: Path) -> None:
    order = ordered_dataset_task_labels_from_combos(
        run_df["dataset_task"].unique().tolist(), DELTA_COMBOS
    )
    if not order:
        raise RuntimeError("No dataset-task labels available for plotting")

    GROUP_SPACING = 1.6
    task_to_x = {label: idx * GROUP_SPACING for idx, label in enumerate(order)}
    pretty_labels = [format_label(label) for label in order]
    family_offsets = {
        "Linear": -0.40,
        "RandomForest": -0.20,
        "medicalnet": 0.0,
        "dinov2": 0.20,
        "curia": 0.40,
    }

    base_w, base_h = MICCAI_DOUBLE_COLUMN_FIGSIZE
    fig_w = max(base_w, 1.55 * len(order) * GROUP_SPACING + 0.6)
    fig_h = max(base_h, 3.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    rng = np.random.default_rng(7)
    for family in MODEL_DISPLAY_ORDER:
        sub = run_df[run_df["model_display"] == family]
        if sub.empty:
            continue
        x_base = sub["dataset_task"].map(task_to_x).astype(float).to_numpy()
        jitter = rng.normal(loc=0.0, scale=0.04, size=sub.shape[0])
        x = x_base + family_offsets[family] + jitter
        alpha = 0.25 if family == "Linear" else 0.8
        ax.scatter(
            x,
            sub["score_norm_run"].to_numpy(dtype=float),
            s=22,
            c=FAMILY_COLORS[family],
            alpha=alpha,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    for family in MODEL_DISPLAY_ORDER:
        sub = run_df[run_df["model_display"] == family]
        if sub.empty:
            continue
        color = FAMILY_COLORS[family]
        for task_label, x_center in task_to_x.items():
            vals = (
                sub.loc[sub["dataset_task"] == task_label, "score_norm_run"]
                .dropna()
                .to_numpy(dtype=float)
            )
            if len(vals) == 0:
                continue
            x_pos = x_center + family_offsets[family]
            if len(vals) >= 3:
                parts = ax.violinplot(
                    vals,
                    positions=[x_pos],
                    widths=0.22,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_edgecolor("white")
                    pc.set_alpha(0.55)
                    pc.set_linewidth(0.5)
                    pc.set_zorder(4)
            median_val = float(np.median(vals))
            ax.scatter(
                [x_pos],
                [median_val],
                marker="D",
                s=16,
                color=color,
                edgecolors="black",
                linewidths=0.35,
                zorder=5,
            )

    for task_label, x_center in task_to_x.items():
        ref_vals = run_df.loc[run_df["dataset_task"] == task_label, "linear_ref"]
        if ref_vals.empty:
            continue
        y_ref = float(ref_vals.iloc[0])
        ax.hlines(
            y=y_ref,
            xmin=x_center - 0.53,
            xmax=x_center + 0.53,
            colors="#1f1f1f",
            linewidth=1.15,
            alpha=0.9,
            zorder=2,
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.65, (len(order) - 1) * GROUP_SPACING + 0.65)
    ax.set_xticks([i * GROUP_SPACING for i in range(len(order))])
    ax.set_xticklabels(pretty_labels, rotation=24, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(r"Score (min-max, dummy$\,{=}\,0$ to perfect$\,{=}\,1$)")
    ax.set_title(PLOT_TITLE, pad=18)
    ax.text(
        0.01,
        1.02,
        PLOT_SUBTITLE,
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        color="#444444",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=FAMILY_COLORS[family],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=6,
            alpha=0.35 if family == "Linear" else 0.9,
            label=FAMILY_LABELS[family],
        )
        for family in MODEL_DISPLAY_ORDER
        if family in run_df["model_display"].unique()
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(legend_handles),
        frameon=False,
        borderaxespad=0.2,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_delta_to_linear(
    parquet_path: str,
    out_dir: str = "exp_outputs/summary/plots/folds",
) -> Path:
    apply_miccai_style()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = _load_scope(parquet_path)
    run_df = _compute_normalized_run_scores(df)
    run_df = _attach_linear_reference(run_df)

    _plot_delta_to_linear(run_df, out_path / "deep_vs_linear_delta.pdf")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot delta-to-linear effect sizes by dataset-task"
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

    out_path = plot_delta_to_linear(args.input, args.outdir)
    print("Saved delta-to-linear plot to", out_path)


if __name__ == "__main__":
    main()
