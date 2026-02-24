"""
combined_plot.py
----------------
Single publication-quality figure combining:
  - Left panel (2/3): model-family comparison vs linear baseline
  - Right panel (1/3): preprocessing sensitivity (range & IQR across feature × tissue)

Both panels share the same y-axis (normalized score, dummy=0 / perfect=1)
and the same dataset-task x-order.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
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

# ── Shared scope ──────────────────────────────────────────────────────────────
COMBOS = [
    ("hcp", "Gender", "binary_classification"),
    ("camcan", "Gender", "binary_classification"),
    ("camcan", "Age", "regression"),
    ("abide", "DX_GROUP", "binary_classification"),
]
FEATURES = {"md", "mk", "sh", "b0"}
TISSUES = {"white", "gray"}

# ── Shared visual constants ───────────────────────────────────────────────────
GROUP_SPACING_LEFT = 1.6  # wider: 5 model families per group
GROUP_SPACING_RIGHT = 1.2  # narrower: one bar stack per group

FAMILY_COLORS = {
    "Linear": "#4C78A8",
    "RandomForest": "#59A14F",
    "medicalnet": "#E15759",
    "dinov2": "#F28E2B",
    "curia": "#B07AA1",
}
FAMILY_LABELS = {
    "Linear": "Linear",
    "RandomForest": "Random\nForest",
    "medicalnet": "MedicalNet",
    "dinov2": "DINOv2",
    "curia": "Curia",
}
FAMILY_OFFSETS = {
    "Linear": -0.40,
    "RandomForest": -0.20,
    "medicalnet": 0.0,
    "dinov2": 0.20,
    "curia": 0.40,
}

RANGE_LINE_COLOR = "#333333"
IQR_COLOR = "#111111"
ANNOT_COLOR = "#555555"
DOT_COLOR = "#444444"

Y_LABEL = "Score (min-max,\ndummy = 0 to perfect = 1)"


# ── Data pipelines ────────────────────────────────────────────────────────────


def _base_df(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["target_clean"] = df["target"].map(clean_target)
    df = filter_combos(df, COMBOS)
    df = df[df["primary_metric"].isin(FEATURES)]
    df = df[df["tissue_type"].isin(TISSUES)]
    df["model_family"] = df["model_name"].map(map_model_family)
    df = df[df["model_family"].notna()].copy()
    if df.empty:
        raise RuntimeError("No rows left after scope filtering")
    return df


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (_, _, task), group in df.groupby(
        ["dataset", "target_clean", "prediction_task"], dropna=False
    ):
        fold_prefix, _ = choose_spread_metric(group, task)
        part = add_score_raw_from_prefix(group, fold_prefix)
        part = part[part["score_raw"].notna()].copy()
        if not part.empty:
            parts.append(part)
    if not parts:
        raise RuntimeError("No valid metrics found")
    score_df = pd.concat(parts, ignore_index=True)
    score_df["score_norm"] = score_df.apply(normalize_score, axis=1)
    score_df["dataset_task"] = score_df.apply(
        lambda r: make_dataset_task_label(r["dataset"], r["target_clean"]), axis=1
    )
    # return score_df[score_df["score_norm"].notna()].copy()
    return score_df[
        score_df["score_norm"].notna() & (score_df["score_norm"] != 0)
    ].copy()


def _load_left_data(parquet_path: str) -> pd.DataFrame:
    """Per-run normalized scores, with linear reference attached."""
    df = _base_df(parquet_path)
    score_df = _normalise(df)

    run_df = aggregate_run_scores(score_df, score_col="score_norm")
    run_df["dataset_task"] = run_df.apply(
        lambda r: make_dataset_task_label(r["dataset"], r["target_clean"]), axis=1
    )
    run_df = run_df[run_df["model_family"].isin(MODEL_FAMILY_ORDER)].copy()
    run_df["model_display"] = run_df["model_name"].map(map_model_display_group)
    run_df = run_df[run_df["model_display"].isin(MODEL_DISPLAY_ORDER)].copy()

    linear_ref = (
        run_df[run_df["model_family"] == "Linear"]
        .groupby("dataset_task", dropna=False)["score_norm_run"]
        .median()
    )
    missing = [l for l in run_df["dataset_task"].unique() if l not in linear_ref.index]
    if missing:
        warnings.warn(
            f"Dropping dataset-task without Linear rows: {', '.join(sorted(missing))}"
        )
    run_df = run_df[run_df["dataset_task"].isin(linear_ref.index)].copy()
    run_df["linear_ref"] = run_df["dataset_task"].map(linear_ref)
    return run_df


def _load_right_data(parquet_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prep-level median scores + per-dataset-task stats."""
    df = _base_df(parquet_path)
    score_df = _normalise(df)

    prep_df = (
        score_df.groupby(
            ["dataset_task", "primary_metric", "tissue_type"], dropna=False
        )["score_norm"]
        .median()
        .reset_index()
        .rename(columns={"score_norm": "prep_score"})
    )
    prep_df = prep_df[prep_df["prep_score"].notna()].copy()

    stats_rows = []
    for label, grp in prep_df.groupby("dataset_task", dropna=False):
        s = grp["prep_score"].dropna()
        mn, mx = float(s.min()), float(s.max())
        stats_rows.append(
            {
                "dataset_task": label,
                "prep_min": mn,
                "prep_max": mx,
                "prep_q1": float(s.quantile(0.25)),
                "prep_q3": float(s.quantile(0.75)),
                "prep_range": mx - mn,
                "n_preps": len(s),
            }
        )
    stats_df = pd.DataFrame(stats_rows)
    return prep_df, stats_df


# ── Panel drawers ─────────────────────────────────────────────────────────────


def _draw_left(ax: plt.Axes, run_df: pd.DataFrame, order: list[str]) -> None:
    """Delta-to-linear panel: violins + scatter per model family."""
    task_to_x = {label: idx * GROUP_SPACING_LEFT for idx, label in enumerate(order)}
    pretty_labels = [format_label(label) for label in order]

    rng = np.random.default_rng(7)

    # Scatter (individual runs)
    for family in MODEL_DISPLAY_ORDER:
        sub = run_df[run_df["model_display"] == family]
        if sub.empty:
            continue
        x_base = sub["dataset_task"].map(task_to_x).astype(float).to_numpy()
        jitter = rng.normal(0.0, 0.04, size=sub.shape[0])
        alpha = 0.25 if family == "Linear" else 0.8
        ax.scatter(
            x_base + FAMILY_OFFSETS[family] + jitter,
            sub["score_norm_run"].to_numpy(dtype=float),
            s=22,
            c=FAMILY_COLORS[family],
            alpha=alpha,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    # Violin + median diamond
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
            x_pos = x_center + FAMILY_OFFSETS[family]
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
            ax.scatter(
                [x_pos],
                [float(np.median(vals))],
                marker="D",
                s=16,
                color=color,
                edgecolors="black",
                linewidths=0.35,
                zorder=5,
            )

    # Linear reference hline
    for task_label, x_center in task_to_x.items():
        ref_vals = run_df.loc[run_df["dataset_task"] == task_label, "linear_ref"]
        if ref_vals.empty:
            continue
        ax.hlines(
            y=float(ref_vals.iloc[0]),
            xmin=x_center - 0.53,
            xmax=x_center + 0.53,
            colors="#1f1f1f",
            linewidth=1.15,
            alpha=0.9,
            zorder=2,
        )

    ax.set_xlim(-0.65, (len(order) - 1) * GROUP_SPACING_LEFT + 0.65)
    ax.set_xticks([i * GROUP_SPACING_LEFT for i in range(len(order))])
    ax.set_xticklabels(pretty_labels, rotation=24, ha="right")
    ax.set_title("Model families vs baseline")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=FAMILY_COLORS[f],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=6,
            alpha=0.35 if f == "Linear" else 0.9,
            label=FAMILY_LABELS[f],
        )
        for f in MODEL_DISPLAY_ORDER
        if f in run_df["model_display"].unique()
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=1,
        frameon=True,
        framealpha=0.85,
        edgecolor="none",
        borderaxespad=0.4,
    )


def _draw_right(
    ax: plt.Axes,
    prep_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    order: list[str],
) -> None:
    """Prep-sensitivity panel: range + IQR bars + jittered dots."""
    task_to_x = {label: idx * GROUP_SPACING_RIGHT for idx, label in enumerate(order)}
    pretty_labels = [format_label(label) for label in order]
    rng = np.random.default_rng(42)

    for task_label, x_center in task_to_x.items():
        row = stats_df[stats_df["dataset_task"] == task_label]
        if row.empty:
            continue
        row = row.iloc[0]

        # IQR bar
        ax.vlines(
            x_center,
            row["prep_q1"],
            row["prep_q3"],
            colors=IQR_COLOR,
            linewidth=5.0,
            alpha=0.30,
            zorder=3,
        )

        # Median marker
        med = prep_df.loc[prep_df["dataset_task"] == task_label, "prep_score"].median()
        ax.scatter(
            [x_center],
            [med],
            marker="D",
            s=18,
            color=IQR_COLOR,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
        )

    # Dots
    x_base = prep_df["dataset_task"].map(task_to_x).astype(float).to_numpy()
    jitter = rng.normal(0.0, 0.045, size=len(prep_df))
    ax.scatter(
        x_base + jitter,
        prep_df["prep_score"].to_numpy(dtype=float),
        s=24,
        c=DOT_COLOR,
        alpha=0.55,
        edgecolors="white",
        linewidths=0.3,
        zorder=5,
    )

    ax.set_xlim(-0.55, (len(order) - 1) * GROUP_SPACING_RIGHT + 0.55)
    ax.set_xticks([i * GROUP_SPACING_RIGHT for i in range(len(order))])
    ax.set_xticklabels(pretty_labels, rotation=24, ha="right")
    ax.set_title("Preprocessing sensitivity")

    iqr_h = Line2D([0], [0], color=IQR_COLOR, linewidth=4, alpha=0.30, label="IQR")
    med_h = Line2D(
        [0],
        [0],
        marker="D",
        linestyle="",
        markerfacecolor=IQR_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.4,
        markersize=5,
        alpha=0.85,
        label="Median",
    )
    ax.legend(
        handles=[iqr_h, med_h],
        loc="upper right",
        ncol=1,
        frameon=True,
        framealpha=0.85,
        edgecolor="none",
        borderaxespad=0.4,
    )


# ── Combined figure ───────────────────────────────────────────────────────────


def plot_combined(
    parquet_path: str,
    out_dir: str = "exp_outputs/summary/plots/folds",
) -> Path:
    apply_miccai_style()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    run_df = _load_left_data(parquet_path)
    prep_df, stats_df = _load_right_data(parquet_path)

    # Determine shared x-order from the union of present dataset-tasks
    all_labels = set(run_df["dataset_task"].unique()) | set(
        prep_df["dataset_task"].unique()
    )
    order = ordered_dataset_task_labels_from_combos(list(all_labels), COMBOS)
    if not order:
        raise RuntimeError("No dataset-task labels for combined plot")

    # Figure: two panels, 2:1 width ratio, shared y-axis
    fig = plt.figure(figsize=MICCAI_DOUBLE_COLUMN_FIGSIZE)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.08)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1], sharey=ax_left)

    _draw_left(ax_left, run_df, order)
    _draw_right(ax_right, prep_df, stats_df, order)

    # Shared y-axis: label + ticks on left; ticks visible on right, no label
    ax_left.set_ylim(0.0, 1.08)
    ax_left.set_ylabel(Y_LABEL)
    ax_right.yaxis.set_tick_params(
        which="both", labelleft=True, labelright=False, length=0, pad=2
    )
    ax_right.yaxis.tick_left()
    ax_right.set_ylabel("")

    for ax in (ax_left, ax_right):
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.grid(axis="x", visible=False)
    sns.despine(ax=ax_left, top=True, right=True)
    sns.despine(ax=ax_right, top=True, right=True, left=False)

    fig.savefig(out_path / "combined_model_vs_prep.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined model-family vs preprocessing-sensitivity figure"
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
    out_path = plot_combined(args.input, args.outdir)
    print("Saved combined plot to", out_path)


if __name__ == "__main__":
    main()
