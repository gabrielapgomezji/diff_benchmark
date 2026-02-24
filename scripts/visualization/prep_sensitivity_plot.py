"""
prep_sensitivity_plot.py
------------------------
Publication-quality figure showing how much normalized performance varies
across preprocessing choices (feature × tissue_type) within each dataset-task.

Visual argument:
  "Preprocessing choice can move a model's score by 0.10–0.30+, even on
   identical data.  Benchmark comparisons must account for this."
"""
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

from config import apply_miccai_style
from utils import (
    add_score_raw_from_prefix,
    choose_spread_metric,
    clean_target,
    filter_combos,
    format_label,
    make_dataset_task_label,
    map_model_family,
    normalize_score,
    ordered_dataset_task_labels_from_combos,
)

# ── Scope ────────────────────────────────────────────────────────────────────
PREP_COMBOS = [
    ("hcp", "Gender", "binary_classification"),
    ("camcan", "Gender", "binary_classification"),
    ("camcan", "Age", "regression"),
    ("abide", "DX_GROUP", "binary_classification"),
]
PREP_FEATURES = {"md", "mk", "sh", "b0"}
PREP_TISSUES = {"white", "gray"}

# ── Visual constants ──────────────────────────────────────────────────────────
RANGE_LINE_COLOR  = "#333333"
IQR_COLOR         = "#111111"
ANNOT_COLOR       = "#444444"
DOT_COLOR         = "#444444"

PLOT_TITLE    = "Sensitivity to preprocessing across dataset-task conditions"
PLOT_SUBTITLE = "Each dot = one preprocessing condition"
Y_LABEL       = "Normalized score (0 = dummy, 1 = perfect)"


# ── Data loading & normalisation ─────────────────────────────────────────────

def _load_scope(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["target_clean"] = df["target"].map(clean_target)
    df = filter_combos(df, PREP_COMBOS)
    df = df[df["primary_metric"].isin(PREP_FEATURES)]
    df = df[df["tissue_type"].isin(PREP_TISSUES)]
    df["model_family"] = df["model_name"].map(map_model_family)
    # exclude dummy models; keep all non-dummy model families
    df = df[df["model_family"].notna()].copy()
    if df.empty:
        raise RuntimeError("No rows left after applying prep-sensitivity scope filters")
    return df


def _compute_prep_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per (dataset_task, primary_metric, tissue_type) with
    `prep_score` = median of score_norm_run across all (model, run, fold) combos.
    """
    parts: list[pd.DataFrame] = []
    for (ds, tgt, task), group in df.groupby(
        ["dataset", "target_clean", "prediction_task"], dropna=False
    ):
        fold_prefix, _label = choose_spread_metric(group, task)
        part = add_score_raw_from_prefix(group, fold_prefix)
        part = part[part["score_raw"].notna()].copy()
        if not part.empty:
            parts.append(part)
    if not parts:
        raise RuntimeError("No valid metrics for prep-sensitivity plot")

    score_df = pd.concat(parts, ignore_index=True)
    score_df["score_norm"] = score_df.apply(normalize_score, axis=1)
    score_df = score_df[score_df["score_norm"].notna()].copy()

    score_df["dataset_task"] = score_df.apply(
        lambda r: make_dataset_task_label(r["dataset"], r["target_clean"]), axis=1
    )

    # Aggregate: median across all models/runs/folds for each prep_id
    prep_df = (
        score_df.groupby(
            ["dataset_task", "primary_metric", "tissue_type"], dropna=False
        )["score_norm"]
        .median()
        .reset_index()
        .rename(columns={"score_norm": "prep_score"})
    )
    prep_df = prep_df[prep_df["prep_score"].notna()].copy()
    if prep_df.empty:
        raise RuntimeError("No prep-level scores computed")
    return prep_df


# ── Statistics per dataset-task ───────────────────────────────────────────────

def _dataset_task_stats(prep_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_label, grp in prep_df.groupby("dataset_task", dropna=False):
        s = grp["prep_score"].dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        mn, mx = float(s.min()), float(s.max())
        rows.append(
            {
                "dataset_task": task_label,
                "prep_min": mn,
                "prep_max": mx,
                "prep_q1": q1,
                "prep_q3": q3,
                "prep_range": mx - mn,
                "n_preps": len(s),
            }
        )
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_prep_sensitivity(
    prep_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_file: Path,
) -> None:
    GROUP_SPACING = 1.6
    order = ordered_dataset_task_labels_from_combos(
        prep_df["dataset_task"].unique().tolist(), PREP_COMBOS
    )
    if not order:
        raise RuntimeError("No dataset-task labels available for plotting")

    task_to_x: dict[str, float] = {
        label: idx * GROUP_SPACING for idx, label in enumerate(order)
    }
    pretty_labels = [format_label(label) for label in order]

    fig, ax = plt.subplots()

    rng = np.random.default_rng(42)

    # ── Per-dataset-task visual elements ─────────────────────────────────────
    for task_label, x_center in task_to_x.items():
        row = stats_df[stats_df["dataset_task"] == task_label]
        if row.empty:
            continue
        row = row.iloc[0]

        # Full range: thin dark line
        ax.vlines(
            x_center,
            row["prep_min"],
            row["prep_max"],
            colors=RANGE_LINE_COLOR,
            linewidth=1.0,
            alpha=0.55,
            zorder=2,
        )
        # Capped ends (whisker caps)
        for y_cap in (row["prep_min"], row["prep_max"]):
            ax.hlines(
                y_cap,
                x_center - 0.10,
                x_center + 0.10,
                colors=RANGE_LINE_COLOR,
                linewidth=1.0,
                alpha=0.55,
                zorder=2,
            )

        # IQR: thick bar
        ax.vlines(
            x_center,
            row["prep_q1"],
            row["prep_q3"],
            colors=IQR_COLOR,
            linewidth=5.0,
            alpha=0.30,
            zorder=3,
        )

        # Annotation: range value + n
        ax.text(
            x_center,
            min(row["prep_max"] + 0.045, 0.97),
            f"Δ{row['prep_range']:.2f}  (n={int(row['n_preps'])})",
            ha="center",
            va="bottom",

            color=ANNOT_COLOR,
            zorder=6,
        )

    # ── Individual prep dots (one per feature × tissue condition) ───────────
    x_base = prep_df["dataset_task"].map(task_to_x).astype(float).to_numpy()
    jitter = rng.normal(loc=0.0, scale=0.055, size=len(prep_df))
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

    # ── Axes decoration ───────────────────────────────────────────────────────
    ax.set_ylim(0.0, 1.08)
    ax.set_xlim(-0.65, (len(order) - 1) * GROUP_SPACING + 0.65)
    ax.set_xticks([i * GROUP_SPACING for i in range(len(order))])
    ax.set_xticklabels(pretty_labels, rotation=24, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(Y_LABEL)
    ax.set_title(PLOT_TITLE, pad=18)
    ax.text(
        0.01,
        1.02,
        PLOT_SUBTITLE,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=ANNOT_COLOR,
    )
    ax.axhline(y=0.0, color="#999999", linewidth=0.7, linestyle="--", zorder=1)

    # ── Legend ────────────────────────────────────────────────────────────────
    range_handle = Line2D(
        [0], [0],
        color=RANGE_LINE_COLOR, linewidth=1.0, alpha=0.55, label="Full range",
    )
    iqr_handle = Line2D(
        [0], [0],
        color=IQR_COLOR, linewidth=5.0, alpha=0.30,
        label="IQR",
    )
    dot_handle = Line2D(
        [0], [0],
        marker="o", linestyle="",
        markerfacecolor=DOT_COLOR, markeredgecolor="white",
        markeredgewidth=0.3, markersize=5, alpha=0.55,
        label="Preprocessing condition",
    )
    ax.legend(
        handles=[dot_handle, iqr_handle, range_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        borderaxespad=0.2,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout(rect=(0.02, 0.0, 1.0, 1.0))
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


# ── Public entry point ────────────────────────────────────────────────────────

def plot_prep_sensitivity(
    parquet_path: str,
    out_dir: str = "exp_outputs/summary/plots/folds",
) -> Path:
    apply_miccai_style()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = _load_scope(parquet_path)
    prep_df = _compute_prep_scores(df)
    stats_df = _dataset_task_stats(prep_df)

    output_file = out_path / "prep_sensitivity.pdf"
    _plot_prep_sensitivity(prep_df, stats_df, output_file)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot preprocessing sensitivity across dataset-task conditions"
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

    out_path = plot_prep_sensitivity(args.input, args.outdir)
    print("Saved prep-sensitivity plot to", out_path)


if __name__ == "__main__":
    main()
