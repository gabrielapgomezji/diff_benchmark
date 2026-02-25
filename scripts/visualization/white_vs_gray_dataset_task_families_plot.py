from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import MICCAI_DOUBLE_COLUMN_FIGSIZE, apply_miccai_style
from utils import (
    choose_spread_metric,
    clean_target,
    filter_combos,
    fold_columns,
    format_label,
    is_dummy_model,
    map_model_family,
    select_best_runs,
)

COMBOS = [
    ("hcp", "Gender", "binary_classification"),
    ("hcp", "Age", "regression"),
    ("camcan", "Age", "regression"),
    ("camcan", "Gender", "binary_classification"),
    ("abide", "DX_GROUP", "binary_classification"),
]

MODEL_FAMILY_MAP = {
    "Linear": "Linear",
    "RandomForest": "Random Forest",
    "DeepEmbedding+LinearHead": "Deep",
}
MODEL_FAMILY_ORDER = ["Linear", "Random Forest", "Deep"]

PLOT_TITLE = "White vs Gray Matter Tissue Effect by Dataset-Target"
Y_AXIS_LABEL = "\u0394 Normalized score (white \u2212 gray)"
TOP_REGION_LABEL = "White matter wins"
BOTTOM_REGION_LABEL = "Gray matter wins"
TOP_REGION_COLOR = "#D6E4F1"
BOTTOM_REGION_COLOR = "#FAEFDB"
Y_LIM = 1.0

FAMILY_COLORS = {
    "Linear": "#D6E3F3",
    "Random Forest": "#D8ECD0",
    "Deep": "#F6D6C8",
}
FAMILY_POINT_COLORS = {
    "Linear": "#4C78A8",
    "Random Forest": "#59A14F",
    "Deep": "#E07B39",
}


def _combo_label(dataset: str, target: str) -> str:
    return f"{dataset}::{str(target).lower()}"


def _load_scope(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["target_clean"] = df["target"].map(clean_target)
    df = filter_combos(df, COMBOS)
    if df.empty:
        raise RuntimeError("No rows left after filtering requested dataset-target-task combos")
    return df


def _select_best_tissue_rows(
    feature_df: pd.DataFrame,
    higher_is_better: bool,
) -> tuple[pd.Series, pd.Series] | None:
    white_rows = feature_df[feature_df["tissue_type"] == "white"]
    gray_rows = feature_df[feature_df["tissue_type"] == "gray"]
    if white_rows.empty or gray_rows.empty:
        return None

    if higher_is_better:
        best_white = white_rows.loc[white_rows["_fold_mean"].idxmax()]
        best_gray = gray_rows.loc[gray_rows["_fold_mean"].idxmax()]
    else:
        best_white = white_rows.loc[white_rows["_fold_mean"].idxmin()]
        best_gray = gray_rows.loc[gray_rows["_fold_mean"].idxmin()]
    return best_white, best_gray


def _normalize_fold_val(val: float, prediction_task: str) -> float:
    if prediction_task == "binary_classification":
        return (val - 0.5) / 0.5
    return float(val)


def _collect_pairwise_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, target, task), group in df.groupby(
        ["dataset", "target_clean", "prediction_task"], dropna=False
    ):
        try:
            fold_prefix, _metric_label = choose_spread_metric(group, task)
        except (RuntimeError, ValueError):
            continue

        best = select_best_runs(group, fold_prefix, higher_is_better=True)
        if best.empty or "_fold_mean" not in best.columns:
            continue

        best = best[~best["model_name"].apply(is_dummy_model)].copy()
        if best.empty:
            continue

        best["family"] = best["model_name"].map(map_model_family).map(MODEL_FAMILY_MAP)
        best = best[best["family"].isin(MODEL_FAMILY_ORDER)].copy()
        if best.empty:
            continue

        fold_cols = fold_columns(best, fold_prefix)
        if not fold_cols:
            continue

        for family, family_df in best.groupby("family", dropna=False):
            for feature, feature_df in family_df.groupby("primary_metric", dropna=False):
                best_tissues = _select_best_tissue_rows(feature_df, higher_is_better=True)
                if best_tissues is None:
                    continue
                best_white, best_gray = best_tissues

                white_vals = np.asarray(best_white[fold_cols].values, dtype=float)
                gray_vals = np.asarray(best_gray[fold_cols].values, dtype=float)
                valid = np.isfinite(white_vals) & np.isfinite(gray_vals)
                white_vals = white_vals[valid]
                gray_vals = gray_vals[valid]
                if white_vals.size == 0:
                    continue

                norm_white = np.array([_normalize_fold_val(v, task) for v in white_vals])
                norm_gray = np.array([_normalize_fold_val(v, task) for v in gray_vals])
                diffs = norm_white - norm_gray

                for fold_idx, diff in enumerate(diffs):
                    rows.append(
                        {
                            "dataset": dataset,
                            "target": target,
                            "task": task,
                            "feature": str(feature),
                            "family": family,
                            "fold": fold_idx,
                            "normalized_diff": float(diff),
                            "combo_label": _combo_label(dataset, target),
                        }
                    )

    return pd.DataFrame(rows)


def _center_dataset_effects(diffs: pd.DataFrame) -> pd.DataFrame:
    """Remove feature baseline offsets using all fold rows (no fold aggregation)."""
    feature_means = (
        diffs.groupby("feature")["normalized_diff"].mean().rename("feature_mean").reset_index()
    )
    out = diffs.merge(feature_means, on="feature", how="left")
    out = out.copy()
    out["normalized_diff"] = out["normalized_diff"] - out["feature_mean"]
    return out.drop(columns=["feature_mean"])


def _ordered_combo_labels(plot_df: pd.DataFrame) -> list[str]:
    present = set(plot_df["combo_label"].dropna().astype(str).tolist())
    ordered = []
    for dataset, target, _task in COMBOS:
        label = _combo_label(dataset, target)
        if label in present:
            ordered.append(label)
    return ordered


def _fold_level_ttest_pvalue(values: np.ndarray) -> float:
    from scipy.stats import ttest_1samp

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    result = ttest_1samp(values, popmean=0.0, nan_policy="omit")
    if np.isnan(result.pvalue):
        return float("nan")
    return float(result.pvalue)


def _compute_family_pvalues(plot_df: pd.DataFrame, order: list[str]) -> dict[tuple[str, str], float]:
    pvals: dict[tuple[str, str], float] = {}
    for combo_label in order:
        combo_df = plot_df[plot_df["combo_label"] == combo_label]
        for family in MODEL_FAMILY_ORDER:
            subset = combo_df[combo_df["family"] == family]
            if subset.empty:
                pvals[(combo_label, family)] = float("nan")
                continue
            pvals[(combo_label, family)] = _fold_level_ttest_pvalue(
                subset["normalized_diff"].to_numpy(dtype=float)
            )
    return pvals


def _format_pvalue(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p=n/a"
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def _plot_family_comparison(plot_df: pd.DataFrame, output_file: Path) -> None:
    if plot_df.empty:
        raise RuntimeError("No white-vs-gray effects available for plotting")

    order = _ordered_combo_labels(plot_df)
    if not order:
        raise RuntimeError("No requested dataset-target combinations available for plotting")

    base_w, base_h = MICCAI_DOUBLE_COLUMN_FIGSIZE
    fig_w = max(base_w, 1.55 * len(order) + 0.6)
    fig_h = max(base_h, 3.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    box_width = 0.72
    sns.boxplot(
        data=plot_df,
        x="combo_label",
        y="normalized_diff",
        hue="family",
        order=order,
        hue_order=MODEL_FAMILY_ORDER,
        palette=FAMILY_COLORS,
        width=box_width,
        linewidth=1.0,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="combo_label",
        y="normalized_diff",
        hue="family",
        order=order,
        hue_order=MODEL_FAMILY_ORDER,
        palette=FAMILY_POINT_COLORS,
        dodge=True,
        size=3.2,
        alpha=0.45,
        jitter=0.16,
        ax=ax,
    )

    if ax.legend_ is not None:
        ax.legend_.remove()
    legend_handles = [
        Patch(facecolor=FAMILY_COLORS[family], edgecolor="#3a3a3a", label=f"{family} models")
        for family in MODEL_FAMILY_ORDER
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    pvals = _compute_family_pvalues(plot_df, order)
    group_max = plot_df.groupby(["combo_label", "family"])["normalized_diff"].max()
    y_pad = 0.06 * Y_LIM
    y_cap = 0.84 * Y_LIM

    n_families = len(MODEL_FAMILY_ORDER)
    offsets = np.linspace(
        -box_width / 2.0 + box_width / (2.0 * n_families),
        box_width / 2.0 - box_width / (2.0 * n_families),
        n_families,
    )
    family_offsets = dict(zip(MODEL_FAMILY_ORDER, offsets, strict=False))

    for idx, combo_label in enumerate(order):
        for family in MODEL_FAMILY_ORDER:
            subset = plot_df[
                (plot_df["combo_label"] == combo_label) & (plot_df["family"] == family)
            ]
            if subset.empty:
                continue
            box_top = group_max.get((combo_label, family), np.nan)
            y_text = y_cap if not np.isfinite(box_top) else min(float(box_top) + y_pad, y_cap)
            ax.text(
                idx + family_offsets[family],
                y_text,
                _format_pvalue(pvals[(combo_label, family)]),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#303030",
            )

    ax.axhspan(0.0, Y_LIM, color=TOP_REGION_COLOR, alpha=0.32, zorder=0)
    ax.axhspan(-Y_LIM, 0.0, color=BOTTOM_REGION_COLOR, alpha=0.32, zorder=0)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_ylim(-Y_LIM, Y_LIM)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("")
    ax.set_ylabel(Y_AXIS_LABEL)
    ax.set_title(PLOT_TITLE, pad=12)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([format_label(label) for label in order], rotation=18, ha="right")
    ax.text(
        0.5,
        Y_LIM * 0.94,
        TOP_REGION_LABEL,
        transform=ax.get_yaxis_transform(),
        ha="center",
        va="top",
        fontweight="bold",
        color="#606060",
    )
    ax.text(
        0.5,
        -Y_LIM * 0.94,
        BOTTOM_REGION_LABEL,
        transform=ax.get_yaxis_transform(),
        ha="center",
        va="bottom",
        fontweight="bold",
        color="#606060",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def generate_white_vs_gray_family_comparison(
    parquet_path: str,
    out_dir: str = "exp_outputs/summary/plots/folds",
) -> Path:
    apply_miccai_style()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = _load_scope(parquet_path)
    plot_df = _collect_pairwise_effects(df)
    plot_df = _center_dataset_effects(plot_df)
    if plot_df.empty:
        raise RuntimeError(
            "No valid white-vs-gray paired effects found for the selected combos and families"
        )

    _plot_family_comparison(
        plot_df,
        output_file=out_path / "white_vs_gray_dataset_task_linear_rf_deep.pdf",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot white-vs-gray normalized score differences by dataset-target for "
            "Linear, Random Forest, and Deep model families"
        )
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

    out_path = generate_white_vs_gray_family_comparison(args.input, args.outdir)
    print("Saved white-vs-gray family comparison plot to", out_path)


if __name__ == "__main__":
    main()
