from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    MICCAI_DOUBLE_COLUMN_FIGSIZE,
    apply_miccai_style,
)
from utils import (
    DEFAULT_COMBOS,
    calculate_paired_stats,
    calculate_paired_ttest,
    choose_fold_metric,
    clean_target,
    filter_combos,
    fold_columns,
    format_label,
    get_display_label,
    is_dummy_model,
    select_best_runs,
)

PLOT_TITLES = {
    "full": "White vs Gray Matter: Standardized Difference (All Dataset/Target/Task/Feature)",
    "dataset": "White vs Gray Matter: Standardized Difference (Dataset Aggregated)",
    "feature": "White vs Gray Matter: Standardized Difference (Feature Aggregated)",
    "pair": "White vs Gray Matter: Aggregated Tissue Effect Comparison",
}
X_AXIS_LABEL = "Standardized Paired Fold Difference"
LEFT_REGION_LABEL = "Gray matter wins"
RIGHT_REGION_LABEL = "White matter wins"
POINTS_COLOR = "#4C78A8"
MEAN_STD_COLOR = "#B22222"
BOX_COLOR = "#D6E3F3"
AGG_PANEL_XLABELS = ("By Dataset", "By Feature")
TOP_REGION_COLOR = "#D6E4F1"
BOTTOM_REGION_COLOR = "#FAEFDB"
TOP_REGION_LABEL = "White matter wins"
BOTTOM_REGION_LABEL = "Gray matter wins"


def _load_filtered_results(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["target_clean"] = df["target"].map(clean_target)
    df = filter_combos(df, DEFAULT_COMBOS)
    if df.empty:
        raise RuntimeError("No rows left after filtering combos")
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


def _run_white_vs_gray_paired_test(
    white_vals: np.ndarray,
    gray_vals: np.ndarray,
    higher_is_better: bool,
) -> dict[str, float | np.ndarray] | None:
    white_vals = np.asarray(white_vals, dtype=float)
    gray_vals = np.asarray(gray_vals, dtype=float)

    if white_vals.shape != gray_vals.shape:
        return None

    valid_mask = np.isfinite(white_vals) & np.isfinite(gray_vals)
    white_vals = white_vals[valid_mask]
    gray_vals = gray_vals[valid_mask]
    if white_vals.size == 0:
        return None

    t_score, mean_diff, std_diff, normalized_diffs = calculate_paired_stats(
        white_vals,
        gray_vals,
        higher_is_better,
    )

    if std_diff == 0.0:
        if mean_diff == 0.0:
            return None
        normalized_diffs = np.full(white_vals.shape[0], np.sign(mean_diff), dtype=float)
        t_score = float(np.sign(mean_diff))

    p_value = calculate_paired_ttest(white_vals, gray_vals)

    return {
        "t_score": float(t_score),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "p_value": float(p_value),
        "normalized_diffs": np.asarray(normalized_diffs, dtype=float),
    }


def _collect_pairwise_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Collect fold-level standardized white-vs-gray effects for each dataset/target/task/feature."""
    rows = []

    for (dataset, target, task), group in df.groupby(["dataset", "target_clean", "prediction_task"]):
        fold_prefix, metric_label, higher_is_better = choose_fold_metric(group, task)
        best = select_best_runs(group, fold_prefix, higher_is_better)
        if best.empty or "_fold_mean" not in best.columns:
            continue

        best = best[~best["model_name"].apply(is_dummy_model)]
        if best.empty:
            continue

        fold_cols = fold_columns(best, fold_prefix)
        if not fold_cols:
            continue

        for feature, feature_df in best.groupby("primary_metric"):
            best_tissues = _select_best_tissue_rows(feature_df, higher_is_better)
            if best_tissues is None:
                continue
            best_white, best_gray = best_tissues

            stats = _run_white_vs_gray_paired_test(
                best_white[fold_cols].values,
                best_gray[fold_cols].values,
                higher_is_better,
            )
            if stats is None:
                continue

            for fold_idx, norm_diff in enumerate(stats["normalized_diffs"]):
                rows.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "task": task,
                        "feature": feature,
                        "metric_label": metric_label,
                        "fold": fold_idx,
                        "normalized_diff": float(norm_diff),
                        "t_score": float(stats["t_score"]),
                        "mean_diff": float(stats["mean_diff"]),
                        "std_diff": float(stats["std_diff"]),
                        "p_value": float(stats["p_value"]),
                    }
                )

    return pd.DataFrame(rows)


def _build_labels(
    diffs: pd.DataFrame,
    view: str,
    aggregate_tasks: bool,
) -> pd.DataFrame:
    plot_df = diffs.copy()

    if view == "full":
        plot_df["label"] = plot_df.apply(
            lambda r: (
                f"{get_display_label(r['dataset'], r['target'], r['task'], r['metric_label'])}"
                f" | {format_label(r['feature'])}"
            ),
            axis=1,
        )
        return plot_df

    if view == "dataset":
        if aggregate_tasks:
            plot_df["label"] = plot_df["dataset"].map(format_label)
        else:
            plot_df["label"] = plot_df.apply(
                lambda r: f"{format_label(r['dataset'])} | {format_label(r['task'])}",
                axis=1,
            )
        return plot_df

    if view == "feature":
        if aggregate_tasks:
            plot_df["label"] = plot_df["feature"].map(format_label)
        else:
            plot_df["label"] = plot_df.apply(
                lambda r: f"{format_label(r['feature'])} | {format_label(r['task'])}",
                axis=1,
            )
        return plot_df

    raise ValueError(f"Unknown view: {view}")


def _get_plot_order(plot_df: pd.DataFrame) -> list[str]:
    return (
        plot_df.groupby("label")["normalized_diff"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )


def _init_plot_canvas(n_labels: int) -> tuple[plt.Figure, plt.Axes]:
    base_w, base_h = MICCAI_DOUBLE_COLUMN_FIGSIZE
    fig_h = max(base_h, 0.42 * n_labels)
    return plt.subplots(figsize=(base_w, fig_h))


def _finalize_effect_axis(ax: plt.Axes, title: str) -> None:
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel("")
    ax.set_title(title, pad=14)
    ax.text(
        0.02,
        1.01,
        LEFT_REGION_LABEL,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        color="gray",
    )
    ax.text(
        0.98,
        1.01,
        RIGHT_REGION_LABEL,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
        color="gray",
    )


def _plot_strip_with_mean_std(
    plot_df: pd.DataFrame,
    title: str,
    output_file: Path,
) -> None:
    if plot_df.empty:
        raise RuntimeError("No white/gray differences available for plotting")

    order = _get_plot_order(plot_df)
    fig, ax = _init_plot_canvas(len(order))

    sns.stripplot(
        data=plot_df,
        x="normalized_diff",
        y="label",
        order=order,
        color=POINTS_COLOR,
        size=5,
        alpha=0.6,
        jitter=0.15,
        ax=ax,
    )

    summary = (
        plot_df.groupby("label")["normalized_diff"]
        .agg(mean="mean", std=lambda s: s.std(ddof=1))
        .reindex(order)
    )
    summary["std"] = summary["std"].fillna(0.0)
    y_pos = np.arange(len(order))

    ax.errorbar(
        x=summary["mean"].to_numpy(dtype=float),
        y=y_pos,
        xerr=summary["std"].to_numpy(dtype=float),
        fmt="D",
        color=MEAN_STD_COLOR,
        ecolor=MEAN_STD_COLOR,
        elinewidth=1.2,
        capsize=3,
        markersize=5,
        zorder=5,
    )

    _finalize_effect_axis(ax, title)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def _draw_box_with_points(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    order: list[str],
    vertical: bool,
) -> None:
    if vertical:
        sns.boxplot(
            data=plot_df,
            x="label",
            y="normalized_diff",
            order=order,
            color=BOX_COLOR,
            width=0.55,
            linewidth=1.0,
            showfliers=False,
            ax=ax,
        )
        sns.stripplot(
            data=plot_df,
            x="label",
            y="normalized_diff",
            order=order,
            color=POINTS_COLOR,
            size=3.5,
            alpha=0.5,
            jitter=0.22,
            ax=ax,
        )
    else:
        sns.boxplot(
            data=plot_df,
            x="normalized_diff",
            y="label",
            order=order,
            color=BOX_COLOR,
            width=0.55,
            linewidth=1.0,
            showfliers=False,
            ax=ax,
        )
        sns.stripplot(
            data=plot_df,
            x="normalized_diff",
            y="label",
            order=order,
            color=POINTS_COLOR,
            size=4,
            alpha=0.5,
            jitter=0.18,
            ax=ax,
        )


def _format_p_value_label(p_value: float) -> str:
    return f"pval={p_value:.5f}"


def _cluster_mean_ttest_pvalue(
    values: np.ndarray,
    cluster_ids: np.ndarray,
) -> float:
    """Aggregate to one effect per independent pair, then test mean effect vs 0."""
    from scipy.stats import ttest_1samp

    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)

    valid_mask = np.isfinite(values)
    values = values[valid_mask]
    cluster_ids = cluster_ids[valid_mask]
    if values.size == 0:
        return float("nan")

    # One independent effect per dataset/target/task(/feature) pair.
    cluster_means = (
        pd.DataFrame({"cluster_id": cluster_ids, "value": values})
        .groupby("cluster_id", sort=False)["value"]
        .mean()
        .to_numpy(dtype=float)
    )
    n_clusters = cluster_means.size
    if n_clusters < 2:
        return float("nan")

    res = ttest_1samp(cluster_means, popmean=0.0, nan_policy="omit")
    if np.isnan(res.pvalue):
        return float("nan")
    return float(res.pvalue)


def _compute_aggregated_label_pvalues(plot_df: pd.DataFrame, order: list[str]) -> pd.Series:
    pvals: dict[str, float] = {}
    for label in order:
        label_rows = plot_df[plot_df["label"] == label]
        cluster_ids = label_rows[["dataset", "target", "task", "feature"]].astype(str).agg(
            "|".join,
            axis=1,
        )
        pvals[label] = _cluster_mean_ttest_pvalue(
            label_rows["normalized_diff"].to_numpy(dtype=float),
            cluster_ids.to_numpy(),
        )
    return pd.Series(pvals).reindex(order)


def _annotate_panel_p_values(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    order: list[str],
    y_lim: float,
) -> None:
    pvals = _compute_aggregated_label_pvalues(plot_df, order)
    ymax = plot_df.groupby("label")["normalized_diff"].max().reindex(order)
    y_padding = 0.06 * y_lim
    y_cap = 0.86 * y_lim

    for xpos, label in enumerate(order):
        p_value = pvals.get(label, np.nan)
        box_top = ymax.get(label, np.nan)
        y_text = y_cap if not np.isfinite(box_top) else min(float(box_top) + y_padding, y_cap)
        ax.text(
            xpos,
            y_text,
            _format_p_value_label(float(p_value)),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#303030",
        )


def _plot_box_with_points(
    plot_df: pd.DataFrame,
    title: str,
    output_file: Path,
) -> None:
    if plot_df.empty:
        raise RuntimeError("No white/gray differences available for plotting")

    order = _get_plot_order(plot_df)
    fig, ax = _init_plot_canvas(len(order))
    _draw_box_with_points(ax, plot_df, order, vertical=False)

    _finalize_effect_axis(ax, title)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def _plot_aggregated_pair_comparison(
    dataset_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    output_file: Path,
) -> None:
    if dataset_df.empty or feature_df.empty:
        raise RuntimeError("No aggregated white/gray differences available for combined plotting")

    dataset_order = _get_plot_order(dataset_df)
    feature_order = _get_plot_order(feature_df)

    base_w, base_h = MICCAI_DOUBLE_COLUMN_FIGSIZE
    # The combined plot needs more height than the base figsize to accommodate
    # a suptitle, supylabel, two sets of rotated x-tick labels, and the plot area.
    fig_h = 3.0

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(base_w, fig_h),
        sharey=True,
        constrained_layout=True,
    )

    _draw_box_with_points(axes[0], dataset_df, dataset_order, vertical=True)
    _draw_box_with_points(axes[1], feature_df, feature_order, vertical=True)

    all_vals = pd.concat([dataset_df["normalized_diff"], feature_df["normalized_diff"]], axis=0)
    max_abs = float(np.nanmax(np.abs(all_vals.to_numpy(dtype=float))))
    y_lim = max(4.0, 1.1 * max_abs)
    forced_ticks = [-4, -2, 0, 2, 4]

    panel_defs = (
        (axes[0], dataset_df, dataset_order),
        (axes[1], feature_df, feature_order),
    )
    for ax, panel_df, panel_order in panel_defs:
        ax.axhspan(0.0, y_lim, color=TOP_REGION_COLOR, alpha=0.32, zorder=0)
        ax.axhspan(-y_lim, 0.0, color=BOTTOM_REGION_COLOR, alpha=0.32, zorder=0)
        ax.axhline(0.0, color="#333333", linewidth=1.0)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_yticks(forced_ticks)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35)
        _annotate_panel_p_values(ax, panel_df, panel_order, y_lim)
        ax.text(
            0.5,
            y_lim * 0.93,
            TOP_REGION_LABEL,
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="top",
            fontweight="bold",
            color="#606060",
        )
        ax.text(
            0.5,
            -y_lim * 0.93,
            BOTTOM_REGION_LABEL,
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="bottom",
            fontweight="bold",
            color="#606060",
        )

    axes[0].set_xlabel(AGG_PANEL_XLABELS[0])
    axes[1].set_xlabel(AGG_PANEL_XLABELS[1])

    # Let constrained_layout position suptitle/supylabel automatically —
    # no hardcoded x/y offsets needed.
    fig.suptitle(PLOT_TITLES["pair"])
    fig.supylabel(X_AXIS_LABEL)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def generate_white_vs_gray_plots(
    parquet_path: str,
    out_dir: str = "exp_outputs/summary/plots/folds",
    aggregate_tasks_dataset: bool = True,
    aggregate_tasks_feature: bool = True,
) -> Path:
    """Generate all white-vs-gray plot variants in a single run."""
    apply_miccai_style()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = _load_filtered_results(parquet_path)
    diffs = _collect_pairwise_effects(df)
    if diffs.empty:
        raise RuntimeError("No valid white/gray differences found")

    full_df = _build_labels(diffs, view="full", aggregate_tasks=False)
    _plot_strip_with_mean_std(
        full_df,
        title=PLOT_TITLES["full"],
        output_file=out_path / "white_vs_gray_tscore.png",
    )

    dataset_df = _build_labels(
        diffs,
        view="dataset",
        aggregate_tasks=aggregate_tasks_dataset,
    )
    dataset_suffix = "dataset" if aggregate_tasks_dataset else "dataset_task_couples"
    _plot_box_with_points(
        dataset_df,
        title=PLOT_TITLES["dataset"],
        output_file=out_path / f"white_vs_gray_tscore_{dataset_suffix}.png",
    )

    feature_df = _build_labels(
        diffs,
        view="feature",
        aggregate_tasks=aggregate_tasks_feature,
    )
    feature_suffix = "feature" if aggregate_tasks_feature else "feature_task_couples"
    _plot_box_with_points(
        feature_df,
        title=PLOT_TITLES["feature"],
        output_file=out_path / f"white_vs_gray_tscore_{feature_suffix}.png",
    )

    _plot_aggregated_pair_comparison(
        dataset_df,
        feature_df,
        output_file=out_path / f"white_vs_gray_tscore_{dataset_suffix}_{feature_suffix}_combined.png",
    )

    return out_path


def plot_white_vs_gray_tscore(
    parquet_path: str,
    out_dir: str = "exp_outputs/summary/plots/folds",
    aggregate_tasks_dataset: bool = True,
    aggregate_tasks_feature: bool = True,
) -> Path:
    """Backward-compatible wrapper kept for existing callers."""
    return generate_white_vs_gray_plots(
        parquet_path,
        out_dir=out_dir,
        aggregate_tasks_dataset=aggregate_tasks_dataset,
        aggregate_tasks_feature=aggregate_tasks_feature,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate white-vs-gray tissue effect plots: full, dataset-aggregated, "
            "and feature-aggregated"
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
    parser.add_argument(
        "--dataset-task-couples",
        action="store_true",
        help="Show dataset|task couples instead of aggregating tasks in dataset view",
    )
    parser.add_argument(
        "--feature-task-couples",
        action="store_true",
        help="Show feature|task couples instead of aggregating tasks in feature view",
    )
    args = parser.parse_args()

    out_path = generate_white_vs_gray_plots(
        args.input,
        out_dir=args.outdir,
        aggregate_tasks_dataset=not args.dataset_task_couples,
        aggregate_tasks_feature=not args.feature_task_couples,
    )
    print("Saved white vs gray plots to", out_path)


if __name__ == "__main__":
    main()
