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
    LINEAR_MODELS,
    RANDOM_FOREST_MODELS,
    calculate_paired_ttest,
    choose_spread_metric,
    clean_target,
    filter_combos,
    fold_columns,
    format_label,
    get_display_label,
    is_dummy_model,
    select_best_runs,
)

CLASSICAL_MODELS = LINEAR_MODELS | RANDOM_FOREST_MODELS

PLOT_TITLES = {
    "full": "White vs Gray Matter: Normalized Score Difference (All Dataset/Target/Task/Feature)",
    "dataset": "White vs Gray Matter: Dataset Tissue Effect (Feature Baseline Removed)",
    "feature": "White vs Gray Matter: Feature Tissue Effect (Dataset-Task Baseline Removed)",
    "pair": "White vs Gray Matter: Aggregated Tissue Effect Comparison",
}
X_AXIS_LABEL = "\u0394 Normalized score (white \u2212 gray)"
LEFT_REGION_LABEL = "Gray matter wins"
RIGHT_REGION_LABEL = "White matter wins"
POINTS_COLOR = "#4C78A8"
MEAN_STD_COLOR = "#B22222"
BOX_COLOR = "#D6E3F3"
AGG_PANEL_XLABELS = ("By Dataset (feature effect removed)", "By Feature (dataset effect removed)")
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


def _normalize_fold_val(val: float, prediction_task: str) -> float:
    """Map a raw fold metric value to [0, 1] relative to dummy baseline.

    - Binary classification (balanced accuracy): dummy = 0.5, perfect = 1.0
    - Regression (R²):                           dummy = 0.0, perfect = 1.0
    """
    if prediction_task == "binary_classification":
        return (val - 0.5) / 0.5
    return float(val)  # R²: already 0 = dummy, 1 = perfect


def _collect_pairwise_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Collect fold-level normalized score differences (white − gray) per dataset/target/task/feature."""
    rows = []

    for (dataset, target, task), group in df.groupby(["dataset", "target_clean", "prediction_task"]):
        try:
            fold_prefix, metric_label = choose_spread_metric(group, task)
        except (RuntimeError, ValueError):
            continue

        # choose_spread_metric always returns higher-is-better metrics (R² / balanced accuracy)
        higher_is_better = True
        best = select_best_runs(group, fold_prefix, higher_is_better)
        if best.empty or "_fold_mean" not in best.columns:
            continue

        best = best[~best["model_name"].apply(is_dummy_model)]
        best = best[best["model_name"].isin(CLASSICAL_MODELS)]
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

            white_vals = np.asarray(best_white[fold_cols].values, dtype=float)
            gray_vals  = np.asarray(best_gray[fold_cols].values, dtype=float)

            valid = np.isfinite(white_vals) & np.isfinite(gray_vals)
            white_vals = white_vals[valid]
            gray_vals  = gray_vals[valid]
            if white_vals.size == 0:
                continue

            # Normalize each fold to [0,1] then subtract: positive = white matter wins
            norm_white = np.array([_normalize_fold_val(v, task) for v in white_vals])
            norm_gray  = np.array([_normalize_fold_val(v, task) for v in gray_vals])
            diffs = norm_white - norm_gray

            # P-value from paired t-test on raw values (linear scaling preserves significance)
            p_value = calculate_paired_ttest(white_vals, gray_vals)

            for fold_idx, diff in enumerate(diffs):
                rows.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "task": task,
                        "feature": feature,
                        "metric_label": metric_label,
                        "fold": fold_idx,
                        "normalized_diff": float(diff),
                        "p_value": float(p_value),
                    }
                )

    return pd.DataFrame(rows)


def _center_dataset_effects(diffs: pd.DataFrame) -> pd.DataFrame:
    """Remove feature baseline offsets to isolate dataset-task-specific tissue effects.

    Procedure:
    1. Collapse fold-level diffs to one mean effect per (dataset, target, task, feature)
       so that fold count imbalance does not distort the baseline estimate.
    2. Compute the feature baseline: mean of those per-combination means across all
       (dataset, target, task) combinations for each feature.
    3. Residualize each individual fold: normalized_diff -= feature_baseline.

    Fold-level rows are preserved so individual folds appear in the plot.
    """
    group_cols = ["dataset", "target", "task", "feature", "metric_label"]
    # Step 1: per-combination means for baseline estimation
    combo_means = (
        diffs.groupby(group_cols, as_index=False)["normalized_diff"]
        .mean()
    )

    # Step 2: feature baseline = mean of combo means across dataset/target/task
    cluster_means = (
        combo_means.groupby(["feature"])["normalized_diff"]
        .mean()
        .rename("cluster_mean")
        .reset_index()
    )

    # Step 3: apply residual to every fold-level row
    result = diffs.merge(cluster_means, on=["feature"], how="left")
    result = result.copy()
    result["normalized_diff"] = result["normalized_diff"] - result["cluster_mean"]
    result = result.drop(columns=["cluster_mean"])

    return result


def _center_feature_effects(diffs: pd.DataFrame) -> pd.DataFrame:
    """Remove dataset-task baseline offsets to isolate feature-specific tissue effects.

    Procedure:
    1. Collapse fold-level diffs to one mean effect per (dataset, target, task, feature)
       so that fold count imbalance does not distort the baseline estimate.
    2. Compute the dataset-task cluster baseline: mean of those per-combination means
       across all features within each (dataset, target, task) group.
    3. Residualize each individual fold: normalized_diff -= cluster_baseline.

    Fold-level rows are preserved so individual folds appear in the plot.
    """
    group_cols = ["dataset", "target", "task", "feature", "metric_label"]
    # Step 1: per-combination means for baseline estimation
    combo_means = (
        diffs.groupby(group_cols, as_index=False)["normalized_diff"]
        .mean()
    )

    # Step 2: dataset-task baseline = mean of combo means across features in each cluster
    cluster_cols = ["dataset", "target", "task"]
    cluster_means = (
        combo_means.groupby(cluster_cols)["normalized_diff"]
        .mean()
        .rename("cluster_mean")
        .reset_index()
    )

    # Step 3: apply residual to every fold-level row
    result = diffs.merge(cluster_means, on=cluster_cols, how="left")
    result = result.copy()
    result["normalized_diff"] = result["normalized_diff"] - result["cluster_mean"]
    result = result.drop(columns=["cluster_mean"])

    return result


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


def _cluster_mean_effect_size(
    values: np.ndarray,
    cluster_ids: np.ndarray,
) -> float:
    """
    Compute paired Cohen's d using one mean effect per independent cluster.
    """
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)

    valid_mask = np.isfinite(values)
    values = values[valid_mask]
    cluster_ids = cluster_ids[valid_mask]
    if values.size == 0:
        return float("nan")

    cluster_means = (
        pd.DataFrame({"cluster_id": cluster_ids, "value": values})
        .groupby("cluster_id", sort=False)["value"]
        .mean()
        .to_numpy(dtype=float)
    )

    n_clusters = cluster_means.size
    if n_clusters < 2:
        return float("nan")

    mean_effect = np.mean(cluster_means)
    std_effect = np.std(cluster_means, ddof=1)

    if std_effect == 0:
        return float("nan")

    return float(mean_effect / std_effect)


def _compute_aggregated_label_effect_sizes(plot_df: pd.DataFrame, order: list[str]) -> pd.Series:
    effect_sizes: dict[str, float] = {}
    for label in order:
        label_rows = plot_df[plot_df["label"] == label]
        cluster_ids = label_rows[["dataset", "target", "task", "feature"]].astype(str).agg(
            "|".join,
            axis=1,
        )

        effect_sizes[label] = _cluster_mean_effect_size(
            label_rows["normalized_diff"].to_numpy(dtype=float),
            cluster_ids.to_numpy(),
        )

    return pd.Series(effect_sizes).reindex(order)


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


def _annotate_panel_statistics(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    order: list[str],
    y_lim: float,
) -> None:
    pvals = _compute_aggregated_label_pvalues(plot_df, order)
    effects = _compute_aggregated_label_effect_sizes(plot_df, order)

    ymax = plot_df.groupby("label")["normalized_diff"].max().reindex(order)
    y_padding = 0.06 * y_lim
    y_cap = 0.86 * y_lim

    for xpos, label in enumerate(order):
        p_value = pvals.get(label, np.nan)
        d_value = effects.get(label, np.nan)

        box_top = ymax.get(label, np.nan)
        y_text = y_cap if not np.isfinite(box_top) else min(float(box_top) + y_padding, y_cap)

        stat_label = f"p={p_value:.4f}\nd={d_value:.2f}"

        ax.text(
            xpos,
            y_text,
            stat_label,
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

    y_lim = 0.8
    forced_ticks = [-0.8, -0.4, 0.0, 0.4, 0.8]

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
        _annotate_panel_statistics(ax, panel_df, panel_order, y_lim)
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
        output_file=out_path / "white_vs_gray_tscore.pdf",
    )

    centered_diffs_dataset = _center_dataset_effects(diffs)
    dataset_df = _build_labels(
        centered_diffs_dataset,
        view="dataset",
        aggregate_tasks=aggregate_tasks_dataset,
    )
    dataset_suffix = "dataset" if aggregate_tasks_dataset else "dataset_task_couples"
    _plot_box_with_points(
        dataset_df,
        title=PLOT_TITLES["dataset"],
        output_file=out_path / f"white_vs_gray_tscore_{dataset_suffix}.pdf",
    )

    centered_diffs = _center_feature_effects(diffs)
    feature_df = _build_labels(
        centered_diffs,
        view="feature",
        aggregate_tasks=aggregate_tasks_feature,
    )
    feature_suffix = "feature" if aggregate_tasks_feature else "feature_task_couples"
    _plot_box_with_points(
        feature_df,
        title=PLOT_TITLES["feature"],
        output_file=out_path / f"white_vs_gray_tscore_{feature_suffix}.pdf",
    )

    _plot_aggregated_pair_comparison(
        dataset_df,
        feature_df,
        output_file=out_path / f"white_vs_gray_tscore_{dataset_suffix}_{feature_suffix}_combined.pdf",
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
        default="exp_outputs/summary/plots/test",
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