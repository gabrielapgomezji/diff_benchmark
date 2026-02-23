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

PLOT_TITLES = {
    "full": "White vs Gray Matter: Normalized Score Difference (All Dataset/Target/Task/Feature)",
    "dataset":            "White vs Gray Matter: Dataset Tissue Effect (Feature Baseline Removed)",
    "dataset_raw":        "White vs Gray Matter: Dataset Tissue Effect (Raw)",
    "feature":            "White vs Gray Matter: Feature Tissue Effect (Dataset-Task Baseline Removed)",
    "feature_raw":        "White vs Gray Matter: Feature Tissue Effect (Raw)",
    "task":               "White vs Gray Matter: Task Tissue Effect (Dataset-Feature Baseline Removed)",
    "task_raw":           "White vs Gray Matter: Task Tissue Effect (Raw)",
    "pair": "White vs Gray Matter: Aggregated Tissue Effect Comparison",
}
X_AXIS_LABEL = "\u0394 Normalized score (white \u2212 gray)"
LEFT_REGION_LABEL = "Gray matter wins"
RIGHT_REGION_LABEL = "White matter wins"
POINTS_COLOR = "#4C78A8"
MEAN_STD_COLOR = "#B22222"
BOX_COLOR = "#D6E3F3"
AGG_PANEL_XLABELS = {
    "feature": ("By Dataset (feature effect removed)", "By Feature (dataset effect removed)"),
    "task":    ("By Dataset (task effect removed)",    "By Dataset × Target (microstructure effect removed)"),
}
TOP_REGION_COLOR = "#D6E4F1"
BOTTOM_REGION_COLOR = "#FAEFDB"
TOP_REGION_LABEL = "White matter wins"
BOTTOM_REGION_LABEL = "Gray matter wins"

# Restrict to these four dataset × target combinations only.
# Each entry is (dataset, target, task).
LOCAL_COMBOS = [
    ("hcp",    "Gender",   "binary_classification"),
    ("camcan", "Gender",   "binary_classification"),
    ("camcan", "Age",      "regression"),
    ("abide",  "DX_GROUP", "binary_classification"),
]

# Human-readable labels for the LOCAL_COMBOS, keyed by (dataset, target).
COMBO_DISPLAY_LABELS: dict[tuple[str, str], str] = {
    ("hcp",    "Gender"):   "HCP – Gender",
    ("camcan", "Gender"):   "CamCAN – Gender",
    ("camcan", "Age"):      "CamCAN – Age",
    ("abide",  "DX_GROUP"): "ABIDE – DX Group",
}

# Fixed display order for the task / dataset×target panel:
# Age first, then Gender (HCP then CamCAN), then DX_GROUP.
TASK_PANEL_ORDER: list[str] = [
    "CamCAN – Age",
    "HCP – Gender",
    "CamCAN – Gender",
    "ABIDE – DX Group",
]


def _dataset_target_label(dataset: str, target: str) -> str:
    """Return the human-readable label for a (dataset, target) pair."""
    return COMBO_DISPLAY_LABELS.get((dataset, target), format_label(f"{dataset} {target}"))


def _load_filtered_results(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["target_clean"] = df["target"].map(clean_target)
    df = filter_combos(df, LOCAL_COMBOS)
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


def _center_task_effects(diffs: pd.DataFrame) -> pd.DataFrame:
    """Remove dataset-feature baseline offsets to isolate task-specific tissue effects.

    Procedure:
    1. Collapse fold-level diffs to one mean effect per (dataset, target, task, feature)
       so that fold count imbalance does not distort the baseline estimate.
    2. Compute the dataset-feature cluster baseline: mean of those per-combination means
       across all tasks within each (dataset, target, feature) group.
    3. Residualize each individual fold: normalized_diff -= cluster_baseline.

    Fold-level rows are preserved so individual folds appear in the plot.
    """
    group_cols = ["dataset", "target", "task", "feature", "metric_label"]
    # Step 1: per-combination means for baseline estimation
    combo_means = (
        diffs.groupby(group_cols, as_index=False)["normalized_diff"]
        .mean()
    )

    # Step 2: dataset-feature baseline = mean of combo means across tasks in each cluster
    cluster_cols = ["dataset", "target", "feature"]
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

    if view == "task":
        plot_df["label"] = plot_df.apply(
            lambda r: _dataset_target_label(r["dataset"], r["target"]),
            axis=1,
        )
        return plot_df

    raise ValueError(f"Unknown view: {view}")


def _get_plot_order(
    plot_df: pd.DataFrame,
    fixed_order: list[str] | None = None,
) -> list[str]:
    if fixed_order is not None:
        # Keep only labels that are actually present in the data, in the given order,
        # then append any remaining labels sorted by mean (fallback).
        present = set(plot_df["label"].unique())
        ordered = [lbl for lbl in fixed_order if lbl in present]
        remaining = sorted(present - set(ordered))
        return ordered + remaining
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
    fixed_order: list[str] | None = None,
) -> None:
    if plot_df.empty:
        raise RuntimeError("No white/gray differences available for plotting")

    order = _get_plot_order(plot_df, fixed_order)
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
    second_panel: str = "task",
    second_fixed_order: list[str] | None = None,
) -> None:
    if dataset_df.empty or feature_df.empty:
        raise RuntimeError("No aggregated white/gray differences available for combined plotting")

    dataset_order = _get_plot_order(dataset_df)
    feature_order = _get_plot_order(feature_df, second_fixed_order)

    base_w, base_h = MICCAI_DOUBLE_COLUMN_FIGSIZE
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

    xlabel_left, xlabel_right = AGG_PANEL_XLABELS[second_panel]
    axes[0].set_xlabel(xlabel_left)
    axes[1].set_xlabel(xlabel_right)

    fig.suptitle(PLOT_TITLES["pair"])
    fig.supylabel(X_AXIS_LABEL)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def generate_white_vs_gray_plots(
    parquet_path: str,
    out_dir: str = "exp_outputs/summary/plots/folds",
    aggregate_tasks_dataset: bool = True,
    aggregate_tasks_feature: bool = True,
    second_panel: str = "feature",
    center: bool = True,
) -> Path:
    """Generate all white-vs-gray plot variants in a single run.

    Parameters
    ----------
    second_panel : {"feature", "task"}
        Controls what the right panel of the combined plot shows:
        - ``"feature"`` (default): tissue effect broken down by microstructural
          feature (dataset-task baseline removed).
        - ``"task"``: tissue effect broken down by prediction task
          (dataset-feature baseline removed).
    center : bool
        If ``True`` (default), the panel-specific baseline is subtracted from
        each panel's diffs before plotting (residualized / centered version).
        If ``False``, raw normalized fold differences (white − gray) are plotted
        directly with no baseline removal.
    """
    if second_panel not in {"feature", "task"}:
        raise ValueError(f"second_panel must be 'feature' or 'task', got {second_panel!r}")

    apply_miccai_style()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = _load_filtered_results(parquet_path)
    diffs = _collect_pairwise_effects(df)
    if diffs.empty:
        raise RuntimeError("No valid white/gray differences found")

    n_combos   = diffs[["dataset", "target", "task"]].drop_duplicates().shape[0]
    n_features = diffs["feature"].nunique()
    n_folds    = diffs["fold"].nunique()
    centering_label = "ON (baseline-residualized)" if center else "OFF (raw normalized diffs)"
    print(
        "\n── Score normalization summary ─────────────────────────────────────────────\n"
        f"  Dataset × target combos : {n_combos}  |  Features : {n_features}  |  Folds : {n_folds}\n"
        f"  Centering (--no-centering flag) : {centering_label}\n"
        "\n"
        "  Raw metric → [0, 1] per fold  (_normalize_fold_val)\n"
        "    • Binary classification (balanced accuracy): (val − 0.5) / 0.5\n"
        "      dummy = 0 │ perfect = 1\n"
        "    • Regression (R²):  val  (already 0 = dummy, 1 = perfect)\n"
        "\n"
    )
    if center:
        print(
            "  LEFT panel  — 'By Dataset'  [centered]\n"
            "    Baseline removed: mean tissue effect averaged across ALL features,\n"
            "    so that cross-feature differences do not inflate/deflate a dataset.\n"
            "    Residual = fold diff − mean(fold diff | same feature, all dataset×target)\n"
            "\n"
        )
        if second_panel == "feature":
            print(
                "  RIGHT panel — 'By Feature'  [centered]\n"
                "    Baseline removed: mean tissue effect averaged across ALL dataset×target\n"
                "    combos, so that dataset-level differences do not inflate a feature.\n"
                "    Residual = fold diff − mean(fold diff | same dataset×target, all features)\n"
                "────────────────────────────────────────────────────────────────────────────\n"
            )
        else:
            print(
                "  RIGHT panel — 'By Dataset × Target'  [centered, microstructure effect removed]\n"
                "    Baseline removed: mean tissue effect averaged across ALL microstructural\n"
                "    features within each dataset×target combo.\n"
                "    Residual = fold diff − mean(fold diff | same dataset×target & feature, all tasks)\n"
                "────────────────────────────────────────────────────────────────────────────\n"
            )
    else:
        print(
            "  Both panels — raw normalized diffs  [no centering]\n"
            "    Values shown are: (white_score − gray_score) after [0,1] rescaling per fold.\n"
            "    No baseline is subtracted; differences across datasets/features/tasks\n"
            "    are confounded and should be interpreted with caution.\n"
            "────────────────────────────────────────────────────────────────────────────\n"
        )

    center_suffix = "" if center else "_raw"

    full_df = _build_labels(diffs, view="full", aggregate_tasks=False)
    _plot_strip_with_mean_std(
        full_df,
        title=PLOT_TITLES["full"],
        output_file=out_path / "white_vs_gray_tscore.png",
    )

    dataset_diffs = _center_dataset_effects(diffs) if center else diffs
    dataset_df = _build_labels(
        dataset_diffs,
        view="dataset",
        aggregate_tasks=aggregate_tasks_dataset,
    )
    dataset_key = "dataset" if center else "dataset_raw"
    dataset_suffix = ("dataset" if aggregate_tasks_dataset else "dataset_task_couples") + center_suffix
    _plot_box_with_points(
        dataset_df,
        title=PLOT_TITLES[dataset_key],
        output_file=out_path / f"white_vs_gray_tscore_{dataset_suffix}.png",
    )

    if second_panel == "feature":
        second_diffs = _center_feature_effects(diffs) if center else diffs
        second_df = _build_labels(
            second_diffs,
            view="feature",
            aggregate_tasks=aggregate_tasks_feature,
        )
        second_key = "feature" if center else "feature_raw"
        second_suffix = ("feature" if aggregate_tasks_feature else "feature_task_couples") + center_suffix
        second_fixed_order = None
    else:  # "task"
        second_diffs = _center_task_effects(diffs) if center else diffs
        second_df = _build_labels(second_diffs, view="task", aggregate_tasks=False)
        second_key = "task" if center else "task_raw"
        second_suffix = "task" + center_suffix
        second_fixed_order = TASK_PANEL_ORDER

    _plot_box_with_points(
        second_df,
        title=PLOT_TITLES[second_key],
        output_file=out_path / f"white_vs_gray_tscore_{second_suffix}.png",
        fixed_order=second_fixed_order,
    )

    _plot_aggregated_pair_comparison(
        dataset_df,
        second_df,
        output_file=out_path / f"white_vs_gray_tscore_{dataset_suffix}_{second_suffix}_combined.png",
        second_panel=second_panel,
        second_fixed_order=second_fixed_order,
    )

    return out_path


def plot_white_vs_gray_tscore(
    parquet_path: str,
    out_dir: str = "exp_outputs/summary/plots/folds",
    aggregate_tasks_dataset: bool = True,
    aggregate_tasks_feature: bool = True,
    second_panel: str = "feature",
    center: bool = True,
) -> Path:
    """Backward-compatible wrapper kept for existing callers."""
    return generate_white_vs_gray_plots(
        parquet_path,
        out_dir=out_dir,
        aggregate_tasks_dataset=aggregate_tasks_dataset,
        aggregate_tasks_feature=aggregate_tasks_feature,
        second_panel=second_panel,
        center=center,
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
    parser.add_argument(
        "--second-panel",
        choices=["feature", "task"],
        default="feature",
        help=(
            "What to show in the right panel of the combined plot: "
            "'feature' (microstructural feature tissue effect, default) or "
            "'task' (prediction-task tissue effect)."
        ),
    )
    parser.add_argument(
        "--no-centering",
        action="store_true",
        help=(
            "Disable baseline removal (centering). By default, each panel removes "
            "its confounding factor (feature or task effect) before plotting. "
            "Use this flag to plot raw normalized white−gray differences instead."
        ),
    )
    args = parser.parse_args()

    out_path = generate_white_vs_gray_plots(
        args.input,
        out_dir=args.outdir,
        aggregate_tasks_dataset=not args.dataset_task_couples,
        aggregate_tasks_feature=not args.feature_task_couples,
        second_panel=args.second_panel,
        center=not args.no_centering,
    )
    print("Saved white vs gray plots to", out_path)


if __name__ == "__main__":
    main()
