from __future__ import annotations

import argparse
from typing import List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    DEFAULT_COMBOS,
    choose_fold_metric,
    clean_target,
    filter_combos,
    format_label,
    is_dummy_model,
    score_from_metric,
    select_best_runs,
    fold_columns,
    zscore,
)


def _collect_differences(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dataset, target, task), group in df.groupby(["dataset", "target_clean", "prediction_task"]):
        fold_prefix, metric_label, higher_is_better = choose_fold_metric(group, task)
        # Select best run per (model, tissue, feature)
        best = select_best_runs(group, fold_prefix, higher_is_better)
        if best.empty or "_fold_mean" not in best.columns:
            continue
        
        # Filter out dummies
        best = best[~best["model_name"].apply(is_dummy_model)]
        if best.empty:
            continue

        # fold columns
        f_cols = fold_columns(best, fold_prefix)
        if not f_cols:
            continue

        for feature, sub in best.groupby("primary_metric"):
            # Select Best White Model and Best Gray Model separately
            white_rows = sub[sub["tissue_type"] == "white"]
            gray_rows = sub[sub["tissue_type"] == "gray"]

            if white_rows.empty or gray_rows.empty:
                continue

            if higher_is_better:
                best_white = white_rows.loc[white_rows["_fold_mean"].idxmax()]
                best_gray = gray_rows.loc[gray_rows["_fold_mean"].idxmax()]
            else:
                best_white = white_rows.loc[white_rows["_fold_mean"].idxmin()]
                best_gray = gray_rows.loc[gray_rows["_fold_mean"].idxmin()]

            # Extract fold values
            w_vals = best_white[f_cols].values.astype(float)
            g_vals = best_gray[f_cols].values.astype(float)

            # Ensure same shape and no nans
            if len(w_vals) != len(g_vals) or np.isnan(w_vals).any() or np.isnan(g_vals).any():
                continue

            # Difference
            diffs = w_vals - g_vals
            
            # If lower is better (MAE), then White < Gray means White is better.
            # We want Positive to mean White is Better.
            if not higher_is_better:
                diffs = -diffs

            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1)

            if std_diff == 0:
                # If variance is 0, we can't normalize. 
                # If mean > 0, it's infinitely good. If 0, it's 0.
                if mean_diff == 0:
                    continue # No diff
                # If there is a diff but no variance (deterministic gap), 
                # we technically have infinite T. 
                # For visualization, maybe skip or set to large value?
                # This is rare in CV.
                continue
            
            # Normalize fold differences
            # normalized_diffs are the values we want to plot (Effect Size distribution)
            norm_diffs = diffs / std_diff
            t_score = mean_diff / std_diff

            for val in norm_diffs:
                rows.append({
                    "dataset": dataset,
                    "target": target,
                    "task": task,
                    "feature": feature,
                    "metric_label": metric_label,
                    "t_score_mean": t_score, # For sorting
                    "normalized_diff": val,
                })

    return pd.DataFrame(rows)


def plot_white_vs_gray_tscore(
    parquet_path: str,
    out_dir: str = "analysis_results/visualization_demo/plots/folds",
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    df["target_clean"] = df["target"].map(clean_target)
    df = filter_combos(df, DEFAULT_COMBOS)

    if df.empty:
        raise RuntimeError("No rows left after filtering combos")

    diffs = _collect_differences(df)
    if diffs.empty:
        raise RuntimeError("No valid white/gray differences found")

    # Sort tasks by mean T-score
    # We need to aggregate first to determine order
    order_df = diffs.groupby(["dataset", "target", "task", "feature"])["t_score_mean"].mean().reset_index()
    order_df = order_df.sort_values("t_score_mean", ascending=False)
    
    # Create label mapping
    order_df["label"] = [
        format_label(f"{row.dataset}|{row.target}|{row.task}|{row.feature}")
        for row in order_df.itertuples(index=False)
    ]
    
    # Map labels back to full df for plotting
    label_map = dict(zip(zip(order_df.dataset, order_df.target, order_df.task, order_df.feature), order_df.label))
    diffs["label"] = diffs.apply(lambda x: label_map.get((x["dataset"], x["target"], x["task"], x["feature"])), axis=1)
    
    # Define order for seaborn
    plot_order = order_df["label"].tolist()

    # Setup Plot
    sns.set_theme(style="whitegrid")
    fig_h = max(6, 0.4 * len(plot_order))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    # Box + Strip Plot
    # Boxplot for the distribution info
    sns.boxplot(
        data=diffs,
        x="normalized_diff",
        y="label",
        order=plot_order,
        color="#E0E0E0",
        fliersize=0, # Turn off outlier diamonds, stripplot will show them
        ax=ax,
        width=0.5
    )
    
    # Stripplot for the individual folds
    sns.stripplot(
        data=diffs,
        x="normalized_diff",
        y="label",
        order=plot_order,
        color="#4C78A8",
        size=4,
        alpha=0.8,
        ax=ax
    )

    ax.axvline(0.0, color="#333333", linewidth=1.0)
    
    ax.set_xlabel("Standardized Fold Difference (Diff / StdDev)")
    ax.set_title("White vs Gray Matter: Standardized Performance Difference (Best Models)", pad=15)
    
    # Annotate regions
    ax.text(0.02, 1.01, "Gray matter wins", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight='bold', color='gray')
    ax.text(0.98, 1.01, "White matter wins", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, fontweight='bold', color='gray')

    fig.tight_layout()
    out_file = out_path / "white_vs_gray_tscore.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot z-scored white vs gray differences per dataset/task/feature")
    parser.add_argument("--input", default="comprehensive_results.parquet", help="Input parquet file")
    parser.add_argument(
        "--outdir",
        default="analysis_results/visualization_demo/plots/folds",
        help="Output directory",
    )
    args = parser.parse_args()

    out_path = plot_white_vs_gray_tscore(args.input, args.outdir)
    print("Saved white vs gray t-score plot to", out_path)


if __name__ == "__main__":
    main()
