from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import apply_miccai_style
from utils import (
    DEFAULT_COMBOS,
    choose_fold_metric,
    clean_target,
    filter_combos,
    format_label,
    is_dummy_model,
    select_best_runs,
    fold_columns,
    calculate_paired_stats,
    calculate_paired_ttest,
    get_display_label,
)


def _collect_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Iterate over Dataset/Target/Task
    for (dataset, target, task), group in df.groupby(["dataset", "target_clean", "prediction_task"]):
        fold_prefix, metric_label, higher_is_better = choose_fold_metric(group, task)
        
        # 1. Select Best Run for every (Model, Tissue, Feature)
        best = select_best_runs(group, fold_prefix, higher_is_better)
        if best.empty or "_fold_mean" not in best.columns:
            continue
            
        # 2. Filter out dummies
        best = best[~best["model_name"].apply(is_dummy_model)]
        if best.empty:
            continue

        f_cols = fold_columns(best, fold_prefix)
        if not f_cols:
            continue

        for tissue, sub in best.groupby("tissue_type"):
            
            # Determine Winner Feature Logic:
            # 1. Gather all features and their mean scores
            feature_scores = []
            
            sub_feats = []
            for feature, sub_f in sub.groupby("primary_metric"):
                # "primary_metric" column holds the feature name (e.g. rtop, spheres) in this parquet
                
                if higher_is_better:
                    best_m = sub_f.loc[sub_f["_fold_mean"].idxmax()]
                else:
                    best_m = sub_f.loc[sub_f["_fold_mean"].idxmin()]
                
                vals = best_m[f_cols].values.astype(float)
                mean_v = np.mean(vals)
                sub_feats.append({
                    "feature": feature,
                    "vals": vals,
                    "mean": mean_v,
                    "model": best_m
                })

            if not sub_feats:
                continue

            # Sort by performance
            sub_feats.sort(key=lambda x: x["mean"], reverse=higher_is_better)
            
            # Best Feature
            best_feat = sub_feats[0]
            
            # Statistical Test vs Second Best (if exists)
            is_clear_winner = False
            if len(sub_feats) > 1:
                second_feat = sub_feats[1]
                # Compare Best vs Second
                p_val = calculate_paired_ttest(best_feat["vals"], second_feat["vals"])
                if p_val < 0.05:
                    is_clear_winner = True
            
            # Create rows for all features
            for item in sub_feats:
                # Status:
                # 2 = Best & Clear Winner
                # 1 = Best but NOT Clear Winner (Tie with 2nd)
                # 0 = Not Best
                
                status = 0
                if item["feature"] == best_feat["feature"]:
                    if is_clear_winner:
                        status = 2 # Win
                    else:
                        status = 1 # Tie/Best
                
                rows.append({
                    "dataset": dataset,
                    "target": target,
                    "task": task,
                    "tissue": tissue,
                    "feature": item["feature"],
                    "metric_label": metric_label,
                    "mean_score": item["mean"],
                    "std_score": np.std(item["vals"], ddof=1),
                    "status": status,
                })

    return pd.DataFrame(rows)


def plot_feature_heatmap(
    parquet_path: str,
    out_dir: str = "analysis_results/visualization_demo/plots/features",
) -> Path:
    apply_miccai_style()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    df["target_clean"] = df["target"].map(clean_target)
    df = filter_combos(df, DEFAULT_COMBOS)

    if df.empty:
        raise RuntimeError("No rows left after filtering combos")

    stats = _collect_feature_stats(df)
    if stats.empty:
        raise RuntimeError("No valid feature stats found")

    # Create composite column label
    # "Dataset - Target (Metric)\n(tissue)"
    stats["col_label"] = stats.apply(
        lambda x: f"{get_display_label(x.dataset, x.target, x.task, x.metric_label)}\n({x.tissue})", 
        axis=1
    )
    
    # Pivot for Status (Color) and Text (Annotation)
    pivot_status = stats.pivot(index="feature", columns="col_label", values="status")
    pivot_mean = stats.pivot(index="feature", columns="col_label", values="mean_score")
    pivot_std = stats.pivot(index="feature", columns="col_label", values="std_score")
    
    # Sort columns
    col_order = sorted(stats["col_label"].unique()) 
    pivot_status = pivot_status[col_order]
    pivot_mean = pivot_mean[col_order]
    pivot_std = pivot_std[col_order]

    # Annotations Matrix
    annot_matrix = pivot_mean.copy().astype(object)
    for c in pivot_mean.columns:
        for i in pivot_mean.index:
            m = pivot_mean.loc[i, c]
            s = pivot_std.loc[i, c]
            if pd.notna(m):
                annot_matrix.loc[i, c] = f"{m:.3f}\n±{s:.3f}"
            else:
                annot_matrix.loc[i, c] = ""

    # User Request: "put best only when there's a clear statistical win"
    # So map Status 1 (Best but not clear) to 0 (No Winner)
    plot_data = pivot_status.copy()
    plot_data = plot_data.replace({1: 0})
    # Now valid values are 0 and 2.
    
    # We want a discrete colormap.
    # 0 = White/Gray
    # 2 = Green/Gold
    
    # Map to 0 and 1 for heatmap plotting
    plot_data = plot_data.replace({2: 1})
    
    cmap = ListedColormap(["#F8F9FA", "#C8E6C9"]) # Very Light Gray, Light Green
    
    sns.set_theme(style="white")
    # Size logic
    w = max(10, 2.5 * len(pivot_mean.columns))
    h = max(4, 1.0 * len(pivot_mean.index))
    
    fig, ax = plt.subplots(figsize=(w, h))
    
    sns.heatmap(
        plot_data, 
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        cbar=True, 
        linewidths=.5,
        ax=ax,
        annot_kws={"fontsize": 9},
        vmin=0, vmax=1
    )
    
    # Fix Colorbar labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(["No Winner (Tie/Non-Sig)", "Best (Stat. Sig.)"])
    
    ax.set_title("Feature Impact (Mean ± Std)", pad=20)
    ax.set_xlabel("")
    ax.set_ylabel("Feature")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0) 
    
    fig.tight_layout()
    out_file = out_path / "feature_impact_heatmap.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot feature impact heatmap")
    parser.add_argument("--input", default="comprehensive_results.parquet", help="Input parquet file")
    parser.add_argument(
        "--outdir",
        default="analysis_results/visualization_demo/plots/features",
        help="Output directory",
    )
    args = parser.parse_args()

    out_path = plot_feature_heatmap(args.input, args.outdir)
    print("Saved feature comparison plot to", out_path)


if __name__ == "__main__":
    main()
