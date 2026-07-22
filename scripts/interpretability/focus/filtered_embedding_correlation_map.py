# Goal is to show that the results of the model predictions are meaningful
import numpy as np
import pandas as pd
import ast
from pathlib import Path
from diff_benchmark.analysis.region_coefficients import load_atlas_from_run
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "coefficients_long.parquet"
OUTPUT_DIR = PROJECT_ROOT / "exp_outputs" / "summary" / "grouped_embedding_correlation_maps" / "hcp" / "md"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# FS_LABELS_JSON = PROJECT_ROOT / "aux_materials" / "fs_labels.json"
SCHAEFER_LABELS_JSON = PROJECT_ROOT / "aux_materials" / "schaefer_labels.json"

# REGION_REPRESENTATIONS = ["flatten", "mean_std", "summary_stats", "percentiles", "pca"]
PERCENTILE = 0.90
MICROSTRUCTURE_SELECTION = "md"
DATASET_SELECTION = "hcp"
TASK_SELECTION = "binary_classification"
TOP_K = 10 
THRESHOLD = 0.50

# -------------------------
# FILTER region_permutation
# -------------------------
def exclude_region_permutation(df):
    cols = [c for c in ["model", "model_type", "model_name"] if c in df.columns]
    if not cols:
        return df
    mask = np.ones(len(df), dtype=bool)
    for c in cols:
        mask &= ~df[c].astype(str).str.contains("region_permutation", case=False, na=False)
    return df.loc[mask].copy()


# -------------------------
# microstructure filter
# -------------------------
def filter_microstructure(df):
    if "microstructure" not in df.columns:
        return df
    return df[df["microstructure"].astype(str) == MICROSTRUCTURE_SELECTION].copy()


# -------------------------
# exp id (optional but useful for grouping runs)
# -------------------------
def build_exp_id(df):
    cols = ["run_id"]
    if "fold" in df.columns:
        cols.append("fold")
    if "seed" in df.columns:
        cols.append("seed")
    return df[cols].astype(str).agg("_".join, axis=1)


# -------------------------
# percentile selection
# -------------------------
def add_percentile_selection(df):
    group_cols = [c for c in ["run_id", "fold", "seed"] if c in df.columns]
    df = df.copy()

    df["selected"] = df.groupby(group_cols)["coef"].transform(
        lambda s: (s.abs() >= s.abs().quantile(PERCENTILE)).astype(int)
    )
    return df


def _plot_frequency_rows(surface_atlas, values_by_name, title, out_file):
    names = list(values_by_name.keys())
    n_rows = len(names)
    fig = plt.figure(figsize=(10, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, 3, width_ratios=[1, 1, 0.06], wspace=0.02, hspace=0.12)
    for i, name in enumerate(names):
        ax_left = fig.add_subplot(gs[i, 0], projection="3d")
        ax_right = fig.add_subplot(gs[i, 1], projection="3d")
        _plot_surface_row(surface_atlas, values_by_name[name], ax_left, ax_right, 0, 1, name, "Reds")

    cax = fig.add_subplot(gs[:, 2])
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cax)

    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(left=0.03, right=0.94, top=0.94, bottom=0.03)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


# -------------------------
# model / embedding parsing (minimal version)
# -------------------------
def safe_parse_embedding(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def extract_embedding(row):
    emb_raw = row.get("embedding", "")
    emb_dict = safe_parse_embedding(emb_raw)

    if "region_representation" in emb_dict:
        return emb_dict["region_representation"]

    model_type = str(row.get("model_type", "")).lower()
    estimator = str(row.get("estimator", "")).lower()
    model_name = str(row.get("model_name", "")).lower()

    # pointnet runs currently appear as "unknown"
    if "pointnet" in model_type or "pointnet" in estimator or "pointnet" in model_name:
        return "pointnet"

    if model_type == "region_pca":
        return "pca"

    return "unknown"


def add_embedding_name(df):
    df = df.copy()
    df["embedding_name"] = df.apply(extract_embedding, axis=1)
    # return df
    return df[df["embedding_name"] != "unknown"].copy()


# -------------------------
# frequency maps
# -------------------------
def compute_frequency_maps(df):
    # dataset/task/embedding/region → frequency
    stats = df.groupby(["dataset", "task", "embedding_name", "region"])["selected"].mean()
    return stats.reset_index(name="selection_freq")


# -------------------------
# FULL PIPELINE
# -------------------------
def build_frequency_maps(df):
    df = exclude_region_permutation(df)
    df = filter_microstructure(df)

    df["exp_id"] = build_exp_id(df)
    df = add_embedding_name(df)
    df = add_percentile_selection(df)

    # -------------------------
    # Embedding frequency maps
    # -------------------------
    embedding_freq = (
        df.groupby(["dataset", "task", "embedding_name", "region"])["selected"]
        .mean()
        .reset_index(name="selection_freq")
    )

    # -------------------------
    # Global frequency map
    # -------------------------
    global_freq = (
        df.groupby(["dataset", "task", "region"])["selected"]
        .mean()
        .reset_index(name="global_selection_freq")
    )

    return embedding_freq, global_freq


def build_frequency_maps_from_selected(df):
    embedding_freq = (
        df.groupby(["dataset", "task", "embedding_name", "region"])["selected"]
        .mean()
        .reset_index(name="selection_freq")
    )

    global_freq = (
        df.groupby(["dataset", "task", "region"])["selected"]
        .mean()
        .reset_index(name="global_selection_freq")
    )

    return embedding_freq, global_freq

def build_estimator_frequency_maps(df):
    estimator_col = _get_estimator_col(df)

    tmp = df.copy()
    tmp["estimator_name"] = (
        tmp[estimator_col]
        .apply(normalize_estimator_name)
    )

    estimator_freq = (
        tmp.groupby(
            ["dataset", "task", "estimator_name", "region"]
        )["selected"]
        .mean()
        .reset_index(name="selection_freq")
    )

    return estimator_freq

def region_embedding_matrix(embedding_freq:pd.DataFrame) -> pd.DataFrame:
    embedding_matrix = embedding_freq.pivot_table(
        index="region",
        columns="embedding_name",
        values="selection_freq",
        aggfunc="mean"
    ).fillna(0.0)
    return embedding_matrix

import matplotlib.pyplot as plt

def plot_corr_matrix(corr_matrix, title, out_file):
    plt.figure(figsize=(6, 5))
    if corr_matrix.values.min() < 0:
        vmin, vmax = -1, 1
        plt.imshow(corr_matrix.values.astype(float), vmin=vmin, vmax=vmax, cmap="coolwarm")
    else:
        vmin, vmax = 0, 1
        plt.imshow(corr_matrix.values.astype(float), vmin=vmin, vmax=vmax, cmap="Reds")
    

    plt.colorbar(label="Pearson r")

    plt.xticks(
        range(len(corr_matrix.columns)),
        corr_matrix.columns,
        rotation=45,
        ha="right"
    )
    plt.yticks(
        range(len(corr_matrix.index)),
        corr_matrix.index
    )

    plt.title(title)
    plt.tight_layout()

    plt.savefig(out_file, dpi=150)
    plt.close()


def _is_lower_better(task: str) -> bool:
    task_lower = str(task).lower()
    return "age" in task_lower or "regression" in task_lower


def _display_task_label(dataset: str, task: str) -> str:
    dataset_lower = str(dataset).lower()
    task_lower = str(task).lower()
    if "regression" in task_lower:
        return "age"
    if task_lower == "binary_classification":
        if dataset_lower in {"hcp", "camcan"}:
            return "sex"
        if "abide" in dataset_lower:
            return "ASD"
    return str(task)


def _clip_violin_half(body, side: str) -> None:
    path = body.get_paths()[0]
    vertices = path.vertices
    center_x = np.mean(vertices[:, 0])
    if side == "left":
        vertices[:, 0] = np.minimum(vertices[:, 0], center_x)
    else:
        vertices[:, 0] = np.maximum(vertices[:, 0], center_x)


def plot_score_distribution_split(df, dataset, task, out_file):
    if "test_score" not in df.columns:
        return

    run_cols = [c for c in ["run_id", "fold", "seed"] if c in df.columns]
    group_cols = ["dataset", "task"] + run_cols
    run_df = df.groupby(group_cols)["test_score"].mean().reset_index()
    run_df = run_df[(run_df["dataset"] == dataset) & (run_df["task"] == task)].copy()
    if run_df.empty:
        return

    run_df = run_df[~np.isnan(run_df["test_score"])].copy()
    scores = run_df["test_score"].to_numpy(dtype=float)
    if scores.size == 0:
        return

    if _is_lower_better(task):
        threshold = np.quantile(scores, 0.75)
        keep_mask = run_df["test_score"] <= threshold
        score_label = "MAE"
    else:
        threshold = np.quantile(scores, 0.25)
        keep_mask = run_df["test_score"] >= threshold
        score_label = "Balanced Accuracy"

    kept_scores = run_df.loc[keep_mask, "test_score"].to_numpy(dtype=float)
    if kept_scores.size == 0:
        kept_scores = scores

    # ── Palette & style ──────────────────────────────────────────────────────
    COLOR_ALL      = "#7EB8D4"   # muted steel blue  — all runs
    COLOR_FILTERED = "#E8825A"   # warm coral        — filtered runs
    COLOR_MEAN_ALL = "#2A6D8F"
    COLOR_MEAN_FLT = "#C04F28"
    COLOR_BG       = "#F7F7F5"
    COLOR_GRID     = "#E0DED9"
    ALPHA_VIOLIN   = 0.82

    sns.set_style("white")
    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Georgia", "DejaVu Serif"],
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.spines.left": True,
        "axes.spines.bottom": False,
    })

    fig, ax = plt.subplots(figsize=(5.5, 6))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    pos = 0
    half_w = 0.38   # half-violin display width

    # ── Draw violins ─────────────────────────────────────────────────────────
    vp_all = ax.violinplot(
        [scores], positions=[pos], widths=0.76,
        showmeans=False, showmedians=False, showextrema=False,
    )
    vp_flt = ax.violinplot(
        [kept_scores], positions=[pos], widths=0.76,
        showmeans=False, showmedians=False, showextrema=False,
    )

    body_all = vp_all["bodies"][0]
    body_flt = vp_flt["bodies"][0]

    _clip_violin_half(body_all, "left")
    _clip_violin_half(body_flt, "right")

    body_all.set_facecolor(COLOR_ALL)
    body_all.set_edgecolor("white")
    body_all.set_linewidth(1.2)
    body_all.set_alpha(ALPHA_VIOLIN)

    body_flt.set_facecolor(COLOR_FILTERED)
    body_flt.set_edgecolor("white")
    body_flt.set_linewidth(1.2)
    body_flt.set_alpha(ALPHA_VIOLIN)

    # ── Median lines (IQR bar) ────────────────────────────────────────────────
    for side_scores, side, color in [
        (scores,       "left",  COLOR_ALL),
        (kept_scores,  "right", COLOR_FILTERED),
    ]:
        q25, q50, q75 = np.percentile(side_scores, [25, 50, 75])
        offset = -0.04 if side == "left" else 0.04
        ax.vlines(pos + offset, q25, q75, color=color, linewidth=2.5, zorder=4)
        ax.scatter(pos + offset, q50, color="white", s=28, zorder=5, linewidth=0)

    # ── Mean markers & annotations ────────────────────────────────────────────
    mean_all = np.mean(scores)
    mean_flt = np.mean(kept_scores)

    MEAN_X_LEFT  = pos - 0.18
    MEAN_X_RIGHT = pos + 0.18

    ax.scatter(MEAN_X_LEFT,  mean_all, color=COLOR_MEAN_ALL,
               s=60, zorder=6, marker="D", linewidths=0)
    ax.scatter(MEAN_X_RIGHT, mean_flt, color=COLOR_MEAN_FLT,
               s=60, zorder=6, marker="D", linewidths=0)

    fmt = ".4f" if score_label == "MAE" else ".3f"
    ax.annotate(
        f"μ = {mean_all:{fmt}}",
        xy=(MEAN_X_LEFT, mean_all),
        xytext=(-38, 0), textcoords="offset points",
        ha="right", va="center",
        fontsize=8.5, color=COLOR_MEAN_ALL,
        fontfamily="monospace",
        arrowprops=dict(arrowstyle="-", color=COLOR_MEAN_ALL,
                        lw=0.8, connectionstyle="arc3,rad=0"),
    )
    ax.annotate(
        f"μ = {mean_flt:{fmt}}",
        xy=(MEAN_X_RIGHT, mean_flt),
        xytext=(38, 0), textcoords="offset points",
        ha="left", va="center",
        fontsize=8.5, color=COLOR_MEAN_FLT,
        fontfamily="monospace",
        arrowprops=dict(arrowstyle="-", color=COLOR_MEAN_FLT,
                        lw=0.8, connectionstyle="arc3,rad=0"),
    )

    # ── Threshold line ────────────────────────────────────────────────────────
    ax.hlines(threshold, pos - half_w, pos + half_w,
              colors="#444444", linestyles=(0, (4, 3)), linewidth=1.2, zorder=3)
    ax.text(pos + half_w + 0.02, threshold,
            f"  threshold\n  {threshold:{fmt}}",
            va="center", ha="left", fontsize=7.5, color="#444444",
            linespacing=1.5)

    # ── Axes & labels ─────────────────────────────────────────────────────────
    task_label = _display_task_label(dataset, task)
    ax.set_xticks([pos])
    ax.set_xticklabels([f"{dataset}  ·  {task_label}"],
                       fontsize=10, color="#333333")
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylabel(score_label, fontsize=10, color="#333333", labelpad=8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
    ax.tick_params(axis="y", labelsize=8.5, color=COLOR_GRID)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(
        "Score distribution: all runs vs. filtered",
        fontsize=11, fontweight="normal", color="#222222",
        pad=14, loc="left",
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_ALL,      label="All runs",      alpha=ALPHA_VIOLIN),
        mpatches.Patch(facecolor=COLOR_FILTERED,  label="Filtered runs", alpha=ALPHA_VIOLIN),
    ]
    ax.legend(handles=legend_handles, frameon=False,
              loc="upper right", fontsize=8.5,
              labelcolor="#333333")

    # ── n-count labels sit below the plot ─────────────────────────────────────
    y_min, y_max = ax.get_ylim()
    y_text = y_min - 0.05 * (y_max - y_min)
    ax.set_ylim(y_text, y_max)
    for xpos, n, color in [
        (pos - 0.22, len(scores),      COLOR_ALL),
        (pos + 0.22, len(kept_scores), COLOR_FILTERED),
    ]:
        ax.text(
            xpos,
            y_text,
            f"n = {n}",
            ha="center",
            va="top",
            fontsize=7.5,
            color=color,
            style="italic",
        )

    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight",
                facecolor=COLOR_BG)
    plt.close(fig)


def _get_estimator_col(df: pd.DataFrame) -> str | None:
    for col in ["estimator", "model_name", "model_type", "model"]:
        if col in df.columns:
            return col
    return None

def normalize_estimator_name(x):
    x = str(x).lower()

    if x in {"region_pca", "pointnet"}:
        return "region_group_lasso"

    return x

def plot_pipeline_factor_boxplots(df, dataset, task, out_file):
    if "test_score" not in df.columns:
        return

    estimator_col = _get_estimator_col(df)
    if estimator_col is None:
        return

    df = df[(df["dataset"] == dataset) & (df["task"] == task)].copy()
    if df.empty:
        return

    if "embedding_name" not in df.columns:
        df = add_embedding_name(df)
    
    # remove unknown embeddings from the plotting dataframe
    df = df[df["embedding_name"] != "unknown"].copy()

    if "microstructure" not in df.columns:
        df["microstructure"] = "unknown"

    run_cols = [c for c in ["run_id", "fold", "seed"] if c in df.columns]
    group_cols = [
        "dataset",
        "task",
        "microstructure",
        "embedding_name",
        estimator_col,
    ] + run_cols

    run_df = df.groupby(group_cols)["test_score"].mean().reset_index()
    run_df[estimator_col] = run_df[estimator_col].apply(normalize_estimator_name)
    scores = run_df["test_score"].to_numpy(dtype=float)
    scores = scores[~np.isnan(scores)]
    if scores.size == 0:
        return

    if _is_lower_better(task):
        threshold = np.quantile(scores, 0.75)
        keep_mask = run_df["test_score"] <= threshold
        score_label = "MAE"
    else:
        threshold = np.quantile(scores, 0.25)
        keep_mask = run_df["test_score"] >= threshold
        score_label = "Balanced Accuracy"

    df_all = run_df.copy()
    df_all["filter_status"] = "All"
    df_filtered = run_df[keep_mask].copy()
    df_filtered["filter_status"] = "Filtered"

    df_plot = pd.concat([df_all, df_filtered], ignore_index=True)

    factors = [
        ("Microstructure", "microstructure"),
        ("Embedding", "embedding_name"),
        ("Estimator", estimator_col),
    ]

    palette = {
        "All": "#7EB8D4",
        "Filtered": "#E8825A",
    }

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(factors), figsize=(6 * len(factors), 5), sharey=True)
    if len(factors) == 1:
        axes = [axes]

    for ax, (title, col) in zip(axes, factors):
        sns.violinplot(
            data=df_plot,
            x=col,
            y="test_score",
            hue="filter_status",
            palette=palette,
            hue_order=["All", "Filtered"],
            split=True,
            inner="quartile",
            cut=0,
            scale="width",
            ax=ax,
        )
        ax.axhline(threshold, color="#444444", linestyle=(0, (4, 3)), linewidth=1.0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        if ax is axes[0]:
            ax.set_ylabel(score_label)
        else:
            ax.set_ylabel("")
        ax.legend_.remove()

        y_min, y_max = ax.get_ylim()
        y_text = y_min - 0.06 * (y_max - y_min)
        ax.set_ylim(y_text, y_max)
        counts_all = run_df.groupby(col)["test_score"].count()
        counts_flt = run_df[keep_mask].groupby(col)["test_score"].count()
        xticks = ax.get_xticks()
        labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
        for xpos, label in zip(xticks, labels):
            n_all = int(counts_all.get(label, 0))
            n_flt = int(counts_flt.get(label, 0))
            ax.text(
                xpos,
                y_text,
                f"n={n_all}/{n_flt}",
                ha="center",
                va="top",
                fontsize=7.0,
                color="#555555",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(
        f"Performance by pipeline factors | {dataset}::{_display_task_label(dataset, task)}",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_region_rank_from_network(global_freq, dataset, task, out_file):
    sub_df = global_freq[
        (global_freq["dataset"] == dataset) &
        (global_freq["task"] == task)
    ].copy()
    if sub_df.empty:
        return

    sub_df = sub_df.sort_values("global_selection_freq", ascending=False)
    values = sub_df["global_selection_freq"].to_numpy(dtype=float)
    ranks = np.arange(1, len(values) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(ranks, values, marker="o", markersize=2, linewidth=1)
    plt.ylim(0, 1)
    plt.xlabel("Region rank (most to least selected)")
    plt.ylabel("Selection frequency")
    plt.title(f"Network-based selection frequency by region | {dataset} | {task}")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()

def plot_network_frequency_barplot(
    network_summary,
    dataset,
    task,
    out_file,
):
    plt.figure(figsize=(8, 6))

    plot_df = network_summary.copy()

    plot_df["label"] = (
        plot_df["network_name"]
        + " ("
        + plot_df["n_regions"].astype(str)
        + "/"
        + "100" #plot_df["total_regions"].astype(str)
        + ")"
    )

    sns.barplot(
        data=plot_df,
        y="label",
        x="mean_freq",
        color="#C44E52",
    )

    plt.xlim(0, 1)

    plt.xlabel("Mean selection frequency")
    plt.ylabel("")

    plt.title(
        f"Network selection frequency | {dataset} | {task}"
    )

    # write value at end of bar
    for i, value in enumerate(plot_df["mean_freq"]):
        plt.text(
            value + 0.01,
            i,
            f"{value:.2f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()

def plot_network_frequency_hierarchy(
    network_summary,
    subnetwork_summary,
    dataset,
    task,
    out_file,
):
    rows = []

    for _, net_row in network_summary.iterrows():

        network = net_row["network_name"]

        rows.append(
            {
                "label": (
                    f"{network} "
                    f"({net_row['n_regions']}/"
                    f"{net_row['total_regions']})"
                ),
                "value": net_row["mean_freq"],
                "level": "network",
            }
        )

        subnets = subnetwork_summary[
            subnetwork_summary["network"] == network
        ].sort_values(
            "mean_freq",
            ascending=False,
        )

        for _, sub_row in subnets.iterrows():

            rows.append(
                {
                    "label": (
                        f"   ↳ {sub_row['subnetwork']} "
                        f"({sub_row['n_regions']})"
                    ),
                    "value": sub_row["mean_freq"],
                    "level": "subnetwork",
                }
            )

    plot_df = pd.DataFrame(rows)

    plt.figure(
        figsize=(9, 0.45 * len(plot_df))
    )

    # colors = plot_df["level"].map(
    #     {
    #         "network": "#C44E52",
    #         "subnetwork": "#4C72B0",
    #     }
    # )
    palette = {
        "network": "#C44E52",
        "subnetwork": "#4C72B0",
    }

    # plt.barh(
    #     plot_df["label"],
    #     plot_df["value"],
    #     color=colors,
    # )
    for i, row in enumerate(plot_df.itertuples()):

        height = 0.8 if row.level == "network" else 0.45

        plt.barh(
            y=i,
            width=row.value,
            height=height,
            color=palette[row.level],
        )
        plt.yticks(
            range(len(plot_df)),
            plot_df["label"],
        )
        
        plt.text(
            row.value + 0.015,
            i,
            f"{row.value:.2f}",
            va="center",
            fontsize=8,
        )

    plt.xlim(0, 1)

    plt.xlabel("Mean selection frequency")

    plt.title(
        f"Network hierarchy | {dataset} | {task}"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(out_file, dpi=200)
    plt.close()

def embedding_pair_surface_map(embedding_matrix, emb_a, emb_b):
    a = embedding_matrix[emb_a]
    b = embedding_matrix[emb_b]

    # standardize per embedding (important)
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)

    # per-region agreement map
    return a * b

def to_label_map(series):
    return {
        int(str(r).split(":")[-1]): float(v)
        for r, v in series.items()
    }

def _surface_texture_from_label_map(surface_atlas, label_values):
	labels = np.asarray(surface_atlas["parcel_labels"]).astype(int)
	n_left = int(surface_atlas["n_left_vertices"])

	texture = np.zeros(len(labels))
	for label, value in label_values.items():
		texture[labels == label] = value

	return texture[:n_left], texture[n_left:]

def _plot_surface_row(surface_atlas, label_values, ax_left, ax_right, vmin, vmax, title: str, cmap: str) -> None:
	from nilearn import plotting

	left_mesh = surface_atlas["left_mesh"]
	right_mesh = surface_atlas["right_mesh"]
	tex_left, tex_right = _surface_texture_from_label_map(surface_atlas, label_values)

	plotting.plot_surf_stat_map(
		left_mesh,
		tex_left,
		hemi="left",
		axes=ax_left,
		vmin=vmin,
		vmax=vmax,
		cmap=cmap,
		colorbar=False,
	)
	plotting.plot_surf_stat_map(
		right_mesh,
		tex_right,
		hemi="right",
		axes=ax_right,
		vmin=vmin,
		vmax=vmax,
		cmap=cmap,
		colorbar=False,
	)
	if title:
		ax_left.set_title(title, fontsize=14)

def _plot_single_surface(surface_atlas, label_values, ax, hemi, view, vmin, vmax, cmap):
    from nilearn import plotting

    mesh = surface_atlas["left_mesh"] if hemi == "left" else surface_atlas["right_mesh"]
    tex_left, tex_right = _surface_texture_from_label_map(surface_atlas, label_values)
    tex = tex_left if hemi == "left" else tex_right

    plotting.plot_surf_stat_map(
        mesh,
        tex,
        hemi=hemi,
        view=view,
        axes=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        colorbar=False,
    )


def plot_brain_4view(surface_atlas, global_vector, title, out_file):
    fig = plt.figure(figsize=(22, 5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.02)

    ax_left_lat  = fig.add_subplot(gs[0, 0], projection="3d")
    ax_left_med  = fig.add_subplot(gs[0, 1], projection="3d")
    ax_right_lat = fig.add_subplot(gs[0, 2], projection="3d")
    ax_right_med = fig.add_subplot(gs[0, 3], projection="3d")
    cax          = fig.add_subplot(gs[0, 4])

    label_map = to_label_map(global_vector)

    _plot_single_surface(surface_atlas, label_map, ax_left_lat,  hemi="left",  view="lateral", vmin=0, vmax=1, cmap="Reds")
    _plot_single_surface(surface_atlas, label_map, ax_left_med,  hemi="left",  view="medial",  vmin=0, vmax=1, cmap="Reds")
    _plot_single_surface(surface_atlas, label_map, ax_right_lat, hemi="right", view="lateral", vmin=0, vmax=1, cmap="Reds")
    _plot_single_surface(surface_atlas, label_map, ax_right_med, hemi="right", view="medial",  vmin=0, vmax=1, cmap="Reds")

    ax_left_lat .set_title("Left – lateral",  fontsize=14)
    ax_left_med .set_title("Left – medial",   fontsize=14)
    ax_right_lat.set_title("Right – lateral", fontsize=14)
    ax_right_med.set_title("Right – medial",  fontsize=14)

    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, cax=cax)

    fig.suptitle(title, fontsize=16, y=1.02)

    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()

def _corr_per_region(a_df: pd.DataFrame, b_df: pd.DataFrame) -> pd.Series:
    regions = sorted(set(a_df.columns) | set(b_df.columns))
    out = {}
    for region in regions:
        if region not in a_df.columns or region not in b_df.columns:
            out[region] = 0.0
            continue
        a = a_df[region]
        b = b_df[region]
        joined = pd.concat([a, b], axis=1, join="inner").dropna()
        if len(joined) < 2:
            out[region] = 0.0
            continue
        if float(joined.iloc[:, 0].std()) == 0.0 or float(joined.iloc[:, 1].std()) == 0.0:
            out[region] = 0.0
            continue
        out[region] = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
    return pd.Series(out)

def inspect_pair(embedding_matrix, emb_a, emb_b, top_k=10):
    
    a = embedding_matrix[emb_a]
    b = embedding_matrix[emb_b]
    
    df = pd.DataFrame({
        "freq_a": a,
        "freq_b": b,
        "abs_diff": (a - b).abs(),
        "rel_diff_max": (a - b).abs() / (np.maximum(a, b) + 1e-8),
    })

    # sort by biggest difference
    df_sorted = df.sort_values("rel_diff_max", ascending=False)
    print(f"\n=== {emb_a} vs {emb_b} ===")
    print(df_sorted.head(top_k))

    return df_sorted

def compute_binary_accuracy_matrix(binary_matrix: pd.DataFrame) -> pd.DataFrame:
    embeddings = binary_matrix.columns
    acc_matrix = pd.DataFrame(index=embeddings, columns=embeddings, dtype=float)

    for a in embeddings:
        for b in embeddings:
            x = binary_matrix[a]
            y = binary_matrix[b]

            # exact agreement
            acc = (x == y).mean()
            acc_matrix.loc[a, b] = acc

    return acc_matrix

def build_binary_tensor(df):
    return df.pivot_table(
        index=["dataset", "task", "exp_id"],   # run-level
        columns=["embedding_name", "region"],
        values="selected",
        aggfunc="mean"   # safe because selected is binary per run
    ).fillna(0.0)
    
def compute_accuracy_from_selected(df):
    embeddings = df["embedding_name"].unique()
    acc = pd.DataFrame(index=embeddings, columns=embeddings, dtype=float)

    for emb_a in embeddings:
        for emb_b in embeddings:

            sub_a = df[df["embedding_name"] == emb_a]
            sub_b = df[df["embedding_name"] == emb_b]

            merged = pd.merge(
                sub_a,
                sub_b,
                on=["dataset", "task", "region", "exp_id"],
                suffixes=("_a", "_b")
            )

            if len(merged) == 0:
                acc.loc[emb_a, emb_b] = np.nan
                continue

            agreement = (merged["selected_a"] == merged["selected_b"]).mean()
            acc.loc[emb_a, emb_b] = agreement

    return acc

def compute_accuracy_subset(df, region_subset):
    embeddings = df["embedding_name"].unique()
    acc = pd.DataFrame(index=embeddings, columns=embeddings, dtype=float)

    df = df[df["region"].isin(region_subset)]

    for emb_a in embeddings:
        for emb_b in embeddings:

            sub_a = df[df["embedding_name"] == emb_a]
            sub_b = df[df["embedding_name"] == emb_b]

            merged = pd.merge(
                sub_a,
                sub_b,
                on=["dataset", "task", "region", "exp_id"],
                suffixes=("_a", "_b")
            )

            if len(merged) == 0:
                acc.loc[emb_a, emb_b] = np.nan
                continue

            acc.loc[emb_a, emb_b] = (
                (merged["selected_a"] == merged["selected_b"]).mean()
            )

    return acc

def compute_binary_jaccard_matrix_og(binary_matrix: pd.DataFrame) -> pd.DataFrame:
    embeddings = binary_matrix.columns
    acc_matrix = pd.DataFrame(index=embeddings, columns=embeddings, dtype=float)

    for a in embeddings:
        for b in embeddings:

            x = binary_matrix[a]
            y = binary_matrix[b]

            intersection = ((x == 1) & (y == 1)).sum()
            union = ((x == 1) | (y == 1)).sum()

            acc_matrix.loc[a, b] = intersection / (union + 1e-8)

    return acc_matrix

def compute_binary_jaccard_matrix(binary_matrix: pd.DataFrame, mode="jaccard") -> pd.DataFrame:
    embeddings = binary_matrix.columns
    out = pd.DataFrame(index=embeddings, columns=embeddings, dtype=float)

    for a in embeddings:
        for b in embeddings:

            x = binary_matrix[a]
            y = binary_matrix[b]

            n11 = ((x == 1) & (y == 1)).sum()
            n00 = ((x == 0) & (y == 0)).sum()
            n10 = ((x == 1) & (y == 0)).sum()
            n01 = ((x == 0) & (y == 1)).sum()

            if mode == "jaccard":
                out.loc[a, b] = n11 / (n11 + n10 + n01 + 1e-8)

            elif mode == "anti":
                # conditional 0-0 agreement (recommended version)
                out.loc[a, b] = n00 / (n00 + n10 + n01 + 1e-8)

            else:
                raise ValueError("mode must be 'jaccard' or 'anti'")

    return out

def compute_binary_balanced_accuracy_matrix(binary_matrix: pd.DataFrame) -> pd.DataFrame:
    embeddings = binary_matrix.columns
    out = pd.DataFrame(index=embeddings, columns=embeddings, dtype=float)

    for a in embeddings:
        for b in embeddings:

            x = binary_matrix[a]
            y = binary_matrix[b]

            tp = ((x == 1) & (y == 1)).sum()
            tn = ((x == 0) & (y == 0)).sum()
            fp = ((x == 1) & (y == 0)).sum()
            fn = ((x == 0) & (y == 1)).sum()

            tpr = tp / (tp + fn + 1e-8)   # recall on positives
            tnr = tn / (tn + fp + 1e-8)   # recall on negatives

            out.loc[a, b] = 0.5 * (tpr + tnr)

    return out

def plot_brain_and_heatmap(surface_atlas, global_vector, heatmap_df, title, out_file):
    import matplotlib.pyplot as plt
    from nilearn import plotting

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.4], wspace=0.08)

    # -------------------------
    # LEFT: brain map
    # -------------------------
    ax_left = fig.add_subplot(gs[0, 0], projection="3d")
    ax_right = fig.add_subplot(gs[0, 1], projection="3d")
    ax_heat = fig.add_subplot(gs[0, 2])

    label_map = to_label_map(global_vector)

    _plot_surface_row(
        surface_atlas,
        label_map,
        ax_left,
        ax_right,
        vmin=0,
        vmax=1,
        title="Global selection frequency",
        cmap="Reds"
    )

    # -------------------------
    # RIGHT: heatmap
    # -------------------------
    # label_map = _load_label_name_map()
    label_map = load_schaefer_label_map(surface_atlas)

    heatmap_df_named = heatmap_df.copy()
    subnetwork_map = load_subnetwork_map()

    heatmap_df_named.index = [
        format_region_name(
            parse_schaefer_label(
                region_id_to_name(r, label_map),
                subnetwork_map
            ),
            max_words_per_line=3   # slightly larger chunks works better here
        )
        for r in heatmap_df.index
    ]
    
    ax = ax_heat

    # data = heatmap_df.values.astype(float)
    data = heatmap_df_named.values.astype(float)

    im = ax.imshow(
        data,
        vmin=0,
        vmax=1,
        cmap="Reds",
        interpolation="nearest",
        aspect="auto",
    )

    # ticks
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right")

    ax.set_yticks(range(len(heatmap_df.index)))
    # ax.set_yticklabels(heatmap_df.index)
    # ax.set_yticklabels(heatmap_df_named.index, fontsize=8)
    ax.set_yticklabels(
        heatmap_df_named.index,
        fontsize=7,
        linespacing=1.2   # ← key for multi-line readability
    )

    # annotations (percentages)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(
                j, i,
                f"{val*100:.0f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=8
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Selection frequency")

    ax.set_title("Top regions across embeddings")

    fig.suptitle(title)
    # plt.tight_layout()
    fig.subplots_adjust(
        left=0.04,   # space for brain
        right=0.92,  # prevents colorbar cutoff
        top=0.92,
        bottom=0.21  # space for x labels
    )
    # fig.subplots_adjust(left=0.03, right=0.98)

    plt.savefig(out_file, dpi=150)
    plt.close()
    
def format_region_name(name: str, max_words_per_line: int = 2) -> str:
    words = name.replace("_", " ").split()

    lines = []
    for i in range(0, len(words), max_words_per_line):
        lines.append(" ".join(words[i:i + max_words_per_line]))

    return "\n".join(lines)

def clean_schaefer_name(name):
    return (
        name
        .replace("17Networks_", "")
        .replace("LH_", "L ")
        .replace("RH_", "R ")
        .replace("_", " ")
    )

def load_subnetwork_map():
    if not SCHAEFER_LABELS_JSON.exists():
        return {}

    try:
        return json.loads(SCHAEFER_LABELS_JSON.read_text())
    except Exception:
        return {}
    
def parse_schaefer_label(raw_name: str, subnetwork_map: dict) -> str:
    """
    Example:
    LH_DefaultA_PFCdPFCm_1

    → 
    L Default A
    Dorsal + medial prefrontal cortex
    """
    name = raw_name.replace("17Networks_", "")
    parts = name.split("_")

    # -------------------------
    # Hemisphere
    # -------------------------
    hemi = "L" if parts[0] == "LH" else "R"

    # -------------------------
    # Network + subdivision (A/B/C)
    # -------------------------
    network_full = parts[1]  # e.g. DefaultA

    network = network_full[:-1] if network_full[-1] in "ABC" else network_full
    subpart = network_full[-1] if network_full[-1] in "ABC" else ""

    # -------------------------
    # Subnetwork key (KEEP A/B/C removed for mapping)
    # -------------------------
    sub_key = "_".join(parts[1:-1])  # DefaultA_PFCdPFCm

    # IMPORTANT:
    # remove A/B/C ONLY for lookup
    sub_key_lookup = sub_key.replace("DefaultA", "Default") \
                           .replace("DefaultB", "Default") \
                           .replace("DefaultC", "Default")

    region_name = subnetwork_map.get(sub_key_lookup, sub_key_lookup)

    # -------------------------
    # Final label
    # -------------------------
    if subpart:
        header = f"{hemi} {network} ({subpart})"
    else:
        header = f"{hemi} {network}"

    return f"{header}\n{region_name}"

def load_schaefer_label_map(surface_atlas):
    import pandas as pd
    import numpy as np

    tsv_path = surface_atlas["atlas_meta"]["label_tsv_path"]
    df = pd.read_csv(tsv_path, sep="\t")

    # remove background / medial wall
    df = df[~df["name"].str.contains("Background", na=False)].copy()

    # extract actual parcel IDs from surface atlas
    parcel_ids = np.unique(surface_atlas["parcel_labels"])
    parcel_ids = np.sort(parcel_ids)

    # remove 0 if present (background)
    parcel_ids = parcel_ids[parcel_ids != 0]

    # sanity check
    if len(parcel_ids) != len(df):
        print("WARNING: mismatch between atlas parcels and TSV labels")

    # map parcel ID → name
    return {
        int(pid): str(name)
        for pid, name in zip(parcel_ids, df["name"])
    }

def region_id_to_name(region_index, label_map):
    try:
        rid = int(str(region_index).split(":")[-1])
        return label_map.get(rid, f"region_{rid}")
    except Exception:
        return str(region_index)


def _region_id_from_region_value(region_value):
    try:
        return int(str(region_value).split(":")[-1])
    except Exception:
        return None


# def _network_from_label_name(label_name: str) -> str:
#     name = label_name.replace("17Networks_", "")
#     parts = name.split("_")
#     if len(parts) < 2:
#         return "Unknown"
#     network_full = parts[1]
#     return network_full[:-1] if network_full[-1] in "ABC" else network_full

def _network_from_label_name(label_name: str) -> str:
    name = label_name.replace("17Networks_", "")
    parts = name.split("_")

    if len(parts) < 2:
        return "Unknown"

    network = parts[1]

    # Remove A/B/C suffixes
    if network.endswith(("A", "B", "C")):
        network = network[:-1]

    # Merge Schaefer subdivisions into canonical Yeo networks
    if network in {"VisCent", "VisPeri"}:
        return "Vis"

    if network == "TempPar":
        return "SalVentAttn"

    return network

# def _subnetwork_from_label_name(label_name: str):
#     name = label_name.replace("17Networks_", "")
#     parts = name.split("_")

#     if len(parts) < 3:
#         return "Unknown"

#     network = parts[1]
#     subnetwork = "_".join(parts[1:-1])

#     return network, subnetwork

def _subnetwork_from_label_name(label_name):

    name = label_name.replace("17Networks_", "")
    parts = name.split("_")

    if len(parts) < 2:
        return "Unknown", "Unknown"

    network_full = parts[1]

    if network_full[-1] in "ABC":
        network = network_full[:-1]
        subnetwork = network_full[-1]
    else:
        network = network_full
        subnetwork = ""

    # Merge Schaefer subdivisions
    if network in {"VisCent", "VisPeri"}:
        network = "Vis"

    if network == "TempPar":
        network = "SalVentAttn"

    return network, subnetwork

def plot_brain_only(surface_atlas, global_vector, title, out_file):
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.06])

    ax_left = fig.add_subplot(gs[0, 0], projection="3d")
    ax_right = fig.add_subplot(gs[0, 1], projection="3d")
    cax = fig.add_subplot(gs[0, 2])

    label_map = to_label_map(global_vector)

    _plot_surface_row(
        surface_atlas,
        label_map,
        ax_left,
        ax_right,
        vmin=0,
        vmax=1,
        title=None, #"Global selection frequency",
        cmap="Reds",
    )

    sm = plt.cm.ScalarMappable(
        cmap="Reds",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    fig.colorbar(sm, cax=cax)

    fig.suptitle(title)
    fig.tight_layout()

    plt.savefig(out_file, dpi=200)
    plt.close()
    
def plot_heatmap_only(
    surface_atlas,
    heatmap_df,
    title,
    out_file,
):
    label_map = load_schaefer_label_map(surface_atlas)
    subnetwork_map = load_subnetwork_map()

    heatmap_df_named = heatmap_df.copy()

    heatmap_df_named.index = [
        format_region_name(
            parse_schaefer_label(
                region_id_to_name(r, label_map),
                subnetwork_map
            ),
            max_words_per_line=3
        )
        for r in heatmap_df.index
    ]

    plt.figure(figsize=(8, max(5, len(heatmap_df_named) * 0.6)))

    sns.heatmap(
        heatmap_df_named,
        cmap="Reds",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".0%",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Selection frequency"},
    )

    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(out_file, dpi=200)
    plt.close()


def build_region_to_network(surface_atlas) -> dict[int, str]:
    label_map = load_schaefer_label_map(surface_atlas)
    return {rid: _network_from_label_name(label) for rid, label in label_map.items()}

def compute_network_summary(global_freq, surface_atlas):
    label_map = load_schaefer_label_map(surface_atlas)

    # region → network
    region_to_network = {
        rid: _network_from_label_name(label)
        for rid, label in label_map.items()
    }

    # total parcels per network
    total_regions = (
        pd.Series(region_to_network)
        .value_counts()
        .to_dict()
    )

    df = global_freq.copy()

    df["region_id"] = df["region"].map(_region_id_from_region_value)
    df["network_name"] = df["region_id"].map(region_to_network)

    network_summary = (
        df.groupby("network_name")
        .agg(
            mean_freq=("global_selection_freq", "mean"),
            n_regions=("region", "count"),
        )
        .reset_index()
    )

    network_summary["total_regions"] = (
        network_summary["network_name"]
        .map(total_regions)
    )

    network_summary = network_summary.sort_values(
        "mean_freq",
        ascending=False,
    )

    return network_summary

def compute_network_summary2(subnetwork_summary):

    network_summary = (
        subnetwork_summary
        .groupby("network")
        .apply(
            lambda x: pd.Series(
                {
                    "mean_freq": np.average(
                        x["mean_freq"],
                        weights=x["n_regions"]
                    ),
                    "n_regions": x["n_regions"].sum(),
                }
            )
        )
        .reset_index()
    )

    total_regions = network_summary["n_regions"].sum()

    network_summary["total_regions"] = total_regions

    network_summary = network_summary.rename(
        columns={"network": "network_name"}
    )

    network_summary = network_summary.sort_values(
        "mean_freq",
        ascending=False,
    )

    return network_summary
# def compute_subnetwork_summary(global_freq, surface_atlas):

#     label_map = load_schaefer_label_map(surface_atlas)

#     rows = []

#     for rid, label in label_map.items():

#         network, subnetwork = _subnetwork_from_label_name(label)

#         rows.append(
#             {
#                 "region_id": rid,
#                 "network": _network_from_label_name(label),
#                 "subnetwork": subnetwork,
#             }
#         )

#     region_info = pd.DataFrame(rows)

#     df = global_freq.copy()

#     df["region_id"] = df["region"].map(
#         _region_id_from_region_value
#     )

#     df = df.merge(
#         region_info,
#         on="region_id",
#         how="left",
#     )

#     sub_summary = (
#         df.groupby(
#             ["network", "subnetwork"]
#         )
#         .agg(
#             mean_freq=("global_selection_freq", "mean"),
#             n_regions=("region", "count"),
#         )
#         .reset_index()
#     )

#     return sub_summary

def compute_subnetwork_summary(global_freq, surface_atlas):

    label_map = load_schaefer_label_map(surface_atlas)

    rows = []

    for rid, label in label_map.items():

        network, subnetwork = _subnetwork_from_label_name(label)

        rows.append(
            {
                "region_id": rid,
                "network": network,
                "subnetwork": subnetwork,
            }
        )

    region_info = pd.DataFrame(rows)

    df = global_freq.copy()

    df["region_id"] = df["region"].map(
        _region_id_from_region_value
    )

    df = df.merge(
        region_info,
        on="region_id",
        how="left",
    )

    sub_summary = (
        df.groupby(
            ["network", "subnetwork"]
        )
        .agg(
            mean_freq=("global_selection_freq", "mean"),
            n_regions=("region", "count"),
        )
        .reset_index()
    )

    return sub_summary

def build_network_embedding_matrix(
    embedding_freq: pd.DataFrame,
    surface_atlas,
) -> pd.DataFrame:
    """
    Creates a network-level embedding matrix.

    Rows:
        networks

    Columns:
        embeddings

    Values:
        mean selection frequency across all regions
        belonging to the network.
    """

    label_map = load_schaefer_label_map(surface_atlas)

    region_to_network = {
        rid: _network_from_label_name(label)
        for rid, label in label_map.items()
    }

    df = embedding_freq.copy()

    df["region_id"] = (
        df["region"]
        .map(_region_id_from_region_value)
    )

    df["network"] = (
        df["region_id"]
        .map(region_to_network)
    )

    network_freq = (
        df.groupby(
            ["network", "embedding_name"]
        )["selection_freq"]
        # .mean()                     # <-- normalization by network size
        .reset_index()
    )

    network_matrix = (
        network_freq.pivot(
            index="network",
            columns="embedding_name",
            values="selection_freq"
        )
        .fillna(0.0)
    )

    return network_matrix

def plot_network_embedding_heatmap(
    embedding_freq,
    surface_atlas,
    title,
    out_file,
):
    network_matrix = build_network_embedding_matrix(
        embedding_freq,
        surface_atlas,
    )

    plt.figure(figsize=(7, 4))

    sns.heatmap(
        network_matrix,
        cmap="Reds",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".0%",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": "Mean selection frequency"
        },
    )

    plt.xlabel("Embedding")
    plt.ylabel("Network")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(out_file, dpi=200)
    plt.close()

def apply_network_selection(df, region_to_network):
    df = df.copy()
    df["region_id"] = df["region"].map(_region_id_from_region_value)
    df["network_name"] = df["region_id"].map(region_to_network).fillna("Unknown")

    group_cols = [
        c for c in ["dataset", "task", "embedding_name", "exp_id", "network_name"]
        if c in df.columns
    ]
    df["selected"] = df.groupby(group_cols)["selected"].transform("max")
    return df.drop(columns=["region_id"])


def _plot_all_maps(
    embedding_freq,
    estimator_freq,
    global_freq,
    df_full,
    suffix,
    title_suffix,
    top_regions_by_group=None,
):
    for (dataset, task), sub_df in embedding_freq.groupby(["dataset", "task"]):
        run_id = df_full[
            (df_full["dataset"] == dataset) &
            (df_full["task"] == task)
        ]["run_id"].iloc[0]

        surface_atlas = load_atlas_from_run(run_id)

        global_vector = (
            global_freq[
                (global_freq["dataset"] == dataset) &
                (global_freq["task"] == task)
            ]
            .set_index("region")["global_selection_freq"]
        )

        embedding_matrix = sub_df.pivot_table(
            index="region",
            columns="embedding_name",
            values="selection_freq",
            aggfunc="mean"
        ).fillna(0.0)
        
        estimator_sub_df = estimator_freq[
            (estimator_freq["dataset"] == dataset)
            & (estimator_freq["task"] == task)
        ]

        estimator_matrix = estimator_sub_df.pivot_table(
            index="region",
            columns="estimator_name",
            values="selection_freq",
            aggfunc="mean"
        ).fillna(0.0)

        global_vector = global_vector.reindex(embedding_matrix.index).fillna(0.0)

        if top_regions_by_group is not None:
            top_regions = top_regions_by_group.get((dataset, task))
        else:
            top_regions = None

        if top_regions is None:
            top_regions = (
                global_vector
                .sort_values(ascending=False)
                .head(TOP_K)
                .index
            )
        heatmap_df = embedding_matrix.loc[top_regions]
        heatmap_df_estimator = estimator_matrix.loc[top_regions]

        if "global" in heatmap_df.columns:
            heatmap_df = heatmap_df.drop(columns=["global"])

        plot_brain_and_heatmap(
            surface_atlas,
            global_vector,
            heatmap_df,
            title=f"Top-{TOP_K} regions | {dataset} | {task}{title_suffix}",
            out_file=OUTPUT_DIR / f"brain_heatmap_{dataset}_{task}{suffix}.pdf"
        )
        
        plot_brain_only(
            surface_atlas,
            global_vector,
            title=f"Global selection frequency | {dataset} | {task}{title_suffix}",
            out_file=OUTPUT_DIR / f"brain_{dataset}_{task}{suffix}.pdf",
        )
        
        plot_brain_4view(surface_atlas, global_vector, 
                         title=f"Global selection frequency | {dataset} | {task}{title_suffix}", 
                         out_file=OUTPUT_DIR / f"brain4_{dataset}_{task}{suffix}.pdf")

        plot_heatmap_only(
            surface_atlas,
            heatmap_df,
            title=f"Top-{TOP_K} regions | {dataset} | {task}{title_suffix}",
            out_file=OUTPUT_DIR / f"heatmap_{dataset}_{task}{suffix}.pdf",
        )
        
        plot_network_embedding_heatmap(
            sub_df,
            surface_atlas,
            title=f"Network-level embedding frequencies | {dataset} | {task}",
            out_file=OUTPUT_DIR /
            f"check2network_embedding_heatmap_{dataset}_{task}{suffix}.pdf",
        )
        
        plot_heatmap_only(
            surface_atlas,
            heatmap_df_estimator,
            title=f"Top-{TOP_K} regions across estimators | {dataset} | {task}{title_suffix}",
            out_file=OUTPUT_DIR / f"heatmap_estimators_{dataset}_{task}{suffix}.pdf",
        )

        embedding_matrix["global"] = global_vector
        corr_matrix = embedding_matrix.corr()
        plot_corr_matrix(
            corr_matrix,
            title=f"Embedding correlations | {dataset} | {task}{title_suffix}",
            out_file=OUTPUT_DIR / f"corr_{dataset}_{task}{suffix}.pdf"
        )

        embeddings = list(embedding_matrix.columns)

        binary_matrix = (embedding_matrix >= THRESHOLD).astype(int)
        acc_matrix = compute_binary_accuracy_matrix(binary_matrix)
        plot_corr_matrix(
            acc_matrix,
            title=f"Binary agreement (≥{THRESHOLD}) | {dataset} | {task}{title_suffix}",
            out_file=OUTPUT_DIR / f"binary_agreement_{dataset}_{task}{suffix}.pdf"
        )

        acc_matrix2 = compute_binary_jaccard_matrix(binary_matrix)
        plot_corr_matrix(
            acc_matrix2,
            title=f"Binary agreement (≥{THRESHOLD}) | {dataset} | {task}{title_suffix}",
            out_file=OUTPUT_DIR / f"binary_agreement_corrected_{dataset}_{task}{suffix}.pdf"
        )

        acc_matrix3 = compute_binary_jaccard_matrix(binary_matrix, mode="anti")
        plot_corr_matrix(
            acc_matrix3,
            title=f"Binary agreement (≥{THRESHOLD}) | {dataset} | {task}{title_suffix}",
            out_file=OUTPUT_DIR / f"binary_agreement_corrected0s_{dataset}_{task}{suffix}.pdf"
        )

        acc_matrix_balanced = compute_binary_balanced_accuracy_matrix(binary_matrix)
        plot_corr_matrix(
            acc_matrix_balanced,
            title=f"Binary balanced agreement (≥{THRESHOLD}) | {dataset} | {task}{title_suffix}",
            out_file=OUTPUT_DIR / f"binary_agreement_balanced_{dataset}_{task}{suffix}.pdf"
        )

        values_by_name = {"global": to_label_map(global_vector)}
        for emb in embeddings:
            values_by_name[emb] = to_label_map(embedding_matrix[emb])

        _plot_frequency_rows(
            surface_atlas,
            values_by_name,
            f"Embedding frequency maps | {dataset} | {task}{title_suffix}",
            OUTPUT_DIR / f"embedding_frequency_maps_{dataset}_{task}{suffix}.pdf",
        )
        
def remove_unknown_embeddings(df):
    df = df.copy()
    return df[df["embedding_name"] != "unknown"]
    
def main():
  
    df_raw = pd.read_parquet(INPUT_TABLE)
    df_raw = exclude_region_permutation(df_raw)

    df_all = filter_microstructure(df_raw)
    plot_score_distribution_split(
        df_all,
        DATASET_SELECTION,
        TASK_SELECTION,
        OUTPUT_DIR / f"test_score_distribution_{DATASET_SELECTION}_{TASK_SELECTION}.pdf",
    )
    plot_pipeline_factor_boxplots(
        df_raw,
        DATASET_SELECTION,
        TASK_SELECTION,
        OUTPUT_DIR / f"pipeline_factor_boxplot_{DATASET_SELECTION}_{TASK_SELECTION}.pdf",
    )

    df = df_all.copy()
    # -------------------------
    # Filter dataset + task
    # -------------------------
    df = df[
        (df["dataset"] == DATASET_SELECTION) &
        (df["task"] == TASK_SELECTION)
    ].copy()

    # -------------------------
    # PREPARE selected (needed for agreement)
    # -------------------------
    df = exclude_region_permutation(df)
    df = filter_microstructure(df)

    df["exp_id"] = build_exp_id(df)
    df = add_embedding_name(df)
    df = remove_unknown_embeddings(df)
    df = add_percentile_selection(df)
    embedding_freq, global_freq = build_frequency_maps_from_selected(df)
    estimator_freq = build_estimator_frequency_maps(df)

    top_regions_by_group = {}
    for (dataset, task), sub_df in global_freq.groupby(["dataset", "task"]):
        top_regions_by_group[(dataset, task)] = (
            sub_df
            .set_index("region")["global_selection_freq"]
            .sort_values(ascending=False)
            .head(TOP_K)
            .index
        )

    run_id_for_atlas = df["run_id"].iloc[0]
    surface_atlas_for_network = load_atlas_from_run(run_id_for_atlas)
    region_to_network = build_region_to_network(surface_atlas_for_network)
    df_network = apply_network_selection(df, region_to_network)
    embedding_freq_net, global_freq_net = build_frequency_maps_from_selected(df_network)
    estimator_freq_net = build_estimator_frequency_maps(df_network)

    _plot_all_maps(embedding_freq, estimator_freq, global_freq, df, suffix="", title_suffix="")
    _plot_all_maps(
        embedding_freq_net,
        estimator_freq_net,
        global_freq_net,
        df_network,
        suffix="_network",
        title_suffix=" | network-stable",
        top_regions_by_group=top_regions_by_group,
    )

    for dataset, task in global_freq_net[["dataset", "task"]].drop_duplicates().itertuples(index=False):
        plot_region_rank_from_network(
            global_freq_net,
            dataset,
            task,
            OUTPUT_DIR / f"region_rank_network_{dataset}_{task}.pdf",
        )
        sub_global = global_freq_net[
            (global_freq_net["dataset"] == dataset)
            & (global_freq_net["task"] == task)
        ]

        network_summary = compute_network_summary(
            sub_global,
            surface_atlas_for_network,
        )

        plot_network_frequency_barplot(
            network_summary,
            dataset,
            task,
            OUTPUT_DIR /
            f"network_frequency_barplot_{dataset}_{task}.pdf",
        )
        subnetwork_summary = compute_subnetwork_summary(
            sub_global,
            surface_atlas_for_network,
        )

        network_summary = compute_network_summary2(
            subnetwork_summary
        )
        plot_network_frequency_hierarchy(
            network_summary,
            subnetwork_summary,
            dataset,
            task,
            OUTPUT_DIR /
            f"network_hierarchy_{dataset}_{task}.pdf",
        )

if __name__ == "__main__":
	main()