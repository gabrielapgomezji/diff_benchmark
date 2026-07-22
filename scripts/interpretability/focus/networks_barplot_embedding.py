import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path

from diff_benchmark.analysis.region_coefficients import load_atlas_from_run

GLOBAL_MAX = 0.80
TASK_SELECTION = "binary_classification"
# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_TABLE = (
    PROJECT_ROOT
    / "exp_outputs"
    / "summary"
    / "coefficients_long.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "exp_outputs"
    / "summary"
    / "networks_embedding"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MICROSTRUCTURE = "md"
TASK = "binary_classification"
PERCENTILE = 0.90

DATASETS = ["hcp", "camcan"]

import ast

def safe_parse_embedding(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


# ============================================================
# Helpers
# ============================================================
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

def exclude_region_permutation(df):

    cols = [
        c
        for c in ["model", "model_type", "model_name"]
        if c in df.columns
    ]

    if not cols:
        return df

    mask = np.ones(len(df), dtype=bool)

    for c in cols:
        mask &= ~df[c].astype(str).str.contains(
            "region_permutation",
            case=False,
            na=False,
        )

    return df.loc[mask].copy()


def build_exp_id(df):

    cols = ["run_id"]

    if "fold" in df.columns:
        cols.append("fold")

    if "seed" in df.columns:
        cols.append("seed")

    return df[cols].astype(str).agg("_".join, axis=1)

def normalize_estimator_name(x):
    x = str(x).lower()

    if x in {"region_pca", "pointnet"}:
        return "region_group_lasso"

    return x

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

def _get_estimator_col(df: pd.DataFrame) -> str | None:
    for col in ["estimator", "model_name", "model_type", "model"]:
        if col in df.columns:
            return col
    return None

def add_embedding_name(df):
    df = df.copy()
    df["embedding_name"] = df.apply(extract_embedding, axis=1)
    # return df
    return df[df["embedding_name"] != "unknown"].copy()

def _is_lower_better(task: str) -> bool:
    task_lower = str(task).lower()
    return "age" in task_lower or "regression" in task_lower

def plot_pipeline_factor_boxplots(df, datasets, task, out_file):
    """
    Split-violin comparison of pipeline factors across two datasets.

    Parameters
    ----------
    df       : raw long-form DataFrame (all datasets/tasks)
    datasets : list of exactly two dataset names, e.g. ["hcp", "camcan"]
               — first goes on the LEFT half of each violin, second on the RIGHT
    task     : task string used to filter df
    out_file : output path (PDF / PNG)

    Layout
    ------
      Row 0 | Estimator  | Microstructure |
      Row 1 | Embedding  | Legend         |
    """
    if len(datasets) != 2:
        raise ValueError("datasets must contain exactly two entries")

    estimator_col = _get_estimator_col(df)
    if estimator_col is None:
        return

    # ----------------------------------------------------------------
    # Per-dataset: aggregate to run level, apply performance filter
    # ----------------------------------------------------------------
    palette   = {datasets[0]: "#43A047",   # green  — left violin half
                 datasets[1]: "#E53935"}    # red    — right violin half

    combined_filtered = []
    thresholds = {}

    for dataset in datasets:
        sub = df[(df["dataset"] == dataset) & (df["task"] == task)].copy()
        if sub.empty:
            continue

        if "embedding_name" not in sub.columns:
            sub = add_embedding_name(sub)
        sub = sub[sub["embedding_name"] != "unknown"].copy()

        if "microstructure" not in sub.columns:
            sub["microstructure"] = "unknown"

        run_cols   = [c for c in ["run_id", "fold", "seed"] if c in sub.columns]
        group_cols = (
            ["dataset", "task", "microstructure", "embedding_name", estimator_col]
            + run_cols
        )

        run_df = sub.groupby(group_cols)["test_score"].mean().reset_index()
        run_df[estimator_col] = run_df[estimator_col].apply(normalize_estimator_name)

        scores = run_df["test_score"].dropna().to_numpy(dtype=float)
        if scores.size == 0:
            continue

        if _is_lower_better(task):
            thr       = float(np.quantile(scores, 0.75))
            keep_mask = run_df["test_score"] <= thr
        else:
            thr       = float(np.quantile(scores, 0.25))
            keep_mask = run_df["test_score"] >= thr

        thresholds[dataset] = thr
        combined_filtered.append(run_df[keep_mask].copy())

    if not combined_filtered:
        return

    df_plot    = pd.concat(combined_filtered, ignore_index=True)
    score_label = "MAE" if _is_lower_better(task) else "Balanced Accuracy"

    # ----------------------------------------------------------------
    # Grid layout: 2 × 2, last cell = legend
    # ----------------------------------------------------------------
    factors = [
        ("Estimator", estimator_col),
        ("Microstructure", "microstructure"),
    ]

    sns.set_style("whitegrid")
    fig  = plt.figure(figsize=(14, 10))
    gs   = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25)

    # Build axes; share y-axis across the three data panels
    ax0  = fig.add_subplot(gs[0, 0])
    ax1  = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2  = fig.add_subplot(gs[1, 0], sharey=ax0)
    ax_legend = fig.add_subplot(gs[1, 1])
    data_axes = [ax0, ax1, ax2]

    for ax, (title, col) in zip(data_axes, factors):

        sns.violinplot(
            data=df_plot,
            x=col,
            y="test_score",
            hue="dataset",
            hue_order=datasets,
            palette=palette,
            split=True,
            inner="quartile",
            cut=0,
            scale="width",
            ax=ax,
        )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel(score_label if ax is ax0 else "")
        ax.legend_.remove()

        # Per-category sample-count annotation below the x-axis
        y_min, y_max = ax.get_ylim()
        y_text = y_min - 0.07 * (y_max - y_min)
        ax.set_ylim(y_text, y_max)

        xtick_labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
        for xpos, label in zip(ax.get_xticks(), xtick_labels):
            parts = []
            for ds in datasets:
                n = int((df_plot[df_plot["dataset"] == ds][col] == label).sum())
                parts.append(str(n))
            ax.text(
                xpos, y_text, "/".join(parts),
                ha="center", va="top", fontsize=12.0, color="#555555",
            )

    # ----------------------------------------------------------------
    # Legend panel — dataset colours + per-dataset threshold line note
    # ----------------------------------------------------------------
    ax_legend.axis("off")

    legend_handles = [
        mpatches.Patch(color=palette[ds], label=(
            f"{ds.upper()}  (threshold: {thresholds.get(ds, float('nan')):.3f})"
        ))
        for ds in datasets
    ]

    ax_legend.legend(
        handles=legend_handles,
        loc="center",
        fontsize=13,
        frameon=True,
        title="Dataset  ·  left / right violin half",
        title_fontsize=14,
    )

    fig.suptitle(
        f"Pipeline factor performance  ·  {_display_task_label(datasets[0], task)}",
        fontsize=16,
        y=0.995,
    )

    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_percentile_selection(df):

    group_cols = [
        c
        for c in ["run_id", "fold", "seed"]
        if c in df.columns
    ]

    df = df.copy()

    df["selected"] = (
        df.groupby(group_cols)["coef"]
        .transform(
            lambda s: (
                s.abs()
                >= s.abs().quantile(PERCENTILE)
            ).astype(int)
        )
    )

    return df


def _region_id_from_region_value(region_value):

    try:
        return int(str(region_value).split(":")[-1])
    except Exception:
        return None


def _network_from_label_name(label_name):

    name = label_name.replace(
        "17Networks_",
        "",
    )

    parts = name.split("_")

    if len(parts) < 2:
        return "Unknown"

    network = parts[1]

    if network.endswith(("A", "B", "C")):
        network = network[:-1]

    if network in {"VisCent", "VisPeri"}:
        return "Vis"

    if network == "TempPar":
        return "SalVentAttn"

    return network


def _short_region_label(label_name):

    name = label_name.replace(
        "17Networks_",
        "",
    )

    parts = name.split("_")

    if len(parts) <= 2:
        return name.replace("_", " ")

    hemi = parts[0]
    rest = "_".join(parts[2:])

    return f"{hemi} {rest}".replace("_", " ")


def load_schaefer_label_map(surface_atlas):

    tsv_path = surface_atlas["atlas_meta"]["label_tsv_path"]

    labels_df = pd.read_csv(
        tsv_path,
        sep="\t",
    )

    labels_df = labels_df[
        ~labels_df["name"].str.contains(
            "Background",
            na=False,
        )
    ]

    parcel_ids = np.unique(
        surface_atlas["parcel_labels"]
    )

    parcel_ids = parcel_ids[
        parcel_ids != 0
    ]

    return {
        int(pid): str(name)
        for pid, name in zip(
            parcel_ids,
            labels_df["name"],
        )
    }


def get_region_to_network(surface_atlas):

    label_map = load_schaefer_label_map(
        surface_atlas
    )

    return {
        rid: _network_from_label_name(label)
        for rid, label in label_map.items()
    }


# ============================================================
# Dataset summary
# ============================================================

def build_dataset_summaries(df, dataset):

    sub = df[
        (df["dataset"] == dataset)
        & (df["task"] == TASK)
        & (df["microstructure"] == MICROSTRUCTURE)
    ].copy()

    embedding_name = (
        sub["embedding_name"].iloc[0]
        if len(sub)
        else "unknown"
    )

    sub["exp_id"] = build_exp_id(sub)

    sub = add_percentile_selection(sub)

    run_id = sub["run_id"].iloc[0]

    surface_atlas = load_atlas_from_run(run_id)

    region_to_network = get_region_to_network(
        surface_atlas
    )

    label_map = load_schaefer_label_map(
        surface_atlas
    )

    region_to_label = {
        rid: _short_region_label(name)
        for rid, name in label_map.items()
    }

    # --------------------------------------------------------
    # Region → Network mapping
    # --------------------------------------------------------

    sub["region_id"] = (
        sub["region"]
        .map(_region_id_from_region_value)
    )

    sub["network_name"] = (
        sub["region_id"]
        .map(region_to_network)
    )

    # ========================================================
    # 1. Parcel-level frequency
    # ========================================================

    global_freq = (
        sub.groupby("region")["selected"]
        .mean()
        .reset_index(
            name="global_selection_freq"
        )
    )

    global_freq["region_id"] = (
        global_freq["region"]
        .map(_region_id_from_region_value)
    )

    global_freq["network_name"] = (
        global_freq["region_id"]
        .map(region_to_network)
    )

    global_freq["region_label"] = (
        global_freq["region_id"]
        .map(region_to_label)
    )

    parcel_summary = (
        global_freq.groupby("network_name")
        .agg(
            mean_freq=(
                "global_selection_freq",
                "mean",
            )
        )
        .reset_index()
    )

    parcel_summary["dataset"] = dataset
    parcel_summary["metric"] = "parcel_mean"

    # ========================================================
    # 2. Network-touch frequency
    # ========================================================

    network_df = sub.copy()

    network_df["selected"] = (
        network_df.groupby(
            [
                "exp_id",
                "network_name",
            ]
        )["selected"]
        .transform("max")
    )

    touch_summary = (
        network_df.groupby("network_name")
        .agg(
            mean_freq=(
                "selected",
                "mean",
            )
        )
        .reset_index()
    )

    touch_summary["dataset"] = dataset
    touch_summary["metric"] = "network_touch"

    # ========================================================
    # 3. Region-level frequency (per-parcel, kept for the
    #    network/region pyramid plot)
    # ========================================================

    region_summary = global_freq.rename(
        columns={"global_selection_freq": "mean_freq"}
    )[["region", "region_id", "region_label", "network_name", "mean_freq"]]

    region_summary["dataset"] = dataset
    region_summary["metric"] = "region_mean"
    
    parcel_summary["embedding_name"] = embedding_name
    touch_summary["embedding_name"] = embedding_name
    region_summary["embedding_name"] = embedding_name

    return pd.concat(
        [
            parcel_summary,
            touch_summary,
            region_summary,
        ],
        ignore_index=True,
    )


# ============================================================
# Network + region pyramid plot
# ============================================================

def plot_network_region_pyramid(
    plot_df,
    dataset_left,
    dataset_right,
    microstructure=MICROSTRUCTURE,
    output_dir=OUTPUT_DIR,
    network_cmap="Reds",
    region_cmap="Blues",
    network_height=0.74,
    region_height=0.46,
    group_gap=0.05,
    center_gap_frac=0.09,
):
    """Horizontal, back-to-back ("tornado") bar chart.

    - `dataset_left` values extend to the left, `dataset_right` to the
      right, both starting from a blank central gutter rather than
      from x = 0 directly
    - Network / region names sit inside that gutter, so they never
      sit on top of a bar — even a bar that is almost zero long
    - Each network is drawn as one bar (red colormap); the Schaefer
      regions that make it up are nested directly underneath as
      thinner bars (blue colormap)
    - Bar shade encodes magnitude (mean of the two datasets), so the
      most consistently-selected rows read as darker
    - Regions with a value of 0 in BOTH datasets are dropped. The
      parent network bar is unaffected — it still reflects the mean
      over *all* regions in that network, exactly as in Figure 1
    - `center_gap_frac` sets the gutter width as a fraction of the
      data span; widen it if long labels still touch a bar
    """

    # --------------------------------------------------------
    # Network-level data (same data as Figure 1)
    # --------------------------------------------------------

    net_df = plot_df[
        (plot_df["metric"] == "parcel_mean")
        & (plot_df["dataset"].isin([dataset_left, dataset_right]))
    ]

    net_wide = (
        net_df
        .pivot_table(
            index="network_name",
            columns="dataset",
            values="mean_freq",
        )
        .reindex(columns=[dataset_left, dataset_right])
    )

    # --------------------------------------------------------
    # Region-level data, keeping only regions with a non-zero
    # value in at least one of the two datasets
    # --------------------------------------------------------

    reg_df = plot_df[
        (plot_df["metric"] == "region_mean")
        & (plot_df["dataset"].isin([dataset_left, dataset_right]))
    ]

    reg_wide = (
        reg_df
        .pivot_table(
            index=["network_name", "region", "region_label"],
            columns="dataset",
            values="mean_freq",
        )
        .reindex(columns=[dataset_left, dataset_right])
        .reset_index()
    )

    reg_wide[[dataset_left, dataset_right]] = (
        reg_wide[[dataset_left, dataset_right]].fillna(0.0)
    )

    reg_wide = reg_wide[
        (reg_wide[dataset_left] > 0)
        | (reg_wide[dataset_right] > 0)
    ]

    # --------------------------------------------------------
    # Build row list: network row + its surviving region rows,
    # networks ordered by combined mean, regions within a
    # network ordered by their own combined mean
    # --------------------------------------------------------

    network_order = (
        net_wide.mean(axis=1)
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    rows = []  # dicts: kind, label, left, right, color_val, group

    for g, network in enumerate(network_order):

        left_val, right_val = net_wide.loc[network]
        left_val = 0.0 if pd.isna(left_val) else float(left_val)
        right_val = 0.0 if pd.isna(right_val) else float(right_val)

        sub = reg_wide[reg_wide["network_name"] == network].copy()

        if sub.empty:
            continue

        sub["mean_val"] = sub[[dataset_left, dataset_right]].mean(axis=1)
        sub = sub.sort_values("mean_val", ascending=False)

        rows.append(dict(
            kind="network", label=str(network),
            left=left_val, right=right_val,
            color_val=(left_val + right_val) / 2, group=g,
        ))

        for _, r in sub.iterrows():
            label = (
                r["region_label"]
                if pd.notna(r["region_label"])
                else r["region"]
            )
            rows.append(dict(
                kind="region", label=str(label),
                left=float(r[dataset_left]), right=float(r[dataset_right]),
                color_val=float(r["mean_val"]), group=g,
            ))

    # --------------------------------------------------------
    # y positions: 1 unit per row, plus extra air between
    # consecutive network groups
    # --------------------------------------------------------

    y_positions = []
    y_cursor = 0.0
    prev_group = None

    for row in rows:
        if prev_group is not None and row["group"] != prev_group:
            y_cursor -= group_gap
        y_cursor -= 1.0
        y_positions.append(y_cursor)
        prev_group = row["group"]

    group_bands = {}
    for y, row in zip(y_positions, rows):
        lo, hi = group_bands.get(row["group"], (y, y))
        group_bands[row["group"]] = (min(lo, y), max(hi, y))

    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    n_rows = len(rows)
    n_groups = len(group_bands)

    max_val = max(
        (max(row["left"], row["right"]) for row in rows),
        default=0.0,
    )

    base_span = max_val * 1.15 if max_val > 0 else 1.0
    center_gap = center_gap_frac * base_span
    xlim = center_gap + base_span

    fig_height = max(6, n_rows * 0.27 + n_groups * 0.15)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    network_norm = Normalize(
        vmin=float(net_wide.min().min()),
        vmax=float(net_wide.max().max()),
    )

    region_values = reg_wide[[dataset_left, dataset_right]].to_numpy()
    region_vmax = float(region_values.max()) if region_values.size else 1.0

    region_norm = Normalize(vmin=0.0, vmax=region_vmax if region_vmax > 0 else 1.0)

    network_cmap_obj = plt.get_cmap(network_cmap)
    region_cmap_obj = plt.get_cmap(region_cmap)

    def _shade(cmap_obj, norm, value, lo=0.40, hi=0.92):
        t = float(np.clip(norm(value), 0.0, 1.0))
        return cmap_obj(lo + (hi - lo) * t)

    # zebra background bands, alternating every other network group
    for i, (g, (lo, hi)) in enumerate(
        sorted(group_bands.items(), key=lambda kv: kv[1][0])
    ):
        if i % 2 == 0:
            ax.axhspan(
                lo - 0.5, hi + 0.5,
                color="#F4F4F4", zorder=0, linewidth=0,
            )

    for y, row in zip(y_positions, rows):

        is_network = row["kind"] == "network"
        height = network_height if is_network else region_height
        left_val, right_val = row["left"], row["right"]

        color = (
            _shade(network_cmap_obj, network_norm, row["color_val"])
            if is_network
            else _shade(region_cmap_obj, region_norm, row["color_val"])
        )

        ax.barh(
            y, -left_val, left=-center_gap, height=height, color=color,
            edgecolor="white", linewidth=0.5, zorder=2,
        )
        ax.barh(
            y, right_val, left=center_gap, height=height, color=color,
            edgecolor="white", linewidth=0.5, zorder=2,
        )

        ax.text(
            0, y, row["label"],
            ha="center", va="center",
            fontsize=9.5 if is_network else 7,
            fontweight="bold" if is_network else "normal",
            style="normal" if is_network else "italic",
            color="#1A1A1A" if is_network else "#444444",
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )

    # --------------------------------------------------------
    # Cosmetics
    # --------------------------------------------------------

    top_y = max(y_positions) if y_positions else 0.0
    bottom_y = min(y_positions) if y_positions else 0.0

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(bottom_y - 1.2, top_y + 1.8)
    ax.set_yticks([])

    # Value gridlines every 0.2, mirrored on both sides of the
    # gutter. Tick *positions* are shifted out by center_gap, but
    # the *labels* show the true underlying value.
    step = 0.2
    v_max_tick = (
        np.ceil(max_val / step) * step if max_val > 0 else step
    )
    v_ticks = np.arange(0, v_max_tick + step / 2, step)

    tick_positions = sorted(
        {center_gap + v for v in v_ticks}
        | {-(center_gap + v) for v in v_ticks}
    )

    ax.set_xticks(tick_positions)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{abs(x) - center_gap:.1f}")
    )
    ax.xaxis.grid(True, color="#E3E3E3", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", colors="#555555", labelsize=9)

    ax.axvline(0, color="#333333", linewidth=1.0, zorder=3)

    ax.set_xlabel(
        "Selection frequency", fontsize=10, color="#444444", labelpad=10,
    )

    ax.text(
        -xlim * 0.99, top_y + 1.0, f"\u2190  {dataset_left.upper()}",
        ha="left", va="bottom", fontsize=12, fontweight="bold", color="#222222",
    )
    ax.text(
        xlim * 0.99, top_y + 1.0, f"{dataset_right.upper()}  \u2192",
        ha="right", va="bottom", fontsize=12, fontweight="bold", color="#222222",
    )

    legend_handles = [
        mpatches.Patch(color=network_cmap_obj(0.7), label="Network"),
        mpatches.Patch(color=region_cmap_obj(0.7), label="Region"),
    ]

    ax.legend(
        handles=legend_handles, loc="lower right", ncol=2,
        frameon=True, #fontsize=9.5, bbox_to_anchor=(0.5, -0.07),
        framealpha=0.9,
        fontsize=9.5,
        borderpad=0.6,
    )

    # ax.set_title(
    #     f"{microstructure.upper()} microstructure  \u00b7  non-zero regions only",
    #     fontsize=9.5, color="#777777", pad=14,
    # )
    # fig.suptitle(
    #     "Network & Region Selection Frequency",
    #     fontsize=15, fontweight="bold", color="#1A1A1A", y=0.995,
    # )

    # plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.tight_layout(pad=0.2)

    out_path = (
        output_dir
        / f"network_region_pyramid_{embedding}_{microstructure}.pdf"
    )

    plt.savefig(out_path, dpi=300, facecolor="white")

    plt.close()

    return out_path


def plot_single_network_pyramid(
    network,
    net_wide,
    reg_wide,
    dataset_left,
    dataset_right,
    output_dir,
    microstructure=MICROSTRUCTURE,
    network_cmap="Reds",
    region_cmap="Blues",
    center_gap_frac=0.10,
):
    if network not in net_wide.index:
        return None

    left_net, right_net = net_wide.loc[network]

    sub = reg_wide[reg_wide["network_name"] == network].copy()
    if sub.empty:
        return None

    sub["mean_val"] = sub[[dataset_left, dataset_right]].mean(axis=1)
    sub = sub.sort_values("mean_val", ascending=False)

    max_val = max(
        float(sub[[dataset_left, dataset_right]].to_numpy().max()),
        float(max(left_net, right_net)),
    )

    # base_span = max_val * 1.2 if max_val > 0 else 1.0
    base_span = GLOBAL_MAX
    center_gap = center_gap_frac * base_span
    xlim = base_span + center_gap

    fig, ax = plt.subplots(figsize=(10, max(3, len(sub) * 0.35 + 2)))

    net_cmap = plt.get_cmap(network_cmap)
    reg_cmap = plt.get_cmap(region_cmap)

    norm_net = Normalize(
        vmin=float(net_wide.min().min()),
        vmax=float(net_wide.max().max()),
    )

    norm_reg = Normalize(vmin=0.0, vmax=float(sub["mean_val"].max() or 1.0))

    def shade(cmap, norm, v):
        t = np.clip(norm(v), 0, 1)
        return cmap(0.3 + 0.6 * t)

    rows = [dict(
        kind="network",
        label=network,
        left=float(left_net),
        right=float(right_net),
        val=(left_net + right_net) / 2,
    )]

    for _, r in sub.iterrows():
        rows.append(dict(
            kind="region",
            label=r["region_label"],
            left=float(r[dataset_left]),
            right=float(r[dataset_right]),
            val=float(r["mean_val"]),
        ))

    y = np.arange(len(rows))[::-1]

    for i, row in enumerate(rows):

        color = shade(
            net_cmap if row["kind"] == "network" else reg_cmap,
            norm_net if row["kind"] == "network" else norm_reg,
            row["val"],
        )

        ax.barh(y[i], -row["left"], left=-center_gap,
                color=color, edgecolor="white", height=0.7)
        ax.barh(y[i], row["right"], left=center_gap,
                color=color, edgecolor="white", height=0.7)

        ax.text(
            0, y[i], row["label"],
            ha="center", va="center",
            fontsize=10 if row["kind"] == "network" else 8,
            fontweight="bold" if row["kind"] == "network" else "normal",
        )

    # ❌ NO BLACK CENTER LINE (removed axvline entirely)

    ax.set_xlim(-xlim, xlim)
    ax.set_yticks([])

    # dataset labels on BOTH sides
    ax.text(-xlim * 0.95, y.max() + 0.4, dataset_left.upper(),
            ha="left", va="bottom", fontsize=11, fontweight="bold")
    ax.text(xlim * 0.95, y.max() + 0.4, dataset_right.upper(),
            ha="right", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Selection frequency", fontsize=10)

    ax.set_title(network, fontsize=12, pad=10)

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{abs(x) - center_gap:.1f}")
    )

    plt.tight_layout()

    out_path = output_dir / f"pyramid_{network}_{embedding}_{microstructure}.pdf"
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path

def plot_network_grid(
    net_wide,
    reg_wide,
    dataset_left,
    dataset_right,
    output_dir,
    microstructure=MICROSTRUCTURE,
    freq_threshold=0.1,
):
    networks = list(net_wide.index)
    n = len(networks)

    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, 4 * nrows),
    )

    axes = np.array(axes).reshape(-1)

    legend_handles = None

    for i, ax in enumerate(axes):

        if i >= n:
            ax.axis("off")
            continue

        network = networks[i]

        left_net, right_net = net_wide.loc[network]

        sub = reg_wide[reg_wide["network_name"] == network].copy()

        # Keep only regions where at least one dataset clears the threshold
        sub = sub[
            (sub[dataset_left] > freq_threshold)
            | (sub[dataset_right] > freq_threshold)
        ]

        if sub.empty:
            ax.axis("off")
            continue

        sub["mean_val"] = sub[[dataset_left, dataset_right]].mean(axis=1)
        sub = sub.sort_values("mean_val", ascending=False)

        base_span = GLOBAL_MAX
        center_gap = 0.1 * base_span
        xlim = base_span + center_gap

        rows = [dict(
            label=network,
            left=float(left_net),
            right=float(right_net),
        )]

        for _, r in sub.iterrows():
            rows.append(dict(
                label=r["region_label"],
                left=float(r[dataset_left]),
                right=float(r[dataset_right]),
            ))

        y = np.arange(len(rows))[::-1]

        for j, row in enumerate(rows):

            color = "firebrick" if j == 0 else "steelblue"

            ax.barh(y[j], row["left"], left=-(center_gap + row["left"]),
                    color=color, alpha=0.8)
            ax.barh(y[j], row["right"], left=center_gap,
                    color=color, alpha=0.8)

            ax.text(0, y[j], row["label"], ha="center", va="center", fontsize=12)

        ax.text(-xlim * 0.95, y.max() + 0, dataset_left.upper(),
                ha="left", va="bottom", fontsize=14, fontweight="bold")
        ax.text(xlim * 0.95, y.max() + 0, dataset_right.upper(),
                ha="right", va="bottom", fontsize=14, fontweight="bold")

        # ax.set_xlim(-xlim, xlim)
        # ax.set_yticks([])

        # # Show true (positive) frequency values on both sides of the gutter
        # ax.xaxis.set_major_formatter(
        #     FuncFormatter(lambda x, _: f"{max(abs(x) - center_gap, 0):.1f}")
        # )
        ax.set_xlim(-xlim, xlim)
        ax.set_yticks([])

        # Build explicit ticks so 0.0 sits exactly where each bar starts
        # (¬±center_gap), not at the geometric center of the gutter
        step = 0.2
        v_max_tick = np.ceil(xlim / step) * step
        v_ticks = np.arange(0, v_max_tick + step / 2, step)
        tick_positions = sorted(
            {center_gap + v for v in v_ticks}
            | {-(center_gap + v) for v in v_ticks}
        )
        ax.set_xticks(tick_positions)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{abs(x) - center_gap:.1f}")
        )

        ax.set_xlabel("Selection frequency", fontsize=14)

        ax.set_title(network, fontsize=16)

        if legend_handles is None:
            legend_handles = [
                mpatches.Patch(color="firebrick", label="Network"),
                mpatches.Patch(color="steelblue", label="Region"),
            ]

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower right",
            frameon=True,
            fontsize=20,
        )

    plt.tight_layout()
    out_path = output_dir / f"network_grid_{embedding}_{microstructure}.pdf"
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path

def generate_all_figures(
    plot_df,
    raw_df,
    output_dir,
    embedding,
):
    # ============================================================
    # Figure 1
    # Parcel-level mean selection frequency
    # ============================================================

    parcel_df = plot_df[
        plot_df["metric"] == "parcel_mean"
    ]

    parcel_order = (
        parcel_df.groupby("network_name")["mean_freq"]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=parcel_df,
        x="network_name",
        y="mean_freq",
        hue="dataset",
        order=parcel_order,
    )

    plt.ylabel(
        "Mean parcel selection frequency"
    )

    plt.xlabel("")

    plt.title(
        f"Parcel-level network selection frequency ({MICROSTRUCTURE})"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.legend(title="Dataset")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"network_parcel_frequency_{embedding}_{MICROSTRUCTURE}.pdf",
        dpi=300,
    )

    plt.close()


    # ============================================================
    # Figure 1b
    # Network + region pyramid (two-dataset comparison)
    # ============================================================

    # plot_network_region_pyramid(
    #     plot_df,
    #     dataset_left=DATASETS[0],
    #     dataset_right=DATASETS[1],
    # )
    net_df = plot_df[plot_df["metric"] == "parcel_mean"]
    net_df = net_df[net_df["dataset"].isin(DATASETS)]

    net_wide = (
        net_df.pivot_table(
            index="network_name",
            columns="dataset",
            values="mean_freq",
        )
    )

    reg_df = plot_df[plot_df["metric"] == "region_mean"]
    reg_df = reg_df[reg_df["dataset"].isin(DATASETS)]

    reg_wide = (
        reg_df.pivot_table(
            index=["network_name", "region", "region_label"],
            columns="dataset",
            values="mean_freq",
        )
        .reset_index()
    )

    reg_wide[DATASETS] = reg_wide[DATASETS].fillna(0.0)

    for network in net_wide.index:
        plot_single_network_pyramid(
            network=network,
            net_wide=net_wide,
            reg_wide=reg_wide,
            dataset_left=DATASETS[0],
            dataset_right=DATASETS[1],
            output_dir=OUTPUT_DIR,
        )

    plot_network_grid(
        net_wide=net_wide,
        reg_wide=reg_wide,
        dataset_left=DATASETS[0],
        dataset_right=DATASETS[1],
        output_dir=OUTPUT_DIR,
    )

    plot_pipeline_factor_boxplots(
        df,
        datasets=["hcp", "camcan"],      # hcp → green left half, camcan → red right half
        task=TASK_SELECTION,
        out_file=OUTPUT_DIR / f"pipeline_factor_boxplot_{TASK_SELECTION}.pdf",
    )
    # ============================================================
    # Figure 2
    # Network-touch frequency
    # ============================================================

    touch_df = plot_df[
        plot_df["metric"] == "network_touch"
    ]

    touch_order = (
        touch_df.groupby("network_name")["mean_freq"]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=touch_df,
        x="network_name",
        y="mean_freq",
        hue="dataset",
        order=touch_order,
    )

    plt.ylabel(
        "Network participation frequency"
    )

    plt.xlabel("")

    plt.title(
        f"Network participation frequency ({MICROSTRUCTURE})"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.legend(title="Dataset")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"network_touch_frequency_{embedding}_{MICROSTRUCTURE}.pdf",
        dpi=300,
    )

    plt.close()

    print(
        f"Saved figures to: {OUTPUT_DIR}"
    )

# ============================================================
# Main
# ============================================================

df = pd.read_parquet(INPUT_TABLE)

df = exclude_region_permutation(df)
all_results = []

df = add_embedding_name(df)

embeddings = sorted(
    df["embedding_name"].dropna().unique()
)

for embedding in embeddings:

    print(f"Processing embedding: {embedding}")

    emb_df = df[
        df["embedding_name"] == embedding
    ].copy()

    emb_output_dir = OUTPUT_DIR / embedding
    emb_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for dataset in DATASETS:
        all_results.append(
            build_dataset_summaries(
                emb_df,
                dataset,
            )
        )

    plot_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    generate_all_figures(
        plot_df=plot_df,
        raw_df=emb_df,
        output_dir=emb_output_dir,
        embedding=embedding,
    )
