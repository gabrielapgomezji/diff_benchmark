import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from itertools import combinations

from diff_benchmark.analysis.region_coefficients import load_atlas_from_run

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PERCENTILE = 0.90

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
    / "new_plots"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK = "binary_classification"
MICROSTRUCTURE = "md"

DATASETS = ["hcp", "camcan"]

# ============================================================
# External functions assumed to exist in your codebase
# ============================================================
# - build_dataset_summaries
# - exclude_region_permutation
# - filter_bottom_runs

def build_exp_id(df):

    cols = ["run_id"]

    if "fold" in df.columns:
        cols.append("fold")

    if "seed" in df.columns:
        cols.append("seed")

    return df[cols].astype(str).agg("_".join, axis=1)

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

def get_region_to_network(surface_atlas):

    label_map = load_schaefer_label_map(
        surface_atlas
    )

    return {
        rid: _network_from_label_name(label)
        for rid, label in label_map.items()
    }

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

def _region_id_from_region_value(region_value):

    try:
        return int(str(region_value).split(":")[-1])
    except Exception:
        return None

def build_dataset_summaries(df, dataset):

    sub = df[
        (df["dataset"] == dataset)
        & (df["task"] == TASK)
        & (df["microstructure"] == MICROSTRUCTURE)
    ].copy()

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

    return pd.concat(
        [
            parcel_summary,
            touch_summary,
            region_summary,
        ],
        ignore_index=True,
    )

# ============================================================
# Matrix building (CORE FIX)
# ============================================================

def build_matrix_from_summary(summary_df: pd.DataFrame):

    region_df = summary_df[
        summary_df["metric"] == "region_mean"
    ].copy()

    pivot = region_df.pivot_table(
        index="embedding_name",
        columns="region",
        values="mean_freq",
        fill_value=0.0,
    )

    pivot = pivot.sort_index()

    return pivot.values


# ============================================================
# Metadata alignment
# ============================================================

def build_metadata(summary_df: pd.DataFrame):

    region_df = summary_df[
        summary_df["metric"] == "region_mean"
    ].copy()

    meta = (
        region_df[["embedding_name", "dataset"]]
        .drop_duplicates()
        .sort_values("embedding_name")
        .to_dict("records")
    )

    return meta


# ============================================================
# Grouping function for Plot B
# ============================================================

def my_group_fn(meta_i, meta_j):

    same_dataset = meta_i["dataset"] == meta_j["dataset"]
    same_embedding = meta_i["embedding_name"] == meta_j["embedding_name"]

    # same embedding across datasets (estimator stable across data)
    if same_embedding and not same_dataset:
        return "same_embedding_diff_estimator"

    # same dataset, different embeddings (embedding sensitivity)
    if same_dataset and not same_embedding:
        return "same_estimator_diff_embedding"

    # cross-dataset unrelated comparisons
    if not same_dataset:
        return "random_baseline"

    return "other"


# ============================================================
# Plot B: correlation distributions
# ============================================================

def plot_stability_map_correlations(
    stability_maps,
    meta=None,
    group_fn=None,
    figsize=(10, 6),
):

    stability_maps = np.asarray(stability_maps)
    n_models = stability_maps.shape[0]

    groups = {
        "same_estimator_diff_embedding": [],
        "same_embedding_diff_estimator": [],
        "random_baseline": [],
        "other": [],
    }

    for i, j in combinations(range(n_models), 2):

        x = stability_maps[i]
        y = stability_maps[j]

        if np.std(x) == 0 or np.std(y) == 0:
            continue

        corr = np.corrcoef(x, y)[0, 1]

        if group_fn is None:
            group = "other"
        else:
            group = group_fn(meta[i], meta[j])

        if group not in groups:
            group = "other"

        groups[group].append(corr)

    plt.figure(figsize=figsize)

    palette = {
        "same_estimator_diff_embedding": "#1f77b4",
        "same_embedding_diff_estimator": "#ff7f0e",
        "random_baseline": "#2ca02c",
        "other": "#7f7f7f",
    }

    for k, values in groups.items():
        if len(values) > 0:
            sns.kdeplot(
                values,
                label=f"{k} (n={len(values)})",
                fill=True,
                alpha=0.3,
                color=palette.get(k, None),
            )

    plt.xlabel("Pearson correlation between stability maps")
    plt.ylabel("Density")
    plt.title("Stability map similarity across experimental factors")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "stability_map_correlations.png",
        dpi=300
    )

    return groups

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

def filter_bottom_runs(df, score_col="test_score", percentile=0.10):
    """
    Remove runs in the bottom percentile based on aggregated score.
    """

    run_cols = [c for c in ["run_id", "fold", "seed"] if c in df.columns]

    run_scores = (
        df.groupby(run_cols)[score_col]
        .mean()
        .reset_index()
    )

    threshold = run_scores[score_col].quantile(percentile)

    good_runs = run_scores[run_scores[score_col] > threshold]

    df = df.merge(good_runs[run_cols], on=run_cols, how="inner")

    return df

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    df = pd.read_parquet(INPUT_TABLE)

    # --------------------------------------------------------
    # preprocessing
    # --------------------------------------------------------
    df = exclude_region_permutation(df)
    df = filter_bottom_runs(df, score_col="test_score", percentile=0.10)

    # --------------------------------------------------------
    # build summaries (KEY CHANGE)
    # --------------------------------------------------------
    summary_df = pd.concat(
        [
            build_dataset_summaries(df, d)
            for d in DATASETS
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # build stability matrix
    # --------------------------------------------------------
    stability_maps = build_matrix_from_summary(summary_df)

    # --------------------------------------------------------
    # metadata alignment
    # --------------------------------------------------------
    meta = build_metadata(summary_df)

    # --------------------------------------------------------
    # Plot B
    # --------------------------------------------------------
    plot_stability_map_correlations(
        stability_maps,
        meta=meta,
        group_fn=my_group_fn
    )


if __name__ == "__main__":
    main()