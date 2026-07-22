import numpy as np
import pandas as pd
import ast
from pathlib import Path
from diff_benchmark.analysis.region_coefficients import load_atlas_from_run
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "coefficients_long.parquet"
OUTPUT_DIR = PROJECT_ROOT / "exp_outputs" / "summary" / "DRAFT_embedding_correlation_maps" / "camcan" / "md"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# FS_LABELS_JSON = PROJECT_ROOT / "aux_materials" / "fs_labels.json"
SCHAEFER_LABELS_JSON = PROJECT_ROOT / "aux_materials" / "schaefer_labels.json"

# REGION_REPRESENTATIONS = ["flatten", "mean_std", "summary_stats", "percentiles", "pca"]
PERCENTILE = 0.90
MICROSTRUCTURE_SELECTION = "md"
DATASET_SELECTION = "camcan"
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

    if row.get("model_type") == "region_pca":
        return "pca"

    return "unknown"


def add_embedding_name(df):
    df = df.copy()
    df["embedding_name"] = df.apply(extract_embedding, axis=1)
    return df


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
    breakpoint()
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

    im = ax.imshow(data, vmin=0, vmax=1, cmap="Reds")

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
    
def main():
  
    df = pd.read_parquet(INPUT_TABLE)
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
    df = add_percentile_selection(df)
    
    embedding_freq, global_freq = build_frequency_maps(df)
    for (dataset, task), sub_df in embedding_freq.groupby(["dataset", "task"]):
        run_id = df[
            (df["dataset"] == dataset) &
            (df["task"] == task)
        ]["run_id"].iloc[0]

        surface_atlas = load_atlas_from_run(run_id)
        
        global_vector = (
            global_freq[
                (global_freq["dataset"] == dataset) &
                (global_freq["task"] == task)
            ]
            .set_index("region")["global_selection_freq"]
        )
        # -------------------------
        # region x embedding matrix (frequency)
        # -------------------------
        embedding_matrix = sub_df.pivot_table(
            index="region",
            columns="embedding_name",
            values="selection_freq",
            aggfunc="mean"
        ).fillna(0.0)
        # breakpoint()
        # emb_a, emb_b = "flatten", "flatten"
        # inspect_pair(embedding_matrix, emb_a, emb_b, top_k=10)
        global_vector = global_vector.reindex(embedding_matrix.index).fillna(0.0)
        # -------------------------
        # Select top-K regions from global
        # -------------------------
        top_regions = (
            global_vector
            .sort_values(ascending=False)
            .head(TOP_K)
            .index
        )
        heatmap_df = embedding_matrix.loc[top_regions]

        # remove "global" column if present
        if "global" in heatmap_df.columns:
            heatmap_df = heatmap_df.drop(columns=["global"])
            
        plot_brain_and_heatmap(
            surface_atlas,
            global_vector,
            heatmap_df,
            title=f"Top-{TOP_K} regions | {dataset} | {task}",
            out_file=OUTPUT_DIR / f"brain_heatmap_{dataset}_{task}.png"
        )
        
        embedding_matrix["global"] = global_vector
        corr_matrix = embedding_matrix.corr()
        plot_corr_matrix(
            corr_matrix,
            title=f"Embedding correlations | {dataset} | {task}",
            out_file=OUTPUT_DIR / f"corr_{dataset}_{task}.png"
        )
        
        embeddings = list(embedding_matrix.columns)
        
        # #  -------------------------
        # # Compute top-K corr matrix
        # #  -------------------------
        # embedding_matrix_top = embedding_matrix.loc[top_regions]
        # # corr_matrix_top = embedding_matrix_top.corr()
        # corr_matrix_top = embedding_matrix_top.drop(columns=["global"]).corr()
        # plot_corr_matrix(
        #     corr_matrix_top,
        #     title=f"Top-{TOP_K} region correlations | {dataset} | {task}",
        #     out_file=OUTPUT_DIR / f"corr_top{TOP_K}_{dataset}_{task}.png"
        # )
        
        # --------------------------
        # Region agreement maps (per embedding pair)
        # --------------------------
        binary_matrix = (embedding_matrix >= THRESHOLD).astype(int)
        acc_matrix = compute_binary_accuracy_matrix(binary_matrix)
        plot_corr_matrix(
            acc_matrix,
            title=f"Binary agreement (≥{THRESHOLD}) | {dataset} | {task}",
            out_file=OUTPUT_DIR / f"binary_agreement_{dataset}_{task}.png"
        )
        
        acc_matrix2 = compute_binary_jaccard_matrix(binary_matrix)
        plot_corr_matrix(
            acc_matrix2,
            title=f"Binary agreement (≥{THRESHOLD}) | {dataset} | {task}",
            out_file=OUTPUT_DIR / f"binary_agreement_corrected_{dataset}_{task}.png"
        )
        
        acc_matrix3 = compute_binary_jaccard_matrix(binary_matrix, mode="anti")
        plot_corr_matrix(
            acc_matrix3,
            title=f"Binary agreement (≥{THRESHOLD}) | {dataset} | {task}",
            out_file=OUTPUT_DIR / f"binary_agreement_corrected0s_{dataset}_{task}.png"
        )
        
        acc_matrix_balanced = compute_binary_balanced_accuracy_matrix(binary_matrix)
        plot_corr_matrix(
            acc_matrix_balanced,
            title=f"Binary balanced agreement (≥{THRESHOLD}) | {dataset} | {task}",
            out_file=OUTPUT_DIR / f"binary_agreement_balanced_{dataset}_{task}.png"
        )
        
        # -------------------------
        # Frequency maps grid (global + embeddings)
        # -------------------------
        run_id = df[(df["dataset"] == dataset) & (df["task"] == task)]["run_id"].iloc[0]
        surface_atlas = load_atlas_from_run(run_id)

        values_by_name = {"global": to_label_map(global_vector)}
        for emb in embeddings:
            values_by_name[emb] = to_label_map(embedding_matrix[emb])

        _plot_frequency_rows(
            surface_atlas,
            values_by_name,
            f"Embedding frequency maps | {dataset} | {task}",
            OUTPUT_DIR / f"embedding_frequency_maps_{dataset}_{task}.png",
        )

if __name__ == "__main__":
	main()