import numpy as np
import pandas as pd
import ast
from pathlib import Path
from diff_benchmark.analysis.region_coefficients import load_atlas_from_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "coefficients_long.parquet"
OUTPUT_DIR = PROJECT_ROOT / "exp_outputs" / "summary" / "EASY_embedding_correlation_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# REGION_REPRESENTATIONS = ["flatten", "mean_std", "summary_stats", "percentiles", "pca"]
PERCENTILE = 0.90
MICROSTRUCTURE_SELECTION = "md"

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

    plt.imshow(corr_matrix.values.astype(float), vmin=-1, vmax=1, cmap="coolwarm")

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

def main():
  
    df = pd.read_parquet(INPUT_TABLE)
    embedding_freq, global_freq = build_frequency_maps(df)
    for (dataset, task), sub_df in embedding_freq.groupby(["dataset", "task"]):
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
        embedding_matrix["global"] = global_vector
        corr_matrix = embedding_matrix.corr()
        plot_corr_matrix(
            corr_matrix,
            title=f"Embedding correlations | {dataset} | {task}",
            out_file=OUTPUT_DIR / f"corr_{dataset}_{task}.png"
        )
        
        embeddings = list(embedding_matrix.columns)

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

        # -------------------------
        # Region-wise relative diff grid (frequency-based)
        # -------------------------
        n = len(embeddings)

        max_rel = 0.0
        for emb_a in embeddings:
            for emb_b in embeddings:
                a = embedding_matrix[emb_a]
                b = embedding_matrix[emb_b]
                # rel = (a - b).abs() / (a.abs() + 1e-8)
                # rel = (a - b).abs() / (np.maximum(a.abs(), b.abs()) + 1e-8)
                eps = 0.35
                rel = (a - b).abs() / (np.maximum(np.maximum(a, b), eps))
                if rel.notna().any():
                    max_rel = max(max_rel, float(rel.max()))
        if max_rel <= 0.0:
            max_rel = 1e-6

        fig = plt.figure(figsize=(2.4 * n * 2 + 1.6, 2.4 * n))
        gs = fig.add_gridspec(n, n * 2 + 1, width_ratios=[1] * (n * 2) + [0.06], wspace=0.02, hspace=0.08)

        vmin, vmax = 0.0, max_rel
        
        for i, emb_a in enumerate(embeddings):
            for j, emb_b in enumerate(embeddings):
                ax_left = fig.add_subplot(gs[i, j * 2], projection="3d")
                ax_right = fig.add_subplot(gs[i, j * 2 + 1], projection="3d")

                a = embedding_matrix[emb_a]
                b = embedding_matrix[emb_b]
                # rel = (a - b).abs() / (a.abs() + 1e-8)
                # rel = (a - b).abs() / (np.maximum(a.abs(), b.abs()) + 1e-8)
                eps = 0.35
                rel = (a - b).abs() / (np.maximum(np.maximum(a, b), eps))
                values = to_label_map(rel)
                title = f"{emb_a} vs {emb_b}"

                _plot_surface_row(
                    surface_atlas,
                    values,
                    ax_left,
                    ax_right,
                    vmin,
                    vmax,
                    title,
                    "Reds"
                )

        cax = fig.add_subplot(gs[:, -1])
        sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cax)

        plt.suptitle(f"Embedding relative diff grid | {dataset} | {task}", fontsize=18)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"embedding_relative_diff_grid_{dataset}_{task}.png",
            dpi=150
        )
        plt.close()

if __name__ == "__main__":
	main()