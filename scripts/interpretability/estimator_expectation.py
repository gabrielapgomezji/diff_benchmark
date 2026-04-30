from __future__ import annotations

import warnings
from pathlib import Path

import ast
import json
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diff_benchmark.analysis.region_coefficients import load_atlas_from_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "coefficients_long.parquet"
BEST_RUNS_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "best_runs_by_config.parquet"
OUTPUT_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "coefficients_selected.parquet"
STABILITY_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "region_binomial_metrics.parquet"
MAPS_DIR = PROJECT_ROOT / "exp_outputs" / "summary" / "brain_maps_binomial_hard_filter"
FS_LABELS_JSON = PROJECT_ROOT / "aux_materials" / "fs_labels.json"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _pick_group_columns(df: pd.DataFrame) -> list[str]:
    # Group experiments by these columns to select best runs within each group.
    # candidates = ["microstructure", "model_type", "embedding", "task", "dataset"]
    candidates = ["task", "dataset"] 
    if "model_type" not in df.columns and "model" in df.columns:
        df["model_type"] = df["model"]
    return [c for c in candidates if c in df.columns]


def _build_exp_id(df: pd.DataFrame) -> pd.Series:
    cols = ["run_id"]
    if "fold" in df.columns:
        cols.append("fold")
    if "seed" in df.columns:
        cols.append("seed")
    return df[cols].astype(str).agg("_".join, axis=1)


def _selection_key_columns(df: pd.DataFrame) -> list[str]:
    keys = [c for c in ["run_id", "fold", "seed"] if c in df.columns]
    if keys:
        return keys
    if "exp_id" in df.columns:
        return ["exp_id"]
    raise ValueError("Cannot build selection groups")

def _safe_parse_embedding(s: str) -> dict:
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}

# ---------------------------------------------------------------------
# Standard naming
# ---------------------------------------------------------------------
def _extract_model_embedding(row: pd.Series) -> tuple[str, str]:
    model = row.get("model_type", row.get("model", "unknown"))
    emb_raw = row.get("embedding", "")
    emb_dict = _safe_parse_embedding(emb_raw)

    # -----------------------------
    # CASE 1: PointNet-like models
    # -----------------------------
    if "region_encoder" in emb_dict:
        encoder = emb_dict.get("region_encoder", {})
        enc_type = encoder.get("type", "unknown")
        include_size = encoder.get("include_size", False)

        if include_size:
            embedding = "pointnet_size"
        else:
            embedding = f"pointnet_{enc_type}"

        # force model name
        model = "region_group_lasso"

        return model, embedding

    # -----------------------------
    # CASE 2: PCA model
    # -----------------------------
    if model == "region_pca":
        return "region_group_lasso", "pca"

    # -----------------------------
    # CASE 3: Classical embeddings
    # -----------------------------
    if "region_representation" in emb_dict:
        embedding = emb_dict["region_representation"]
    else:
        embedding = "unknown"

    return model, embedding

def _add_model_embedding_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    parsed = out.apply(_extract_model_embedding, axis=1)
    out["model_name"] = [p[0] for p in parsed]
    out["embedding_name"] = [p[1] for p in parsed]

    return out


def _exclude_region_permutation_models(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["model", "model_type", "model_name"] if c in df.columns]
    if not cols:
        return df
    mask = np.ones(len(df), dtype=bool)
    for col in cols:
        mask &= ~df[col].astype(str).str.contains("region_permutation", case=False, na=False)
    return df.loc[mask].copy()

# ---------------------------------------------------------------------
# Compute maps
# ---------------------------------------------------------------------
def _compute_binomial_maps(df: pd.DataFrame, extra_group_cols: list[str]) -> pd.DataFrame:
    group_cols = ["dataset", "task"] + extra_group_cols + ["region"]

    stats = (
        df.groupby(group_cols)
        .apply(_selection_stats_binomial)
        .reset_index()
    )

    return stats

# ---------------------------------------------------------------------
# Run filtering (unchanged)
# ---------------------------------------------------------------------

def _select_best_runs(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    score_df = (
        df[["run_id", "test_score"] + group_cols]
        .dropna(subset=["test_score"])
        .drop_duplicates()
    )

    run_scores = (
        score_df.groupby(["run_id"] + group_cols, as_index=False)["test_score"]
        .mean()
        .rename(columns={"test_score": "mean_test_score"})
    )

    def _filter(group: pd.DataFrame):
        best = group["mean_test_score"].max()
        return group[group["mean_test_score"] >= best * 0.9]
    return run_scores.groupby(group_cols).apply(_filter).reset_index(drop=True)

# ---------------------------------------------------------------------
# Run filtering fold level
# ---------------------------------------------------------------------

def _select_best_experiments(df: pd.DataFrame, group_cols: list[str], outlier_iqr_mult: float = 1.5) -> pd.DataFrame:
    score_df = (
        df[["exp_id", "test_score"] + group_cols]
        .dropna(subset=["test_score"])
        .drop_duplicates()
    )

    # NO averaging anymore → one score per exp_id
    exp_scores = score_df.rename(columns={"test_score": "score"})

    if exp_scores.empty:
        return exp_scores

    def _filter_group(group: pd.DataFrame) -> pd.DataFrame:
        out = group.copy()
        scores = out["score"].astype(float)

        best_score = float(scores.max())
        gaps = best_score - scores

        # IQR filtering
        q1_gap = float(gaps.quantile(0.25))
        q3_gap = float(gaps.quantile(0.75))
        iqr_gap = q3_gap - q1_gap
        gap_cutoff = q3_gap + outlier_iqr_mult * iqr_gap if iqr_gap > 0 else q3_gap

        q1_score = float(scores.quantile(0.25))
        q3_score = float(scores.quantile(0.75))
        iqr_score = q3_score - q1_score
        # score_floor = q1_score - outlier_iqr_mult * iqr_score if iqr_score > 0 else q1_score
        score_floor = q1_score #- outlier_iqr_mult * iqr_score if iqr_score > 0 else q1_score

        keep_mask = (gaps <= gap_cutoff) & (scores >= score_floor)
        keep_mask = keep_mask | (scores == best_score)

        if keep_mask.sum() == 0:
            keep_mask = scores == best_score

        out["best_test_score"] = best_score
        out["score_gap_from_best"] = gaps

        return out.loc[keep_mask.values]

    grouped = exp_scores.groupby(group_cols, dropna=False, as_index=False)

    try:
        selected = grouped.apply(_filter_group, include_groups=False).reset_index(drop=True)
    except TypeError:
        selected = grouped.apply(_filter_group).reset_index(drop=True)

    return selected

# ---------------------------------------------------------------------
# Selection (binary only)
# ---------------------------------------------------------------------

def _add_percentile_selection(df: pd.DataFrame, percentile: float = 0.90) -> pd.DataFrame:
    group_cols = _selection_key_columns(df)
    out = df.copy()

    out["selected"] = out.groupby(group_cols)["coef"].transform(
        lambda s: (s.abs() >= s.abs().quantile(percentile)).astype(int)
    )
    return out


# ---------------------------------------------------------------------
# De Moivre–Laplace statistics
# ---------------------------------------------------------------------

def _selection_stats_binomial(group: pd.DataFrame) -> pd.Series:
    s = group["selected"].astype(int)

    N = len(s)
    if N == 0:
        return pd.Series({
            "selection_freq": 0.0,
            "selection_std": 0.0,
            "selection_ci_low": 0.0,
            "selection_ci_high": 0.0,
            "n_runs": 0,
        })

    p = float(s.mean())

    var = p * (1 - p) / N
    std = np.sqrt(var)

    ci_low = max(0.0, p - 1.96 * std)
    ci_high = min(1.0, p + 1.96 * std)

    return pd.Series({
        "selection_freq": p,
        "selection_std": std,
        "selection_ci_low": ci_low,
        "selection_ci_high": ci_high,
        "n_runs": N,
    })


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def _label_value_map(values_by_region: pd.Series) -> dict[int, float]:
    out = {}
    for region, value in values_by_region.items():
        try:
            label = int(str(region).split(":")[-1])
            out[label] = float(value)
        except Exception:
            continue
    return out


def _surface_texture_from_label_map(surface_atlas, label_values):
    labels = np.asarray(surface_atlas["parcel_labels"]).astype(int)
    n_left = int(surface_atlas["n_left_vertices"])

    texture = np.zeros(len(labels))
    for label, value in label_values.items():
        texture[labels == label] = value

    return texture[:n_left], texture[n_left:]


def _plot_surface_metric(surface_atlas, label_values, title, out_file, vmin, vmax):
    from nilearn import plotting

    left_mesh = surface_atlas["left_mesh"]
    right_mesh = surface_atlas["right_mesh"]

    tex_left, tex_right = _surface_texture_from_label_map(surface_atlas, label_values)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    plotting.plot_surf_stat_map(left_mesh, tex_left, hemi="left", axes=ax1, vmin=vmin, vmax=vmax, cmap="Reds")
    plotting.plot_surf_stat_map(right_mesh, tex_right, hemi="right", axes=ax2, vmin=vmin, vmax=vmax, cmap="Reds")

    fig.suptitle(title)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def _load_label_name_map() -> dict[int, str]:
    if not FS_LABELS_JSON.exists():
        return {}
    try:
        raw = json.loads(FS_LABELS_JSON.read_text())
    except Exception:
        return {}
    out: dict[int, str] = {}
    for name, idx in raw.items():
        try:
            out[int(idx)] = str(name)
        except Exception:
            continue
    return out


def _region_color_lookup(region_ids: list[int]) -> dict[int, tuple[float, float, float, float]]:
    # Fallback palette when adjacency cannot be computed.
    if not region_ids:
        return {}
    warm = [plt.get_cmap("YlOrRd")(x) for x in np.linspace(0.35, 0.95, max(4, len(region_ids) // 2 + 1))]
    cool = [plt.get_cmap("YlGnBu")(x) for x in np.linspace(0.35, 0.95, max(4, len(region_ids) // 2 + 1))]
    palette = []
    for i in range(max(len(warm), len(cool))):
        if i < len(warm):
            palette.append(warm[i])
        if i < len(cool):
            palette.append(cool[i])
    return {rid: palette[i % len(palette)] for i, rid in enumerate(region_ids)}


def _build_region_adjacency(surface_atlas, region_ids: list[int]) -> dict[int, set[int]]:
    region_set = set(region_ids)
    adjacency: dict[int, set[int]] = {rid: set() for rid in region_ids}

    parcel_labels = np.asarray(surface_atlas["parcel_labels"]).astype(int)
    n_left = int(surface_atlas["n_left_vertices"])
    left_faces = np.asarray(surface_atlas["left_mesh"][1]).astype(int)
    right_faces = np.asarray(surface_atlas["right_mesh"][1]).astype(int) + n_left

    for faces in (left_faces, right_faces):
        for tri in faces:
            tri_labels = [int(parcel_labels[int(v)]) for v in tri]
            uniq = [lab for lab in set(tri_labels) if lab in region_set]
            if len(uniq) < 2:
                continue
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = uniq[i], uniq[j]
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    return adjacency


def _rgb_distance(c1, c2) -> float:
    v1 = np.asarray(c1[:3], dtype=float)
    v2 = np.asarray(c2[:3], dtype=float)
    return float(np.linalg.norm(v1 - v2))


def _contrast_color_assignment(region_ids: list[int], adjacency: dict[int, set[int]]) -> dict[int, tuple[float, float, float, float]]:
    if not region_ids:
        return {}

    warm = [plt.get_cmap("YlOrRd")(x) for x in np.linspace(0.35, 0.98, max(6, len(region_ids) // 2 + 2))]
    cool = [plt.get_cmap("YlGnBu")(x) for x in np.linspace(0.35, 0.98, max(6, len(region_ids) // 2 + 2))]
    palette: list[tuple[float, float, float, float]] = []
    for i in range(max(len(warm), len(cool))):
        if i < len(warm):
            palette.append(warm[i])
        if i < len(cool):
            palette.append(cool[i])

    ordering = sorted(region_ids, key=lambda r: len(adjacency.get(r, set())), reverse=True)
    assigned: dict[int, tuple[float, float, float, float]] = {}

    for rid in ordering:
        neighbor_colors = [assigned[n] for n in adjacency.get(rid, set()) if n in assigned]
        if not neighbor_colors:
            assigned[rid] = palette[len(assigned) % len(palette)]
            continue

        best_color = palette[0]
        best_score = -1.0
        for cand in palette:
            min_dist = min(_rgb_distance(cand, nb) for nb in neighbor_colors)
            if min_dist > best_score:
                best_score = min_dist
                best_color = cand
        assigned[rid] = best_color

    return assigned


def _plot_region_index_legend_map(surface_atlas, region_ids: list[int], title: str, out_file: Path) -> None:
    from nilearn import plotting

    region_ids = sorted({int(r) for r in region_ids})
    if not region_ids:
        return

    label_name_map = _load_label_name_map()
    adjacency = _build_region_adjacency(surface_atlas, region_ids)
    color_lookup = _contrast_color_assignment(region_ids, adjacency)
    if not color_lookup:
        color_lookup = _region_color_lookup(region_ids)
    id_to_idx = {rid: idx for idx, rid in enumerate(region_ids)}
    listed_cmap = ListedColormap([color_lookup[rid] for rid in region_ids])

    parcel_labels = np.asarray(surface_atlas["parcel_labels"]).astype(int)
    n_left = int(surface_atlas["n_left_vertices"])
    left_mesh = surface_atlas["left_mesh"]
    right_mesh = surface_atlas["right_mesh"]

    texture_idx = np.full(len(parcel_labels), np.nan, dtype=float)
    for rid, idx in id_to_idx.items():
        texture_idx[parcel_labels == rid] = float(idx)
    tex_left = texture_idx[:n_left]
    tex_right = texture_idx[n_left:]

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.6])
    ax_left = fig.add_subplot(gs[0, 0], projection="3d")
    ax_right = fig.add_subplot(gs[0, 1], projection="3d")
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis("off")

    plotting.plot_surf_stat_map(
        left_mesh,
        tex_left,
        hemi="left",
        view="lateral",
        cmap=listed_cmap,
        vmin=0,
        vmax=max(len(region_ids) - 1, 1),
        colorbar=False,
        axes=ax_left,
        title="Left",
    )
    plotting.plot_surf_stat_map(
        right_mesh,
        tex_right,
        hemi="right",
        view="lateral",
        cmap=listed_cmap,
        vmin=0,
        vmax=max(len(region_ids) - 1, 1),
        colorbar=False,
        axes=ax_right,
        title="Right",
    )

    # Annotate each parcel with its numeric region id directly on the mesh.
    left_coords = np.asarray(left_mesh[0])
    right_coords = np.asarray(right_mesh[0])
    left_labels = parcel_labels[:n_left]
    right_labels = parcel_labels[n_left:]

    for rid in region_ids:
        lm = left_labels == rid
        if np.any(lm):
            c = left_coords[lm].mean(axis=0)
            ax_left.text(float(c[0]), float(c[1]), float(c[2]), str(rid), fontsize=6, color="black")
        rm = right_labels == rid
        if np.any(rm):
            c = right_coords[rm].mean(axis=0)
            ax_right.text(float(c[0]), float(c[1]), float(c[2]), str(rid), fontsize=6, color="black")

    n_cols = 4
    n_rows = int(np.ceil(len(region_ids) / n_cols))
    col_x = np.linspace(0.02, 0.77, n_cols)
    row_step = 0.95 / max(1, n_rows)

    for i, rid in enumerate(region_ids):
        col = i % n_cols
        row = i // n_cols
        x = float(col_x[col])
        y = float(0.98 - row * row_step)
        color = color_lookup[rid]
        label_name = label_name_map.get(rid, "unknown")

        ax_legend.add_patch(
            Rectangle(
                (x, y - 0.022),
                0.018,
                0.018,
                transform=ax_legend.transAxes,
                facecolor=color,
                edgecolor="black",
                linewidth=0.2,
            )
        )
        ax_legend.text(
            x + 0.024,
            y - 0.004,
            f"{rid}: {label_name}",
            fontsize=8,
            ha="left",
            va="top",
            transform=ax_legend.transAxes,
        )

    fig.suptitle(title)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

# def build_binomial_selection_maps():
#     df = pd.read_parquet(INPUT_TABLE)
#     df["exp_id"] = _build_exp_id(df)

#     group_cols = _pick_group_columns(df)
#     breakpoint()
#     # best_runs = _select_best_runs(df, group_cols)
#     best_runs = _select_best_experiments(df, group_cols)
    
#     # df = df[df["run_id"].isin(best_runs["run_id"])]
#     df = df[df["exp_id"].isin(best_runs["exp_id"])].copy()

#     df = _add_percentile_selection(df)

#     # Compute binomial stats
#     stats = (
#         df.groupby(["dataset", "task", "region"])
#         .apply(_selection_stats_binomial)
#         .reset_index()
#     )

#     STABILITY_TABLE.parent.mkdir(parents=True, exist_ok=True)
#     stats.to_parquet(STABILITY_TABLE, index=False)

#     # Plot
#     for (dataset, task), combo in stats.groupby(["dataset", "task"]):

#         run_id = df[(df["dataset"] == dataset) & (df["task"] == task)]["run_id"].iloc[0]

#         atlas = load_atlas_from_run(run_id)

#         values = _label_value_map(combo.set_index("region")["selection_freq"])
#         std_vals = _label_value_map(combo.set_index("region")["selection_std"])

#         _plot_surface_metric(
#             atlas,
#             values,
#             f"Selection Probability | {dataset} | {task}",
#             MAPS_DIR / f"selection_prob_{dataset}_{task}.png",
#             0,
#             1,
#         )

#         _plot_surface_metric(
#             atlas,
#             std_vals,
#             f"Selection Std | {dataset} | {task}",
#             MAPS_DIR / f"selection_std_{dataset}_{task}.png",
#             0,
#             max(std_vals.values()) if std_vals else 1e-6,
#         )

#     return stats

def build_binomial_selection_maps():
    df = pd.read_parquet(INPUT_TABLE)
    df = _exclude_region_permutation_models(df)
    df["exp_id"] = _build_exp_id(df)

    # --------------------------------------------------
    # ADD: normalize model / embedding
    # --------------------------------------------------
    df = _add_model_embedding_columns(df)

    group_cols = _pick_group_columns(df)

    best_runs = _select_best_experiments(df, group_cols)

    df = df[df["exp_id"].isin(best_runs["exp_id"])].copy()

    df = _add_percentile_selection(df)
    
    # --------------------------------------------------
    # Helper to compute + save + plot
    # --------------------------------------------------
    def _run_analysis(df, extra_group_cols, prefix):
        group_cols_full = ["dataset", "task"] + extra_group_cols + ["region"]

        stats = (
            df.groupby(group_cols_full)
            .apply(_selection_stats_binomial)
            .reset_index()
        )

        # Save table
        out_table = STABILITY_TABLE.with_name(f"{prefix}_binomial_metrics.parquet")
        stats.to_parquet(out_table, index=False)

        # Plot
        grouping = ["dataset", "task"] + extra_group_cols
        i = 0
        for keys, combo in stats.groupby(grouping):
            if not isinstance(keys, tuple):
                keys = (keys,)

            dataset, task = keys[:2]
            extra_vals = keys[2:]

            # build readable name
            extra_str = "_".join(map(str, extra_vals)) if extra_vals else "global"

            run_id = df[
                (df["dataset"] == dataset) & (df["task"] == task)
            ]["run_id"].iloc[0]

            atlas = load_atlas_from_run(run_id)

            values = _label_value_map(combo.set_index("region")["selection_freq"])
            std_vals = _label_value_map(combo.set_index("region")["selection_std"])

            # Frequency map
            _plot_surface_metric(
                atlas,
                values,
                f"{prefix} | {extra_str} | {dataset} | {task}",
                MAPS_DIR / f"{prefix}_prob_{extra_str}_{dataset}_{task}.png",
                0,
                1,
            )

            # Std map
            _plot_surface_metric(
                atlas,
                std_vals,
                f"{prefix} STD | {extra_str} | {dataset} | {task}",
                MAPS_DIR / f"{prefix}_std_{extra_str}_{dataset}_{task}.png",
                0,
                max(std_vals.values()) if std_vals else 1e-6,
            )
            if i == 0:
                # Independent atlas legend map with region ids on-surface + colored legend.
                _plot_region_index_legend_map(
                    atlas,
                    sorted(values.keys()),
                    f"{prefix} REGION LEGEND | {extra_str} | {dataset} | {task}",
                    MAPS_DIR / f"_region_legend.png",
                )
                i = 1

        return stats

    # --------------------------------------------------
    # 1. ORIGINAL (global)
    # --------------------------------------------------
    stats_global = _run_analysis(df, [], "global")

    # --------------------------------------------------
    # 2. PER MODEL
    # --------------------------------------------------
    stats_model = _run_analysis(df, ["model_name"], "model")

    # --------------------------------------------------
    # 3. PER EMBEDDING
    # --------------------------------------------------
    stats_embedding = _run_analysis(df, ["embedding_name"], "embedding")

    # --------------------------------------------------
    # 4. PER MODEL + EMBEDDING
    # --------------------------------------------------
    stats_model_embedding = _run_analysis(
        df,
        ["model_name", "embedding_name"],
        "model_embedding",
    )

    return {
        "global": stats_global,
        "model": stats_model,
        "embedding": stats_embedding,
        "model_embedding": stats_model_embedding,
    }


if __name__ == "__main__":
    stats = build_binomial_selection_maps()
    print(f"Saved results: {STABILITY_TABLE}")