from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from diff_benchmark.preprocessing.utils.utils_brain_feature_extraction import (
    build_parcel_label_vector,
    load_template_surface,
    resample_schaefer_onto_fs_lr,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "exp_outputs" / "summary" / "schaefer_networks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE = 100
SURFACE_SPACE = "fslr_32k"

logger = logging.getLogger(__name__)


def _sort_by_suffix(label: str) -> int:
    match = re.search(r"_(\d+)$", label)
    return int(match.group(1)) if match else 0


def _load_label_map(scale: int, surface_space: str) -> dict[int, str]:
    schaefer = resample_schaefer_onto_fs_lr(scale=scale, target_space=surface_space)
    tsv_path = Path(schaefer["atlas_meta"]["label_tsv_path"])
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[~df["name"].str.contains("Background", na=False)].copy()

    left_ids = np.unique(schaefer["left.data"])
    right_ids = np.unique(schaefer["right.data"])

    left_ids = np.sort(left_ids[left_ids != 0])
    right_ids = np.sort(right_ids[right_ids != 0])

    left_names = df[df["name"].str.startswith("LH_")]["name"].tolist()
    right_names = df[df["name"].str.startswith("RH_")]["name"].tolist()

    left_names = sorted(left_names, key=_sort_by_suffix)
    right_names = sorted(right_names, key=_sort_by_suffix)

    if len(left_ids) + len(right_ids) != len(df):
        logger.warning("Mismatch between atlas parcels and TSV labels.")

    label_map = {int(pid): str(name) for pid, name in zip(left_ids, left_names)}
    offset = int(left_ids.max()) if len(left_ids) > 0 else 0
    label_map.update(
        {int(pid + offset): str(name) for pid, name in zip(right_ids, right_names)}
    )
    return label_map


def _extract_network_name(label_name: str) -> str:
    name = label_name.replace("LH_", "").replace("RH_", "")
    name = name.replace("17Networks_", "")
    parts = name.split("_")
    return parts[0] if parts else name


def _build_network_texture(
    parcel_labels: np.ndarray,
    label_map: dict[int, str],
) -> tuple[np.ndarray, list[str]]:
    parcel_ids = sorted(label_map.keys())
    network_names = sorted({_extract_network_name(label_map[pid]) for pid in parcel_ids})
    network_to_id = {name: idx for idx, name in enumerate(network_names)}

    texture = np.zeros(parcel_labels.shape[0], dtype=float)
    for pid in parcel_ids:
        net_name = _extract_network_name(label_map[pid])
        texture[parcel_labels == int(pid)] = float(network_to_id[net_name])

    return texture, network_names


def _plot_networks(
    *,
    texture: np.ndarray,
    n_left: int,
    network_names: list[str],
    surface_space: str,
    out_file: Path,
) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from nilearn import plotting

    left_mesh = load_template_surface(
        hemi="L", space=surface_space, surf_type="midthickness"
    )
    right_mesh = load_template_surface(
        hemi="R", space=surface_space, surf_type="midthickness"
    )

    n_clusters = len(network_names)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    cmap = ListedColormap(colors)

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    plotting.plot_surf_stat_map(
        left_mesh,
        texture[:n_left],
        hemi="left",
        cmap=cmap,
        colorbar=False,
        vmin=0,
        vmax=max(n_clusters - 1, 1),
        axes=ax1,
        title="Left",
    )
    plotting.plot_surf_stat_map(
        right_mesh,
        texture[n_left:],
        hemi="right",
        cmap=cmap,
        colorbar=False,
        vmin=0,
        vmax=max(n_clusters - 1, 1),
        axes=ax2,
        title="Right",
    )

    legend_handles = [
        Patch(color=colors[i], label=network_names[i])
        for i in range(n_clusters)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(6, max(n_clusters, 1)),
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle("Schaefer 17 networks (scale=100)")
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    schaefer = resample_schaefer_onto_fs_lr(scale=SCALE, target_space=SURFACE_SPACE)
    left_vertices, _ = load_template_surface(
        hemi="L", space=SURFACE_SPACE, surf_type="midthickness"
    )
    right_vertices, _ = load_template_surface(
        hemi="R", space=SURFACE_SPACE, surf_type="midthickness"
    )

    parcel_labels = build_parcel_label_vector(
        schaefer,
        n_left=left_vertices.shape[0],
        n_right=right_vertices.shape[0],
    )
    breakpoint()
    label_map = _load_label_map(SCALE, SURFACE_SPACE)
    texture, network_names = _build_network_texture(parcel_labels, label_map)

    out_file = OUTPUT_DIR / f"schaefer17_networks_scale{SCALE}.png"
    _plot_networks(
        texture=texture,
        n_left=left_vertices.shape[0],
        network_names=network_names,
        surface_space=SURFACE_SPACE,
        out_file=out_file,
    )


if __name__ == "__main__":
    main()
