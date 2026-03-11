"""Graph export utilities for surface mesh data.

This module serialises :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData`
objects to Apache Parquet files and provides the inverse loader that
reconstructs a :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` from
those files.

Export format
-------------
For each subject two files are written inside *output_dir*:

``{subject_id}_nodes.parquet``
    One row per vertex.  Columns:

    ============  ============================================================
    subject_id    Subject identifier string.
    node_id       0-based vertex index (int32).
    x, y, z       Vertex coordinates in mm (float32).
    parcel_label  Schaefer parcel ID (int32); 0 = medial wall / unlabelled.
                  Right-hemisphere labels are offset by the maximum LH label
                  so all parcel IDs are globally unique across both hemispheres.
    hemisphere    Hemisphere indicator (int8): 0 = left, 1 = right.
    feature_0 …   Per-vertex microstructure values (float32).  The column
                  name suffix matches the column index in ``features``.
    ============  ============================================================

``{subject_id}_edges.parquet``
    One row per undirected edge.  Columns:

    ===  ===================================================================
    src  Source vertex index (int32).
    dst  Destination vertex index (int32).  Always ``src < dst``.
    ===  ===================================================================

    Edges are derived from mesh faces: for each triangle ``(a, b, c)`` the
    three half-edges ``(a,b)``, ``(b,c)``, ``(a,c)`` are generated, sorted so
    that ``src < dst``, deduplicated, and written as a canonical undirected
    edge list.

Design notes
------------
- Everything is derived from :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData`
  — no dataset-specific paths are referenced.
- Per-vertex arrays are validated before serialisation.
- Reading back with :func:`load_graph_from_parquet` reconstructs a
  :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` that is
  numerically identical to the original (within float32 precision).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

# Mapping from full atlas names to their short prefix used in filenames.
_ATLAS_PREFIXES: dict[str, str] = {
    "schaefer": "scha",
    "scha": "scha",
}


def build_mesh_stem(
    subject_id: str,
    metric: str,
    tissue_type: str,
    atlas_name: str,
    n_parcels: int,
) -> str:
    """Return the BIDS-style filename stem for mesh output files.

    The stem follows the pattern::

        sub-<subject_id>_param-<metric>_tissue-<tissue_type>_atlas-<prefix><n_parcels>

    where ``<prefix>`` is the short identifier for *atlas_name* (e.g.
    ``"scha"`` for Schaefer).

    Args:
        subject_id: Subject identifier (without ``sub-`` prefix).
        metric: Microstructure metric (e.g. ``"md"``, ``"ndi"``).
        tissue_type: Tissue type (e.g. ``"gray"``, ``"white"``).
        atlas_name: Full or short atlas name (e.g. ``"schaefer"`` or
            ``"scha"``).  Case-insensitive.
        n_parcels: Number of parcels in the atlas (e.g. ``1000``).

    Returns:
        Filename stem string, e.g.
        ``"sub-100206_param-md_tissue-gray_atlas-scha1000"``.

    Raises:
        ValueError: If *atlas_name* is not recognised.
    """
    prefix = _ATLAS_PREFIXES.get(atlas_name.lower())
    if prefix is None:
        raise ValueError(
            f"Unknown atlas_name '{atlas_name}'. "
            f"Recognised names: {list(_ATLAS_PREFIXES)}"
        )
    return (
        f"sub-{subject_id}_param-{metric}_tissue-{tissue_type}"
        f"_atlas-{prefix}{n_parcels}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_mesh_graph(
    mesh: "SurfaceMeshData",  # noqa: F821 — forward ref; avoid circular import
    subject_id: str,
    output_dir: Path,
    *,
    metric: str,
    tissue_type: str,
    atlas_name: str = "schaefer",
    n_parcels: int,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Serialise a :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` to Parquet.

    Writes BIDS-named files inside *output_dir*:

    - ``sub-<id>_param-<metric>_tissue-<tissue_type>_atlas-<prefix><n_parcels>_nodes.parquet``
    - ``sub-<id>_param-<metric>_tissue-<tissue_type>_atlas-<prefix><n_parcels>_edges.parquet``

    Skips serialisation (returns existing paths) when both files already exist
    and *overwrite* is ``False``.

    Args:
        mesh: Surface mesh object produced by
            :class:`~diff_benchmark.preprocessing.brain_feature_extraction.MeshPipeline`.
        subject_id: Subject identifier (used for file naming).
        output_dir: Directory where the Parquet files are written.  Created
            automatically if it does not exist.
        metric: Microstructure metric name (e.g. ``"ndi"``).
        tissue_type: Tissue type (e.g. ``"white"`` or ``"gray"``).
        atlas_name: Atlas name used for the filename identifier (e.g.
            ``"schaefer"``).  Defaults to ``"schaefer"``.
        n_parcels: Number of parcels in the atlas (e.g. ``1000``).
        overwrite: When ``True``, re-export even if the files already exist.

    Returns:
        ``(nodes_path, edges_path)`` — absolute :class:`Path` objects.

    Raises:
        ValueError: If the mesh arrays have inconsistent shapes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = build_mesh_stem(subject_id, metric, tissue_type, atlas_name, n_parcels)
    nodes_path = output_dir / f"{stem}_nodes.parquet"
    edges_path = output_dir / f"{stem}_edges.parquet"

    if nodes_path.exists() and edges_path.exists() and not overwrite:
        logger.debug(
            "[%s] Graph parquet files already exist — skipping export", subject_id
        )
        return nodes_path, edges_path

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    vertices = mesh.vertices      # (N, 3) float32
    faces = mesh.faces            # (M, 3) int32
    features = mesh.features      # (N, F) float32
    parcel_labels = mesh.parcel_labels  # (N,) int32

    n_nodes = vertices.shape[0]

    assert vertices.shape[0] == parcel_labels.shape[0], (
        f"[{subject_id}] vertices.shape[0]={vertices.shape[0]} != "
        f"parcel_labels.shape[0]={parcel_labels.shape[0]}"
    )
    assert vertices.shape[0] == features.shape[0], (
        f"[{subject_id}] vertices.shape[0]={vertices.shape[0]} != "
        f"features.shape[0]={features.shape[0]}"
    )

    # ------------------------------------------------------------------
    # Node DataFrame
    # ------------------------------------------------------------------
    n_feat = features.shape[1] if features.ndim == 2 else 1
    feat_2d = features if features.ndim == 2 else features[:, np.newaxis]

    # Build hemisphere indicator: 0 = LH, 1 = RH.
    # Use n_left_vertices stored on the mesh when available; fall back to
    # splitting at the midpoint for legacy objects that lack the attribute.
    n_left = getattr(mesh, "n_left_vertices", None)
    if n_left is None:
        n_left = n_nodes // 2
    hemi_col = np.zeros(n_nodes, dtype=np.int8)
    hemi_col[n_left:] = 1  # RH vertices

    node_dict: dict = {
        "subject_id": np.full(n_nodes, subject_id, dtype=object),
        "node_id": np.arange(n_nodes, dtype=np.int32),
        "x": vertices[:, 0],
        "y": vertices[:, 1],
        "z": vertices[:, 2],
        "parcel_label": parcel_labels.astype(np.int32),
        "hemisphere": hemi_col,
    }
    for f_idx in range(n_feat):
        node_dict[f"feature_{f_idx}"] = feat_2d[:, f_idx].astype(np.float32)

    nodes_df = pd.DataFrame(node_dict)
    nodes_df.to_parquet(nodes_path, index=False, engine="pyarrow")
    logger.info(
        "[%s] Wrote %d nodes → %s", subject_id, n_nodes, nodes_path
    )

    # ------------------------------------------------------------------
    # Edge DataFrame  (undirected, deduplicated)
    # ------------------------------------------------------------------
    edges_path_result = _export_edges(faces, n_nodes, subject_id, edges_path)

    return nodes_path, edges_path_result


def mesh_parquet_paths(
    subject_id: str,
    graph_dir: Path,
    metric: str,
    tissue_type: str,
    atlas_name: str = "schaefer",
    n_parcels: int = 1000,
) -> tuple[Path, Path]:
    """Return the expected BIDS-named Parquet paths for *subject_id*.

    Args:
        subject_id: Subject identifier.
        graph_dir: Directory containing the Parquet files.
        metric: Microstructure metric name.
        tissue_type: Tissue type.
        atlas_name: Atlas name (e.g. ``"schaefer"``).  Defaults to
            ``"schaefer"``.
        n_parcels: Number of parcels (e.g. ``1000``).  Defaults to ``1000``.

    Returns:
        ``(nodes_path, edges_path)`` as :class:`Path` objects.
    """
    graph_dir = Path(graph_dir)
    stem = build_mesh_stem(subject_id, metric, tissue_type, atlas_name, n_parcels)
    return graph_dir / f"{stem}_nodes.parquet", graph_dir / f"{stem}_edges.parquet"


def load_graph_from_parquet(
    subject_id: str,
    graph_dir: Path,
    metric: str,
    tissue_type: str,
    atlas_name: str = "schaefer",
    n_parcels: int = 1000,
) -> "SurfaceMeshData":  # noqa: F821
    """Reconstruct a :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` from Parquet files.

    Reads BIDS-named Parquet files:

    - ``sub-<id>_param-<metric>_tissue-<tissue_type>_atlas-<prefix><n_parcels>_nodes.parquet``
    - ``sub-<id>_param-<metric>_tissue-<tissue_type>_atlas-<prefix><n_parcels>_edges.parquet``

    Because the edge list is stored without face connectivity, the ``faces``
    array is set to an empty ``(0, 3)`` array (faces are not needed for GNN or
    visualisation use-cases that only consume the node table and edge list).

    Args:
        subject_id: Subject identifier used for file naming.
        graph_dir: Directory containing the Parquet files.
        metric: Microstructure metric name used in the filename.
        tissue_type: Tissue type used in the filename.
        atlas_name: Atlas name (e.g. ``"schaefer"``).  Defaults to
            ``"schaefer"``.
        n_parcels: Number of parcels (e.g. ``1000``).  Defaults to ``1000``.

    Returns:
        :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` with
        ``faces`` set to an empty array.

    Raises:
        FileNotFoundError: If either Parquet file is missing.
    """
    # Lazy import to avoid circular dependency at module load time
    from diff_benchmark.data.surface_mesh import SurfaceMeshData

    nodes_path, edges_path = mesh_parquet_paths(
        subject_id, graph_dir, metric, tissue_type, atlas_name, n_parcels
    )

    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")

    nodes_df = pd.read_parquet(nodes_path, engine="pyarrow")
    edges_df = pd.read_parquet(edges_path, engine="pyarrow")

    # Vertices
    vertices = nodes_df[["x", "y", "z"]].to_numpy(dtype=np.float32)

    # Parcel labels
    parcel_labels = nodes_df["parcel_label"].to_numpy(dtype=np.int32)

    # Hemisphere indicator (0=LH, 1=RH) — present in newly-exported files;
    # fall back to a midpoint split for legacy parquets that lack the column.
    if "hemisphere" in nodes_df.columns:
        hemi_col = nodes_df["hemisphere"].to_numpy(dtype=np.int8)
        n_left = int(np.sum(hemi_col == 0))
    else:
        n_left = len(nodes_df) // 2
        hemi_col = np.zeros(len(nodes_df), dtype=np.int8)
        hemi_col[n_left:] = 1

    # Features — all columns that start with "feature_"
    feat_cols = sorted(
        [c for c in nodes_df.columns if c.startswith("feature_")],
        key=lambda c: int(c.split("_")[1]),
    )
    if feat_cols:
        features = nodes_df[feat_cols].to_numpy(dtype=np.float32)
    else:
        features = np.zeros((len(nodes_df), 0), dtype=np.float32)

    # Faces — not stored; reconstruct a (0, 3) placeholder
    faces = np.zeros((0, 3), dtype=np.int32)

    # Validate edge indices
    max_node = len(nodes_df) - 1
    if len(edges_df) > 0:
        edge_max = int(edges_df[["src", "dst"]].max().max())
        if edge_max > max_node:
            logger.warning(
                "[%s] Edge index %d exceeds node count %d — data may be corrupt",
                subject_id, edge_max, max_node,
            )

    return SurfaceMeshData(
        vertices=vertices,
        faces=faces,
        features=features,
        parcel_labels=parcel_labels,
        subject_id=subject_id,
        metric=None,
        hemisphere="LR",
        n_left_vertices=n_left,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _export_edges(
    faces: np.ndarray,
    n_nodes: int,
    subject_id: str,
    edges_path: Path,
) -> Path:
    """Build an undirected edge list from triangle faces and write it to Parquet.

    For each face ``(a, b, c)`` the three undirected edges
    ``{a,b}``, ``{b,c}``, ``{a,c}`` are emitted.  Duplicates are removed by
    normalising each edge so that ``src < dst`` and then using
    :func:`numpy.unique`.

    The vectorised implementation avoids any Python-level loops and handles
    meshes with ~120 k faces (≈ 60 k-vertex fsLR 32k surface) without
    significant overhead.

    Args:
        faces: ``(M, 3)`` int32 face array.
        n_nodes: Total number of nodes (used for validation only).
        subject_id: Subject identifier (for log messages).
        edges_path: Output file path.

    Returns:
        *edges_path* (unchanged).
    """
    # Generate the three edge pairs from each triangle
    a = faces[:, 0]
    b = faces[:, 1]
    c = faces[:, 2]

    # Raw directed half-edges: (M*3, 2)
    raw_edges = np.concatenate(
        [
            np.stack([a, b], axis=1),
            np.stack([b, c], axis=1),
            np.stack([a, c], axis=1),
        ],
        axis=0,
    ).astype(np.int32)

    # Normalise to canonical form (src < dst)
    raw_edges = np.sort(raw_edges, axis=1)

    # Deduplicate using the packed int64 trick for speed
    packed = raw_edges[:, 0].astype(np.int64) * (n_nodes + 1) + raw_edges[:, 1]
    unique_packed = np.unique(packed)
    src = (unique_packed // (n_nodes + 1)).astype(np.int32)
    dst = (unique_packed % (n_nodes + 1)).astype(np.int32)

    # Validate: no self-loops, all indices in range
    assert np.all(src != dst), f"[{subject_id}] Self-loops found in edge list"
    assert int(src.max()) < n_nodes and int(dst.max()) < n_nodes, (
        f"[{subject_id}] Edge index out of range: "
        f"max={max(int(src.max()), int(dst.max()))}, n_nodes={n_nodes}"
    )

    edges_df = pd.DataFrame({"src": src, "dst": dst})
    edges_df.to_parquet(edges_path, index=False, engine="pyarrow")
    logger.info(
        "[%s] Wrote %d undirected edges → %s", subject_id, len(edges_df), edges_path
    )
    return edges_path
