"""Mesh visualisation utilities for surface graph data.

Two backends are provided:

1. **Plotly** (:func:`plot_mesh_plotly`) — interactive 3-D scatter + line plot
   loaded directly from Parquet files.  Works in any browser / Jupyter
   environment.  An ``updatemenus`` button lets the user toggle node colouring
   between microstructure feature values and parcel labels without reloading.

2. **nilearn** (:func:`plot_mesh_nilearn`) — static / inline matplotlib figure
   via :func:`nilearn.plotting.plot_surf`.  Accepts a
   :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` object directly
   so it can be used immediately after mesh construction without a round-trip
   through Parquet.

Design principles
-----------------
- **One subject at a time** — neither function loads all subjects.
- **Dataset-agnostic** — all information is sourced from
  :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` or the Parquet
  files produced by
  :func:`~diff_benchmark.preprocessing.utils.utils_graph_export.export_mesh_graph`.
- **Minimal new dependencies** — Plotly and nilearn are already transitively
  present in the environment; no new heavyweight packages are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers — shared between both backends
# ---------------------------------------------------------------------------


def _load_nodes_edges(
    subject_id: str, graph_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load node and edge Parquet files for *subject_id*.

    Args:
        subject_id: Subject identifier used for file naming.
        graph_dir: Directory containing the Parquet files.

    Returns:
        ``(nodes_df, edges_df)``.

    Raises:
        FileNotFoundError: If either file is missing.
    """
    graph_dir = Path(graph_dir)
    nodes_path = graph_dir / f"{subject_id}_nodes.parquet"
    edges_path = graph_dir / f"{subject_id}_edges.parquet"

    for p in (nodes_path, edges_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Graph parquet file not found: {p}\n"
                f"Run MeshPipeline.run_analysis() first to generate it."
            )

    nodes_df = pd.read_parquet(nodes_path, engine="pyarrow")
    edges_df = pd.read_parquet(edges_path, engine="pyarrow")
    return nodes_df, edges_df


def _reconstruct_arrays(
    nodes_df: pd.DataFrame, edges_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract numpy arrays from node/edge DataFrames.

    Returns:
        ``(vertices, edges, features, parcel_labels)`` where

        - *vertices*      ``(N, 3)`` float32
        - *edges*         ``(E, 2)`` int32 — undirected edge list
        - *features*      ``(N, F)`` float32
        - *parcel_labels* ``(N,)``   int32
    """
    vertices = nodes_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    parcel_labels = nodes_df["parcel_label"].to_numpy(dtype=np.int32)

    feat_cols = sorted(
        [c for c in nodes_df.columns if c.startswith("feature_")],
        key=lambda c: int(c.split("_")[1]),
    )
    features = nodes_df[feat_cols].to_numpy(dtype=np.float32) if feat_cols else np.zeros((len(nodes_df), 0), dtype=np.float32)

    edges = edges_df[["src", "dst"]].to_numpy(dtype=np.int32)

    # Validation
    assert vertices.shape[0] == parcel_labels.shape[0], (
        f"vertices ({vertices.shape[0]}) and parcel_labels ({parcel_labels.shape[0]}) mismatch"
    )
    assert vertices.shape[0] == features.shape[0], (
        f"vertices ({vertices.shape[0]}) and features ({features.shape[0]}) mismatch"
    )
    if len(edges) > 0:
        max_idx = int(edges.max())
        assert max_idx < vertices.shape[0], (
            f"Edge index {max_idx} >= n_nodes {vertices.shape[0]}"
        )

    return vertices, edges, features, parcel_labels


# ---------------------------------------------------------------------------
# 1. Plotly interactive visualisation
# ---------------------------------------------------------------------------


def plot_mesh_plotly(
    subject_id: str,
    graph_dir: Path,
    *,
    show_edges: bool = False,
    feature_index: int = 0,
    output_html: Optional[Path] = None,
):
    """Render an interactive 3-D cortical mesh with Plotly.

    Loads ``{subject_id}_nodes.parquet`` and (optionally)
    ``{subject_id}_edges.parquet`` from *graph_dir* and builds a
    :class:`plotly.graph_objects.Figure` with:

    - **Nodes** coloured by microstructure feature value (default) or by
      parcel label.
    - **Edges** as thin grey lines (opt-in via *show_edges*; disabled by
      default because plotting ~180 k edges is slow).
    - An interactive **button menu** that toggles node colouring between
      ``"feature"`` and ``"parcel_label"`` without reloading the page.
    - Rich **hover text** showing node_id, parcel_label, feature value, and
      (x, y, z) coordinates.

    Args:
        subject_id: Subject identifier used for file naming.
        graph_dir: Directory containing the Parquet files.
        show_edges: When ``True``, render mesh edges as thin grey lines.
            Disabled by default for performance on ~60 k-vertex meshes.
        feature_index: Which feature column to use for colouring when the mesh
            carries multiple features (0-based).
        output_html: If given, the figure is saved as a standalone HTML file
            at this path *in addition to* being returned.

    Returns:
        :class:`plotly.graph_objects.Figure` (call ``.show()`` to display).

    Raises:
        ImportError: If ``plotly`` is not installed.
        FileNotFoundError: If the Parquet files are missing.
    """
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "plotly is required for plot_mesh_plotly(). "
            "Install it with: pip install plotly"
        ) from exc

    nodes_df, edges_df = _load_nodes_edges(subject_id, graph_dir)
    vertices, edges, features, parcel_labels = _reconstruct_arrays(nodes_df, edges_df)

    n_nodes = vertices.shape[0]
    logger.info(
        "[%s] Plotting %d nodes, %d edges (show_edges=%s)",
        subject_id, n_nodes, len(edges), show_edges,
    )

    # ------------------------------------------------------------------
    # Feature values for colouring
    # ------------------------------------------------------------------
    if features.shape[1] > feature_index:
        feat_vals = features[:, feature_index].astype(float)
        feat_col_name = f"feature_{feature_index}"
    else:
        feat_vals = np.zeros(n_nodes)
        feat_col_name = "feature_0 (missing)"

    # ------------------------------------------------------------------
    # Hover text (computed once, shared between both colour modes)
    # ------------------------------------------------------------------
    hover_text = [
        (
            f"node_id: {i}<br>"
            f"parcel: {int(parcel_labels[i])}<br>"
            f"{feat_col_name}: {feat_vals[i]:.4f}<br>"
            f"x={vertices[i, 0]:.1f}, y={vertices[i, 1]:.1f}, z={vertices[i, 2]:.1f}"
        )
        for i in range(n_nodes)
    ]

    # ------------------------------------------------------------------
    # Node trace — feature colouring (visible by default)
    # ------------------------------------------------------------------
    node_trace_feature = go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=feat_vals,
            colorscale="Viridis",
            colorbar=dict(title=feat_col_name, thickness=15, x=1.02),
            opacity=0.85,
        ),
        text=hover_text,
        hoverinfo="text",
        name=feat_col_name,
        visible=True,
    )

    # ------------------------------------------------------------------
    # Node trace — parcel label colouring (hidden by default)
    # ------------------------------------------------------------------
    node_trace_parcel = go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=parcel_labels.astype(float),
            colorscale="Turbo",
            colorbar=dict(title="parcel_label", thickness=15, x=1.02),
            opacity=0.85,
        ),
        text=hover_text,
        hoverinfo="text",
        name="parcel_label",
        visible=False,
    )

    traces = [node_trace_feature, node_trace_parcel]

    # ------------------------------------------------------------------
    # Edge trace (vectorised — no Python loop)
    # ------------------------------------------------------------------
    if show_edges and len(edges) > 0:
        # Build arrays of (x0, x1, None) triples for all edges at once
        src_idx = edges[:, 0]
        dst_idx = edges[:, 1]

        # Each edge needs 3 points: start, end, None (to lift the pen)
        x_edges = np.empty(len(edges) * 3)
        y_edges = np.empty(len(edges) * 3)
        z_edges = np.empty(len(edges) * 3)

        x_edges[0::3] = vertices[src_idx, 0]
        x_edges[1::3] = vertices[dst_idx, 0]
        x_edges[2::3] = np.nan

        y_edges[0::3] = vertices[src_idx, 1]
        y_edges[1::3] = vertices[dst_idx, 1]
        y_edges[2::3] = np.nan

        z_edges[0::3] = vertices[src_idx, 2]
        z_edges[1::3] = vertices[dst_idx, 2]
        z_edges[2::3] = np.nan

        edge_trace = go.Scatter3d(
            x=x_edges,
            y=y_edges,
            z=z_edges,
            mode="lines",
            line=dict(color="rgba(150,150,150,0.3)", width=0.5),
            hoverinfo="none",
            name="edges",
            visible=True,
        )
        traces.append(edge_trace)
        n_edge_traces = 1
    else:
        n_edge_traces = 0

    # ------------------------------------------------------------------
    # Toggle buttons: "Feature" / "Parcel label"
    # ------------------------------------------------------------------
    # Visibility list: [node_feature, node_parcel, (edge?)]
    edge_vis = [True] * n_edge_traces

    btn_feature = dict(
        label="Feature",
        method="update",
        args=[{"visible": [True, False] + edge_vis}],
    )
    btn_parcel = dict(
        label="Parcel label",
        method="update",
        args=[{"visible": [False, True] + edge_vis}],
    )

    layout = go.Layout(
        title=dict(text=f"Surface mesh — subject {subject_id}", x=0.5),
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
            bgcolor="white",
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                y=1.08,
                showactive=True,
                buttons=[btn_feature, btn_parcel],
            )
        ],
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=60),
        paper_bgcolor="white",
    )

    fig = go.Figure(data=traces, layout=layout)

    if output_html is not None:
        output_html = Path(output_html)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html))
        logger.info("[%s] Saved interactive HTML → %s", subject_id, output_html)

    return fig


# ---------------------------------------------------------------------------
# 2. nilearn static visualisation
# ---------------------------------------------------------------------------


def plot_mesh_nilearn(
    mesh: "SurfaceMeshData",  # noqa: F821
    *,
    mode: Literal["feature", "parcel"] = "feature",
    feature_index: int = 0,
    hemi: str = "left",
    view: str = "lateral",
    colormap: str = "cold_hot",
    bg_on_stat: bool = True,
    title: Optional[str] = None,
    output_file: Optional[Path] = None,
):
    """Render a cortical surface mesh with :func:`nilearn.plotting.plot_surf`.

    Uses the vertex positions, triangular faces, node features, and parcel
    labels already stored in a
    :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` object — no
    extra file loading required.

    Args:
        mesh: Surface mesh object (from
            :class:`~diff_benchmark.preprocessing.brain_feature_extraction.MeshPipeline`
            or :func:`~diff_benchmark.preprocessing.utils.utils_graph_export.load_graph_from_parquet`).
        mode: Colour the surface by microstructure ``"feature"`` values or by
            ``"parcel"`` labels.
        feature_index: Which feature column to use when ``mode="feature"``
            (0-based).
        hemi: Which hemisphere to display — ``"left"`` or ``"right"`` — when
            the mesh carries both hemispheres combined.  The split is inferred
            from the number of vertices (fsLR 32k: 32 492 per hemisphere).
        view: Viewing angle accepted by nilearn (e.g. ``"lateral"``,
            ``"medial"``, ``"dorsal"``).
        colormap: Matplotlib colormap name.
        bg_on_stat: Overlay sulcal depth as background shading.
        title: Optional figure title.  Defaults to
            ``"subject_id — mode"``.
        output_file: If given, save the figure to this path.

    Returns:
        The nilearn display object (call ``.show()`` or save via
        ``output_file``).

    Raises:
        ImportError: If ``nilearn`` is not installed.
        ValueError: If *mode* is not ``"feature"`` or ``"parcel"``.
    """
    try:
        from nilearn import plotting as nplot  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "nilearn is required for plot_mesh_nilearn(). "
            "Install it with: pip install nilearn"
        ) from exc

    if mode not in ("feature", "parcel"):
        raise ValueError(f"mode must be 'feature' or 'parcel', got '{mode}'")

    # ------------------------------------------------------------------
    # Select hemisphere arrays
    # ------------------------------------------------------------------
    vertices = mesh.vertices  # (N, 3) — may be combined L+R
    faces = mesh.faces        # (M, 3)
    n_total = vertices.shape[0]

    # For combined L+R meshes the hemispheres are stored contiguously.
    # fsLR 32k has exactly 32 492 vertices per hemisphere.
    FSLR_32K_PER_HEMI = 32_492
    n_left_guess = FSLR_32K_PER_HEMI if n_total >= 2 * FSLR_32K_PER_HEMI else n_total // 2

    if hemi == "left":
        idx_start, idx_end = 0, n_left_guess
    elif hemi == "right":
        idx_start, idx_end = n_left_guess, n_total
    else:
        raise ValueError(f"hemi must be 'left' or 'right', got '{hemi}'")

    hemi_vertices = vertices[idx_start:idx_end]
    n_hemi = hemi_vertices.shape[0]

    # Filter faces to only those within the selected hemisphere
    mask = (
        (faces[:, 0] >= idx_start) & (faces[:, 0] < idx_end) &
        (faces[:, 1] >= idx_start) & (faces[:, 1] < idx_end) &
        (faces[:, 2] >= idx_start) & (faces[:, 2] < idx_end)
    )
    hemi_faces = faces[mask] - idx_start  # re-index to 0-based within hemi

    # ------------------------------------------------------------------
    # Stat map (colour values)
    # ------------------------------------------------------------------
    if mode == "feature":
        if mesh.features.shape[1] > feature_index:
            stat_map = mesh.features[idx_start:idx_end, feature_index].astype(float)
        else:
            logger.warning(
                "feature_index=%d out of range (n_features=%d), using zeros",
                feature_index, mesh.features.shape[1],
            )
            stat_map = np.zeros(n_hemi)
        cbar_label = f"feature_{feature_index}"
    else:  # parcel
        stat_map = mesh.parcel_labels[idx_start:idx_end].astype(float)
        cbar_label = "parcel_label"

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    surf = (hemi_vertices, hemi_faces)
    plot_title = title or f"{mesh.subject_id} — {mode}"

    display = nplot.plot_surf(
        surf_mesh=surf,
        surf_map=stat_map,
        hemi=hemi,
        view=view,
        cmap=colormap,
        bg_on_stat=bg_on_stat,
        title=plot_title,
        output_file=str(output_file) if output_file is not None else None,
        colorbar=True,
    )
    logger.info(
        "[%s] nilearn plot rendered (mode=%s, hemi=%s, view=%s)",
        mesh.subject_id, mode, hemi, view,
    )
    return display
