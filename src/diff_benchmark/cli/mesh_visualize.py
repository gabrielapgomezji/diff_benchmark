"""CLI for surface mesh visualisation and validation.

This script loads already-computed mesh graphs (Parquet files produced by the
mesh pipeline) and renders interactive or static visualisations.  It can also
validate mesh arrays and export debug ``.npz`` snapshots.

Outputs are written to ``./exp_outputs/meshes/``.

Usage
-----
Visualise one subject with Plotly (default)::

    diffbenchmark-mesh-visualize dataset=hcp --subject 100307

Static nilearn render::

    diffbenchmark-mesh-visualize dataset=hcp --subject 100307 --visualize nilearn

Validation + NPZ export only::

    diffbenchmark-mesh-visualize dataset=hcp --subject 100307 --visualize none

All subjects (generates an HTML per subject in ``exp_outputs/meshes/plots/``)::

    diffbenchmark-mesh-visualize dataset=hcp
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from diff_benchmark.data.surface_mesh import SurfaceMeshData
from diff_benchmark.preprocessing.brain_feature_extraction import MeshPipeline
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.preprocessing.utils.utils_graph_export import (
    load_graph_from_parquet,
)
from diff_benchmark.preprocessing.utils.utils_mesh_visualization import (
    plot_mesh_nilearn,
    plot_mesh_plotly,
)
from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)

_VISUALIZE_CHOICES = ("plotly", "nilearn", "none")

# Root for all mesh visualisation outputs (mirrors analysis.py pattern)
_EXP_OUTPUTS_ROOT = Path("./exp_outputs/meshes")


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def _parse_extra_args(overrides: list[str]) -> dict:
    """Extract ``--subject`` and ``--visualize`` from raw argv.

    Hydra consumes ``key=value`` tokens, so custom ``--flag`` arguments must be
    parsed before Hydra sees them.

    Returns:
        Dict with keys:

        * ``"subject"`` – subject ID string or ``None`` (all subjects).
        * ``"visualize"`` – one of ``"plotly"``, ``"nilearn"``, ``"none"``.
    """
    args: dict = {"subject": None, "visualize": "plotly"}
    i = 0
    while i < len(overrides):
        token = overrides[i]
        if token in ("--subject", "-s") and i + 1 < len(overrides):
            args["subject"] = overrides[i + 1]
            i += 2
        elif token.startswith("--subject="):
            args["subject"] = token.split("=", 1)[1]
            i += 1
        elif token in ("--visualize", "--viz") and i + 1 < len(overrides):
            val = overrides[i + 1].lower()
            if val not in _VISUALIZE_CHOICES:
                raise SystemExit(
                    f"--visualize must be one of {_VISUALIZE_CHOICES}, got '{val}'"
                )
            args["visualize"] = val
            i += 2
        elif token.startswith("--visualize=") or token.startswith("--viz="):
            val = token.split("=", 1)[1].lower()
            if val not in _VISUALIZE_CHOICES:
                raise SystemExit(
                    f"--visualize must be one of {_VISUALIZE_CHOICES}, got '{val}'"
                )
            args["visualize"] = val
            i += 1
        else:
            i += 1
    return args


# ---------------------------------------------------------------------------
# Mesh validation
# ---------------------------------------------------------------------------


def _validate_mesh(mesh: SurfaceMeshData, subject_id: str) -> None:
    """Assert internal shape consistency of a :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData`.

    Raises:
        AssertionError: On vertex / feature / label count mismatch or invalid
            face indices.
    """
    n = mesh.vertices.shape[0]
    assert n == mesh.features.shape[0], (
        f"[{subject_id}] vertices ({n}) vs features ({mesh.features.shape[0]}) mismatch"
    )
    assert n == mesh.parcel_labels.shape[0], (
        f"[{subject_id}] vertices ({n}) vs parcel_labels "
        f"({mesh.parcel_labels.shape[0]}) mismatch"
    )
    if mesh.faces.shape[0] > 0:
        assert int(mesh.faces.max()) < n, (
            f"[{subject_id}] face index {int(mesh.faces.max())} >= n_vertices {n}"
        )
    logger.info(
        "[mesh_visualize] [%s] Validation passed (%d vertices)", subject_id, n
    )


# ---------------------------------------------------------------------------
# Debug NPZ export
# ---------------------------------------------------------------------------


def _export_debug_npz(
    mesh: SurfaceMeshData, subject_id: str, output_dir: Path
) -> Path:
    """Save mesh arrays to a compressed ``.npz`` for offline inspection.

    Writes ``{subject_id}_mesh.npz`` containing arrays:
    ``vertices``, ``faces``, ``features``, ``parcel_labels``.

    Returns:
        Path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{subject_id}_mesh.npz"
    np.savez_compressed(
        npz_path,
        vertices=mesh.vertices,
        faces=mesh.faces,
        features=mesh.features,
        parcel_labels=mesh.parcel_labels,
    )
    logger.info("[mesh_visualize] [%s] Debug NPZ saved → %s", subject_id, npz_path)
    return npz_path


# ---------------------------------------------------------------------------
# Per-subject visualisation
# ---------------------------------------------------------------------------


def _visualize_subject(
    mesh: SurfaceMeshData,
    subject_id: str,
    visualize: str,
    graph_dir: Path,
    exp_outputs_dir: Path,
) -> None:
    """Validate and visualise a single subject's mesh.

    Args:
        mesh: The :class:`~diff_benchmark.data.surface_mesh.SurfaceMeshData` to
            visualise.
        subject_id: Used for file names and log messages.
        visualize: One of ``"plotly"``, ``"nilearn"``, ``"none"``.
        graph_dir: Directory containing the Parquet graph files.
        exp_outputs_dir: Root output directory (``exp_outputs/meshes/``).
    """
    # 1. Validate
    _validate_mesh(mesh, subject_id)

    # 2. Debug NPZ
    _export_debug_npz(mesh, subject_id, exp_outputs_dir)

    # 3. Visualisation
    if visualize == "none":
        logger.info(
            "[mesh_visualize] [%s] Visualisation skipped (--visualize none)",
            subject_id,
        )
        return

    plots_dir = exp_outputs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if visualize == "plotly":
        html_path = plots_dir / f"{subject_id}_mesh.html"
        logger.info(
            "[mesh_visualize] [%s] Rendering Plotly mesh → %s", subject_id, html_path
        )
        print(f"[mesh_visualize] [{subject_id}] Rendering Plotly mesh → {html_path}")
        plot_mesh_plotly(
            subject_id=subject_id,
            graph_dir=graph_dir,
            show_edges=False,
            output_html=html_path,
        )

    elif visualize == "nilearn":
        img_path = plots_dir / f"{subject_id}_mesh.png"
        logger.info(
            "[mesh_visualize] [%s] Rendering nilearn mesh → %s", subject_id, img_path
        )
        print(f"[mesh_visualize] [{subject_id}] Rendering nilearn mesh → {img_path}")
        plot_mesh_nilearn(
            mesh,
            mode="feature",
            hemi="left",
            view="lateral",
            output_file=img_path,
        )


# ---------------------------------------------------------------------------
# Full-dataset sweep
# ---------------------------------------------------------------------------


def _run_all_subjects(pipeline: MeshPipeline, visualize: str) -> None:
    """Validate and visualise every subject present in ``pipeline.results``.

    If ``pipeline.results`` is empty (pipeline not run yet), tries to load each
    subject's graph from its Parquet files under
    ``<results_root>/derivatives/sub-<id>/dwi/``.

    Args:
        pipeline: Initialised :class:`~diff_benchmark.preprocessing.brain_feature_extraction.MeshPipeline`.
        visualize: Visualisation backend (``"plotly"``, ``"nilearn"``, ``"none"``).
    """
    results = pipeline.results

    if not results:
        logger.warning(
            "[mesh_visualize] pipeline.results is empty — "
            "run the mesh pipeline first with diffbenchmark-mesh"
        )
        print(
            "[mesh_visualize] WARNING: No subjects in pipeline.results. "
            "Run diffbenchmark-mesh first."
        )
        return

    for subject_id, mesh in results.items():
        graph_dir = (
            pipeline.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        _visualize_subject(
            mesh=mesh,
            subject_id=subject_id,
            visualize=visualize,
            graph_dir=graph_dir,
            exp_outputs_dir=_EXP_OUTPUTS_ROOT,
        )

    print(f"[mesh_visualize] Done — processed {len(results)} subjects.")


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(
    version_base="1.3",
    config_path="pkg://diff_benchmark.configs",
    config_name="main",
)
def main(cfg: DictConfig) -> None:
    """Main CLI entry point for mesh visualisation.

    Accepts all standard Hydra overrides plus:

    ``--subject SUBJECT_ID``
        Visualise only this subject.  If omitted, all subjects in
        ``pipeline.results`` are processed.

    ``--visualize {plotly,nilearn,none}``
        Visualisation backend.  Defaults to ``plotly``.

    Outputs go to ``./exp_outputs/meshes/``.
    """
    subject_id="100307"
    visualize="plotly"

    logger.info(
        "[mesh_visualize] Starting (subject=%s, visualize=%s)", subject_id, visualize
    )

    # ------------------------------------------------------------------
    # Build DatasetConfig from Hydra config
    # ------------------------------------------------------------------
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    cluster_cfg = cfg.cluster.paths[dataset_cfg["name"]]

    dataset_selected = DatasetConfig(
        **dataset_cfg,
        base_dir=Path(cluster_cfg.base_dir),
        results_dir=Path(cluster_cfg.results_dir),
    )

    surface_type = getattr(dataset_selected, "mesh_surface_type", "midthickness")
    pipeline = MeshPipeline(dataset_selected, surface_type=surface_type)

    _EXP_OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single-subject mode
    # ------------------------------------------------------------------
    if subject_id is not None:
        graph_dir = (
            pipeline.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        # Load mesh from already-exported Parquet files
        try:
            mesh = load_graph_from_parquet(subject_id=subject_id, graph_dir=graph_dir)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"[mesh_visualize] Graph Parquet not found for subject '{subject_id}' "
                f"under {graph_dir}.\n"
                "Run diffbenchmark-mesh first to generate the graph files."
            ) from exc

        _visualize_subject(
            mesh=mesh,
            subject_id=subject_id,
            visualize=visualize,
            graph_dir=graph_dir,
            exp_outputs_dir=_EXP_OUTPUTS_ROOT,
        )

    # ------------------------------------------------------------------
    # All-subjects mode
    # ------------------------------------------------------------------
    else:
        _run_all_subjects(pipeline, visualize)


if __name__ == "__main__":
    main()
