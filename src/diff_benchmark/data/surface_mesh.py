"""Surface mesh data representation for cortical microstructure analysis.

This module provides :class:`SurfaceMeshData`, a lightweight container that
bundles a cortical surface mesh with per-vertex microstructural features and
parcellation labels.  The design is intentionally framework-agnostic: the
class works with plain NumPy arrays and can be serialised to PyTorch tensors
or lifted to a ``torch_geometric.data.Data`` object for graph neural network
pipelines.

Typical lifecycle
-----------------
1. Load from pre-computed mesh and derivative files via
   :meth:`SurfaceMeshData.from_gifti_files`.
2. Attach parcel labels via :meth:`SurfaceMeshData.attach_parcel_labels`.
3. Pass to :class:`~diff_benchmark.data.generate_dataset.CustomDataset`
   (the dataset stores a ``SurfaceMeshData`` per subject and converts it to
   tensors inside ``__getitem__``).

PyTorch Geometric compatibility
--------------------------------
Call :meth:`SurfaceMeshData.to_pyg` (requires ``torch_geometric`` to be
installed) to obtain a ``Data`` object ready for GNN forward passes::

    data = mesh.to_pyg()
    # data.x        — feature matrix  (N, F)
    # data.edge_index — COO adjacency  (2, E)
    # data.pos      — vertex positions (N, 3)
    # data.y        — parcel labels   (N,)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import nibabel as nib
import numpy as np
import torch

from diff_benchmark.utils.logger import setup_logger

# torch_geometric is an optional dependency — only imported at runtime inside
# to_pyg() via a lazy import.  No top-level import is needed.

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------


@dataclass
class SurfaceMeshData:
    """Container for a projected cortical surface mesh with microstructure features.

    All arrays are stored in **NumPy format**; call :meth:`to_tensors` to
    convert them to PyTorch or :meth:`to_pyg` for PyTorch Geometric.

    Attributes:
        vertices: ``(N, 3)`` float32 array of vertex coordinates (mm).
        faces: ``(M, 3)`` int32 array of triangle face indices.
        features: ``(N, F)`` float32 array of per-vertex microstructure values.
            Each column corresponds to one metric/time-point.
        parcel_labels: ``(N,)`` int32 array mapping each vertex to a parcellation
            region ID (0 = unlabelled / medial-wall).
        subject_id: Optional string identifying the source subject.
        metric: Optional name of the microstructure metric stored in *features*.
        hemisphere: ``"L"`` (left), ``"R"`` (right), or ``"LR"`` (combined).
        n_left_vertices: Number of left-hemisphere vertices when
            ``hemisphere == "LR"``.  Vertices ``[0, n_left_vertices)`` belong
            to LH; ``[n_left_vertices, N)`` to RH.  ``None`` if not a combined
            mesh.
    """

    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (M, 3) int32
    features: np.ndarray  # (N, F) float32  — F == 1 for a single metric
    parcel_labels: np.ndarray  # (N,)  int32
    subject_id: Optional[str] = field(default=None)
    metric: Optional[str] = field(default=None)
    hemisphere: str = field(default="LR")
    # Number of left-hemisphere vertices when hemisphere == "LR".
    # Vertices [0, n_left_vertices) belong to LH; the rest to RH.
    n_left_vertices: Optional[int] = field(default=None)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_gifti_files(
        cls,
        left_surf: Path,
        right_surf: Path,
        left_scalar: Path,
        right_scalar: Path,
        subject_id: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> "SurfaceMeshData":
        """Build a combined (left + right) mesh from GIFTI surface and scalar files.

        The left and right hemispheres are concatenated along the vertex axis.
        Right-hemisphere face indices are offset so they remain valid after
        concatenation.

        Args:
            left_surf: Path to the left-hemisphere ``.surf.gii`` file.
            right_surf: Path to the right-hemisphere ``.surf.gii`` file.
            left_scalar: Path to the left-hemisphere ``.scalar.gii`` file
                (values aligned with *left_surf* vertices).
            right_scalar: Path to the right-hemisphere ``.scalar.gii`` file.
            subject_id: Optional subject identifier.
            metric: Optional metric name.

        Returns:
            :class:`SurfaceMeshData` with both hemispheres combined.

        Raises:
            ValueError: If the vertex count of the scalar file does not match
                the surface file.
        """
        lv, lf = cls._load_surf_gii(left_surf)
        rv, rf = cls._load_surf_gii(right_surf)

        ls = cls._load_scalar_gii(left_scalar)   # (N_L,) or (N_L, F)
        rs = cls._load_scalar_gii(right_scalar)  # (N_R,) or (N_R, F)

        if lv.shape[0] != ls.shape[0]:
            raise ValueError(
                f"Left surface has {lv.shape[0]} vertices but scalar has "
                f"{ls.shape[0]} rows [{left_scalar}]"
            )
        if rv.shape[0] != rs.shape[0]:
            raise ValueError(
                f"Right surface has {rv.shape[0]} vertices but scalar has "
                f"{rs.shape[0]} rows [{right_scalar}]"
            )

        # Offset right-hemisphere face indices
        n_left = lv.shape[0]
        rf_offset = rf + n_left

        vertices = np.concatenate([lv, rv], axis=0).astype(np.float32)
        faces = np.concatenate([lf, rf_offset], axis=0).astype(np.int32)

        # Ensure features are 2-D (N, F)
        ls_2d = ls[:, np.newaxis] if ls.ndim == 1 else ls
        rs_2d = rs[:, np.newaxis] if rs.ndim == 1 else rs
        features = np.concatenate([ls_2d, rs_2d], axis=0).astype(np.float32)

        # Parcel labels default to zeros — call attach_parcel_labels to fill
        parcel_labels = np.zeros(vertices.shape[0], dtype=np.int32)

        logger.debug(
            "[%s] Mesh loaded: %d vertices, %d faces, %d feature cols",
            subject_id,
            vertices.shape[0],
            faces.shape[0],
            features.shape[1],
        )
        return cls(
            vertices=vertices,
            faces=faces,
            features=features,
            parcel_labels=parcel_labels,
            subject_id=subject_id,
            metric=metric,
            hemisphere="LR",
        )

    # ------------------------------------------------------------------
    # Parcellation
    # ------------------------------------------------------------------

    def attach_parcel_labels(
        self,
        schaefer_resampled: dict,
        n_left_vertices: Optional[int] = None,
    ) -> "SurfaceMeshData":
        """Assign each vertex its Schaefer parcel integer ID (in-place + return self).

        Uses the pre-computed ``schaefer_resampled`` dict produced by
        :func:`~diff_benchmark.preprocessing.utils.utils_surface_skeleton.resample_schaefer_onto_fs_lr`.

        The combined label vector is::

            parcel_labels = [ left_parcel_ids | right_parcel_ids ]

        Right-hemisphere IDs are kept as-is (Schaefer labels are globally
        unique across hemispheres).  Vertices on the medial wall receive ID 0.

        Args:
            schaefer_resampled: Dict with keys ``"left.data"`` and
                ``"right.data"`` containing per-vertex parcel ID arrays.
            n_left_vertices: Number of left-hemisphere vertices.  If ``None``,
                inferred from ``schaefer_resampled["left.data"]``.

        Returns:
            ``self`` (mutated in-place for convenience in method chaining).
        """
        left_labels = schaefer_resampled["left.data"].astype(np.int32)
        right_labels = schaefer_resampled["right.data"].astype(np.int32)

        expected_n = len(left_labels) + len(right_labels)
        if self.vertices.shape[0] != expected_n:
            logger.warning(
                "[%s] Vertex count mismatch: mesh has %d vertices but parcellation "
                "has %d (L=%d + R=%d).  Parcel labels will be zero-padded / truncated.",
                self.subject_id,
                self.vertices.shape[0],
                expected_n,
                len(left_labels),
                len(right_labels),
            )

        combined = np.concatenate([left_labels, right_labels])

        # Align length to mesh (pad with 0 or truncate if sizes differ)
        n = self.vertices.shape[0]
        if len(combined) >= n:
            self.parcel_labels = combined[:n].astype(np.int32)
        else:
            padded = np.zeros(n, dtype=np.int32)
            padded[: len(combined)] = combined
            self.parcel_labels = padded

        return self

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    def to_tensors(
        self, device: Optional[torch.device] = None
    ) -> dict[str, torch.Tensor]:
        """Return all arrays as a dictionary of PyTorch tensors.

        The dictionary keys mirror the field names so callers can unpack them
        uniformly regardless of hemisphere or metric configuration::

            batch = mesh.to_tensors()
            # batch["vertices"]      — FloatTensor (N, 3)
            # batch["faces"]         — LongTensor  (M, 3)
            # batch["features"]      — FloatTensor (N, F)
            # batch["parcel_labels"] — LongTensor  (N,)

        Args:
            device: Optional target device.  Defaults to CPU.

        Returns:
            Dictionary of named tensors.
        """
        tensors: dict[str, torch.Tensor] = {
            "vertices": torch.from_numpy(self.vertices).float(),
            "faces": torch.from_numpy(self.faces).long(),
            "features": torch.from_numpy(self.features).float(),
            "parcel_labels": torch.from_numpy(self.parcel_labels).long(),
        }
        if device is not None:
            tensors = {k: v.to(device) for k, v in tensors.items()}
        return tensors

    def to_pyg(self, device: Optional[torch.device] = None):
        """Convert to a ``torch_geometric.data.Data`` object.

        The mapping is:

        - ``data.x``          ← ``features``  (N, F)
        - ``data.pos``        ← ``vertices``  (N, 3)
        - ``data.face``       ← ``faces``     (3, M) — PyG convention
        - ``data.y``          ← ``parcel_labels`` (N,)
        - ``data.edge_index`` ← COO adjacency derived from *faces*

        Args:
            device: Optional target device.

        Returns:
            ``torch_geometric.data.Data`` instance.

        Raises:
            ImportError: If ``torch_geometric`` is not installed.
        """
        try:
            from torch_geometric.data import Data  # type: ignore[import-untyped]
            from torch_geometric.utils import to_undirected  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "torch_geometric is required for SurfaceMeshData.to_pyg(). "
                "Install it with: pip install torch-geometric"
            ) from exc

        t = self.to_tensors(device=device)
        faces_t = t["faces"].t().contiguous()  # (3, M)

        # Build undirected edge_index from face triangles
        edge_index = torch.cat(
            [
                faces_t[[0, 1]],
                faces_t[[1, 2]],
                faces_t[[0, 2]],
            ],
            dim=1,
        )
        edge_index = to_undirected(edge_index)

        return Data(
            x=t["features"],
            pos=t["vertices"],
            face=faces_t,
            edge_index=edge_index,
            y=t["parcel_labels"],
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        n_parcels = int(np.unique(self.parcel_labels[self.parcel_labels > 0]).size)
        return (
            f"SurfaceMeshData("
            f"subject={self.subject_id!r}, "
            f"metric={self.metric!r}, "
            f"hemi={self.hemisphere!r}, "
            f"vertices={self.vertices.shape[0]}, "
            f"faces={self.faces.shape[0]}, "
            f"features={self.features.shape}, "
            f"parcels={n_parcels})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_surf_gii(path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load vertices and faces from a GIFTI surface file.

        Args:
            path: Path to a ``.surf.gii`` file.

        Returns:
            ``(vertices, faces)`` as float32 and int32 arrays.
        """
        img = nib.load(str(path))
        vertices = img.darrays[0].data.astype(np.float32)  # (N, 3)
        faces = img.darrays[1].data.astype(np.int32)       # (M, 3)
        return vertices, faces

    @staticmethod
    def _load_scalar_gii(path: Path) -> np.ndarray:
        """Load scalar data from a GIFTI scalar file.

        Args:
            path: Path to a ``.scalar.gii`` file.

        Returns:
            1-D float32 array of length N (number of vertices).
        """
        img = nib.load(str(path))
        return img.darrays[0].data.astype(np.float32)


# ---------------------------------------------------------------------------
# Batch collation helper
# ---------------------------------------------------------------------------


def collate_mesh_batch(
    mesh_list: Sequence[Optional[SurfaceMeshData]],
) -> Optional[dict[str, torch.Tensor]]:
    """Collate a list of :class:`SurfaceMeshData` objects into a batched dict.

    Skips ``None`` entries (failed loads).  All meshes in the batch **must**
    share the same vertex and face count (as is the case for fsLR 32k template
    meshes).

    When the list is entirely ``None`` or empty, returns ``None`` so that
    downstream code can detect the absence of mesh data.

    Args:
        mesh_list: Sequence of mesh objects, potentially containing ``None``.

    Returns:
        Dictionary with stacked tensors (batch dimension prepended) or ``None``.
    """
    valid = [m for m in mesh_list if m is not None]
    if not valid:
        return None

    # Stack each field — shapes are (B, N, 3), (B, M, 3), (B, N, F), (B, N)
    try:
        return {
            "vertices": torch.stack(
                [torch.from_numpy(m.vertices).float() for m in valid]
            ),
            "faces": torch.stack(
                [torch.from_numpy(m.faces).long() for m in valid]
            ),
            "features": torch.stack(
                [torch.from_numpy(m.features).float() for m in valid]
            ),
            "parcel_labels": torch.stack(
                [torch.from_numpy(m.parcel_labels).long() for m in valid]
            ),
        }
    except RuntimeError as exc:
        logger.error(
            "collate_mesh_batch: shape mismatch across meshes — %s. "
            "All meshes must share the same template space.",
            exc,
        )
        return None
