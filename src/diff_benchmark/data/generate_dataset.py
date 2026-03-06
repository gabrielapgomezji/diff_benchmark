from pathlib import Path
from typing import Callable, Dict, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


class CustomDataset(Dataset):
    """PyTorch :class:`Dataset` wrapping brain features, targets, and gender labels.

    Features can be:

    - **Numeric arrays** — loaded from a DataFrame and converted to tensors directly.
    - **File paths** — NIfTI images loaded on-the-fly; optional *transform* is
      applied slice-by-slice.
    - **Mesh paths** — when *mesh_data* is provided, ``X`` is a dict of Parquet
      file paths (no data loaded at dataset time).

    ``__getitem__`` **always** returns a three-element tuple::

        (X, target, gender)

    where ``X`` is:

    - A float tensor for array / image pipelines.
    - A dict ``{"nodes_path": Path, "edges_path": Path}`` for mesh pipelines.

    The mesh dict is intentionally lightweight — actual graph data is loaded by
    the model (or a collate/pre-processing step) from the Parquet files.

    Args:
        features: DataFrame with a ``subject_id`` column plus either numeric
            feature columns or a single path column.
        targets: 1-D array of target values aligned with *features*.
        gender: 1-D integer array of gender labels aligned with *features*.
        transform: Optional callable applied to each 2-D slice when in path mode.
        mesh_data: Optional mapping ``{subject_id: {"nodes": Path, "edges": Path}}``
            populated by
            :meth:`~diff_benchmark.preprocessing.brain_feature_extraction.MeshPipeline.get_mesh_parquet_paths`.
            When provided the dataset operates in **mesh mode**.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        gender: np.ndarray,
        transform: Optional[Callable] = None,
        mesh_data: Optional[Dict] = None,
    ):
        self._subject_ids = features["subject_id"].tolist()
        self.features = features.drop(columns=["subject_id"])
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.gender = torch.tensor(gender, dtype=torch.int64)
        self.transform = transform
        self._mesh_data = mesh_data  # {subject_id: {"nodes": Path, "edges": Path}} or None

        self.mode = self._detect_mode()

        if self.mode == "features":
            self.features = torch.tensor(self.features.to_numpy(), dtype=torch.float32)
        elif self.mode == "paths":
            self.features = self.features.iloc[:, 0].tolist()
        # mesh mode: features DataFrame is unused (X comes from nodes.parquet)

    def _detect_mode(self) -> str:
        """Determine whether features are numeric arrays, file paths, or mesh data.

        Returns:
            ``"mesh"`` when *mesh_data* is provided, ``"features"`` for numeric
            data, or ``"paths"`` for file-path data.
        """
        if self._mesh_data is not None:
            return "mesh"
        if np.issubdtype(self.features.dtypes.iloc[0], np.number):
            return "features"
        return "paths"

    def __len__(self) -> int:
        return len(self._subject_ids)

    def __getitem__(self, idx: int):
        """Return a sample for *idx* as ``(X, target, gender)``.

        For **mesh** mode ``X`` is a lightweight dict::

            {"nodes_path": Path(..._nodes.parquet), "edges_path": Path(..._edges.parquet)}

        No graph data is loaded here; the model (or a pre-processing step) reads
        the Parquet files from the paths when needed.

        For **array** mode ``X`` is a float tensor of scalar features.

        For **image** (paths) mode the NIfTI image is loaded, clipped to
        ``[0, 7]``, normalised to ``[0, 1]``, and each axial slice is
        optionally transformed.  Returns ``None`` if the file cannot be read.

        Args:
            idx: Sample index.

        Returns:
            ``(X, target, gender)`` tuple, or ``None`` on I/O error.
        """
        # ----------------------------------------------------------------
        # MESH mode — X is a dict of Parquet file paths (lazy loading)
        # ----------------------------------------------------------------
        if self.mode == "mesh":
            subject_id = self._subject_ids[idx]
            mesh_paths = self._mesh_data.get(subject_id)
            if mesh_paths is None:
                logger.warning(
                    "No mesh data found for subject %s — returning None", subject_id
                )
                return None

            X = {
                "nodes_path": mesh_paths["nodes"],
                "edges_path": mesh_paths["edges"],
            }
            return X, self.targets[idx], self.gender[idx]

        # ----------------------------------------------------------------
        # ARRAY mode — X is a scalar feature tensor
        # ----------------------------------------------------------------
        if self.mode == "features":
            return self.features[idx], self.targets[idx], self.gender[idx]

        # ----------------------------------------------------------------
        # IMAGE (paths) mode — load NIfTI on-the-fly
        # ----------------------------------------------------------------
        try:
            img = nib.load(Path(self.features[idx]))
            data = np.nan_to_num(img.get_fdata()).clip(0, 7) / 7.0
            scalar_feat = torch.tensor(data, dtype=torch.float32)

            if self.transform is not None:
                slices = [
                    self.transform(scalar_feat[i, :, :])
                    for i in range(scalar_feat.shape[0])
                ]
                scalar_feat = torch.stack(slices, dim=0)       # (D, 1, H, W)
                scalar_feat = scalar_feat.permute(1, 0, 2, 3)  # (C=1, D, H, W)

        except (OSError, FileNotFoundError) as e:
            logger.warning(f"Dropping subject {Path(self.features[idx])}: {e}")
            return None

        return scalar_feat, self.targets[idx], self.gender[idx]

    # Expose feature-detection publicly for external inspection.
    get_features_model = _detect_mode

    @property
    def subject_ids(self) -> list:
        """List of subject IDs aligned with dataset indices."""
        return self._subject_ids

    @property
    def has_mesh(self) -> bool:
        """``True`` when mesh data is available for at least one subject."""
        return self._mesh_data is not None and len(self._mesh_data) > 0
