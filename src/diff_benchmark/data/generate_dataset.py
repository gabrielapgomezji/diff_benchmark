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

    Features can be either:

    - **Numeric arrays** — loaded from a DataFrame and converted to tensors directly.
    - **File paths** — NIfTI images loaded on-the-fly; optional *transform* is
      applied slice-by-slice.

    An optional *mesh_data* mapping can be attached after construction to make
    each ``__getitem__`` call return a four-element tuple::

        (features, target, gender, mesh_dict)

    where ``mesh_dict`` is a ``dict[str, torch.Tensor]`` produced by
    :meth:`~diff_benchmark.data.surface_mesh.SurfaceMeshData.to_tensors`.
    When no mesh data is present the classic three-tuple is returned, preserving
    full backward compatibility.

    Args:
        features: DataFrame with a ``subject_id`` column plus either numeric
            feature columns or a single path column.
        targets: 1-D array of target values aligned with *features*.
        gender: 1-D integer array of gender labels aligned with *features*.
        transform: Optional callable applied to each 2-D slice when in path mode.
        mesh_data: Optional mapping ``{subject_id: SurfaceMeshData}`` populated
            by :class:`~diff_benchmark.preprocessing.brain_feature_extraction.MeshPipeline`.
            When provided, each sample also returns a tensor dict with keys
            ``"vertices"``, ``"faces"``, ``"features"``, and ``"parcel_labels"``.
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
        self._mesh_data = mesh_data  # {subject_id: SurfaceMeshData} or None

        self.mode = self._detect_mode()

        if self.mode == "features":
            self.features = torch.tensor(self.features.to_numpy(), dtype=torch.float32)
        elif self.mode == "paths":
            self.features = self.features.iloc[:, 0].tolist()

    def _detect_mode(self) -> str:
        """Determine whether features are numeric arrays or file paths.

        Returns:
            ``"features"`` for numeric data or ``"paths"`` for file-path data.
        """
        if np.issubdtype(self.features.dtypes.iloc[0], np.number):
            return "features"
        return "paths"

    def __len__(self) -> int:
        return len(self.features) if self.mode == "features" else len(self.features)

    def __getitem__(self, idx: int):
        """Return a sample for *idx*.

        Return value depends on whether mesh data is attached:

        - **Without mesh**: ``(features, target, gender)`` — three tensors.
        - **With mesh**:    ``(features, target, gender, mesh_dict)`` where
          ``mesh_dict`` is a ``dict[str, torch.Tensor]`` with keys
          ``"vertices"``, ``"faces"``, ``"features"``, ``"parcel_labels"``.

        When in ``"paths"`` mode, the NIfTI image is loaded, clipped to [0, 7],
        normalised to [0, 1], and each axial slice is optionally transformed.
        Returns ``None`` if the file cannot be read.

        Args:
            idx: Sample index.

        Returns:
            Tuple described above, or ``None`` on I/O error.
        """
        # ---- Build scalar features ----
        if self.mode == "features":
            scalar_feat = self.features[idx]
        else:
            # Path mode: load NIfTI on-the-fly.
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

        # ---- Optionally attach mesh ----
        if self._mesh_data is not None:
            subject_id = self._subject_ids[idx]
            mesh_obj = self._mesh_data.get(subject_id)
            if mesh_obj is not None:
                mesh_dict = mesh_obj.to_tensors()
            else:
                # Subject not present in mesh results — return empty dict sentinel
                mesh_dict = {}
            return scalar_feat, self.targets[idx], self.gender[idx], mesh_dict

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
