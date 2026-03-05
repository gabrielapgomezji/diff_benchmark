from pathlib import Path
from typing import Callable, Optional

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

    Args:
        features: DataFrame with a ``subject_id`` column plus either numeric
            feature columns or a single path column.
        targets: 1-D array of target values aligned with *features*.
        gender: 1-D integer array of gender labels aligned with *features*.
        transform: Optional callable applied to each 2-D slice when in path mode.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        gender: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        self._subject_ids = features["subject_id"].tolist()
        self.features = features.drop(columns=["subject_id"])
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.gender = torch.tensor(gender, dtype=torch.int64)
        self.transform = transform

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
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return ``(features, target, gender)`` for sample *idx*.

        When in ``"paths"`` mode, the NIfTI image is loaded, clipped to [0, 7],
        normalised to [0, 1], and each axial slice is optionally transformed.
        Returns ``None`` if the file cannot be read (the DataLoader's collate
        function should handle ``None`` entries).

        Args:
            idx: Sample index.

        Returns:
            Tuple of ``(feature_tensor, target_tensor, gender_tensor)`` or
            ``None`` on I/O error.
        """
        if self.mode == "features":
            return self.features[idx], self.targets[idx], self.gender[idx]

        # Path mode: load NIfTI on-the-fly.
        try:
            img = nib.load(Path(self.features[idx]))
            data = np.nan_to_num(img.get_fdata()).clip(0, 7) / 7.0
            final_features = torch.tensor(data, dtype=torch.float32)

            if self.transform is not None:
                slices = [
                    self.transform(final_features[i, :, :])
                    for i in range(final_features.shape[0])
                ]
                final_features = torch.stack(slices, dim=0)       # (D, 1, H, W)
                final_features = final_features.permute(1, 0, 2, 3)  # (C=1, D, H, W)

        except (OSError, FileNotFoundError) as e:
            logger.warning(f"Dropping subject {Path(self.features[idx])}: {e}")
            return None

        return final_features, self.targets[idx], self.gender[idx]

    # Expose feature-detection publicly for external inspection.
    get_features_model = _detect_mode

    @property
    def subject_ids(self) -> list:
        """List of subject IDs aligned with dataset indices."""
        return self._subject_ids
