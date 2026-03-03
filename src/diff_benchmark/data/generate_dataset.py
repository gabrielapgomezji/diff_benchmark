from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


class CustomDataset(Dataset):
    """PyTorch Dataset wrapping brain features, targets, and gender labels."""

    def __init__(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        gender: np.ndarray,
        transform: Callable = None,
    ):
        self._subject_ids = features["subject_id"].tolist()
        self.features = features.drop(columns=["subject_id"])
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.gender = torch.tensor(gender, dtype=torch.int64)
        self.transform = transform

        self.mode = self.get_features_model()

        if self.mode == "features":
            self.features = self.features.to_numpy()
            self.features = torch.tensor(self.features, dtype=torch.float32)
        if self.mode == "paths":
            self.features = self.features[0].tolist()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(features, target, gender)`` for sample *idx*."""
        if self.mode == "features":
            final_features = self.features[idx]
        if self.mode == "paths":
            try:
                img = nib.load(Path(self.features[idx]))
                data = np.nan_to_num(img.get_fdata()).clip(0, 7)
                data /= 7.0
                final_features = torch.tensor(data, dtype=torch.float32)
                if self.transform is not None:
                    slices = []
                    for i in range(
                        final_features.shape[0]
                    ):
                        slice_2d = final_features[i, :, :]
                        slice_2d = self.transform(slice_2d)
                        slices.append(slice_2d)
                    final_features = torch.stack(slices, dim=0)  # (D,1,H,W)
                    final_features = final_features.permute(1, 0, 2, 3)  # (C=1,D,H,W)
            except (OSError, FileNotFoundError) as e:
                logger.warning(
                    f"[Warning] Dropping subject {Path(self.features[idx])}: {e}"
                )
                return None

        return final_features, self.targets[idx], self.gender[idx]

    def get_features_model(self) -> str:
        """Detect whether features are numeric arrays (``'features'``) or file paths (``'paths'``)."""

        if np.issubdtype(self.features.dtypes[0], np.number):
            self.mode = "features"
        else:
            self.mode = "paths"
        return self.mode

    @property
    def subject_ids(self) -> list:
        """List of subject IDs aligned with dataset indices."""
        return self._subject_ids
