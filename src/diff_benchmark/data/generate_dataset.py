from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Initializes the dataset object with features, labels, and gender information.
    Parameters:
        X (array-like): The input features for the dataset.
        y (array-like): The target labels for the dataset.
        gender (array-like): The gender information associated with each sample.
    Attributes:
        X (torch.Tensor): A tensor representation of the input features.
        y (torch.Tensor): A tensor representation of the target labels.
        gender (torch.Tensor): A tensor representation of the gender information.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        gender: np.ndarray,
        transform=None,
    ):
        # self.features = torch.tensor(features, dtype=torch.float32)
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
        """
        Returns the number of elements in the dataset.
        This method overrides the built-in __len__ method to provide the length
        of the dataset, which is determined by the number of samples in the
        attribute `self.X`.
        Returns:
            int: The number of samples in the dataset.
        """

        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a single data sample from the dataset.
        Args:
            idx (int): The index of the data sample to retrieve.
        Returns:
            tuple: A tuple containing the features (self.X[idx]),
                   the target variable (self.y[idx]),
                   and the gender information (self.gender[idx])
                   corresponding to the specified index.
        """
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
                    ):  # iterate through depth dimension
                        slice_2d = final_features[i, :, :]
                        slice_2d = self.transform(slice_2d)
                        slices.append(slice_2d)
                    final_features = torch.stack(slices, dim=0)  # (D,1,H,W)
                    final_features = final_features.permute(1, 0, 2, 3)  # (C=1,D,H,W)
            except (OSError, FileNotFoundError) as e:
                print(f"[Warning] Dropping subject {Path(self.features[idx])}: {e}")
                return None

        return final_features, self.targets[idx], self.gender[idx]

    def get_features_model(self) -> str:
        """
        Determines the mode of the features based on their data type.
        This method checks if the first column of the features DataFrame, excluding
        the 'subject_id' column, is of a numeric subtype. If it is numeric, the mode
        is set to "features"; otherwise, it is set to "paths".
        Returns:
            str: The mode of the features, either "features" or "paths".
        """

        if np.issubdtype(self.features.dtypes[0], np.number):
            self.mode = "features"
        else:
            self.mode = "paths"
        return self.mode
