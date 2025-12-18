from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch

from torch.utils.data import Dataset

class PreprocessedData:
    """
    PreprocessedData is a class for handling and processing datasets for machine learning tasks.
    Attributes:
        X (any): The features of the dataset.
        y (any): The labels of the dataset.
        mode (str): The mode of dataset processing, e.g., "all".
    Methods:
        build_dataset(): Constructs the dataset based on the provided configuration.
        get_folds_as_dataloaders(): Retrieves the dataset folds as dataloaders for training and validation.
    """

    def __init__(self, config):

        if config["mode"] == "all":
            self.features, self.targets = self.build_dataset()
            self.mode = "all"

    def build_dataset(self):
        """
        Builds the dataset for the project.
        This method is responsible for generating and preparing the dataset
        needed for the benchmarking process. It may involve loading data,
        processing it, and saving it in the required format.
        Currently, this method is not implemented.
        """

    def get_folds_as_dataloaders(self):
        """
        Retrieves the dataset folds as PyTorch DataLoader instances.
        This method is intended to be implemented to generate and return
        DataLoader objects for each fold of the dataset, which can be used
        for training and validation in a machine learning context.
        Returns:
            List[DataLoader]: A list of DataLoader instances, each corresponding
            to a different fold of the dataset.
        """


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

    def __init__(self, features, targets, gender, transform=None):
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

    def __len__(self):
        """
        Returns the number of elements in the dataset.
        This method overrides the built-in __len__ method to provide the length
        of the dataset, which is determined by the number of samples in the
        attribute `self.X`.
        Returns:
            int: The number of samples in the dataset.
        """

        return len(self.features)

    def __getitem__(self, idx):
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
                        slice_2d = final_features[
                            i, :, :
                        ]
                        slice_2d = self.transform(slice_2d)
                        slices.append(slice_2d)
                    final_features = torch.stack(slices, dim=0)  # (D,1,H,W)
                    final_features = final_features.permute(
                        1, 0, 2, 3
                    )  # (C=1,D,H,W)
            except (OSError, FileNotFoundError) as e:
                print(f"[Warning] Dropping subject {Path(self.features[idx])}: {e}")
                return None

        return final_features, self.targets[idx], self.gender[idx]

    def get_features_model(self):
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
