from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, TensorDataset


@dataclass
class DatasetSpecs:
    """
    DatasetSpecs is a class that holds specifications for a dataset.
    Attributes:
        num_samples (int): The total number of samples in the dataset.
        num_features (int): The number of features for each sample.
        num_targets (int): The number of target variables for each sample.
        gender_distribution (dict): A dictionary representing the distribution of genders in the dataset.
    """

    num_samples: int
    num_features: int
    num_targets: int
    gender_distribution: dict


class PreprocessedData:
    """
    PreprocessedData is a class for handling preprocessed datasets, providing functionality
    to create stratified folds for training and validation, and to generate data loaders
    for each fold.
    Attributes:
        X (np.ndarray): Feature data.
        y (np.ndarray): Target data.
        genders (np.ndarray): Gender labels for stratification.
        skf (StratifiedKFold): Stratified K-Folds cross-validator.
    Methods:
        get_fold_indices():
            Returns the indices for each fold in the stratified K-Folds.
        get_dataloader_fold(dataset, fold_idx, fold_indices, batch_size=32, shuffle=True):
        get_arrays_from_indices(dataset, fold_idx, fold_indices):
            Given full arrays and fold index, returns X, y, and gender arrays for train/test sets.
        get_folds_as_dataloaders(batch_size=32, shuffle=True):
            Generates and returns DataLoaders for all folds.
        get_folds_as_arrays():
            Generates and returns arrays for all folds.
        get_specs() -> DatasetSpecs:
            Returns specifications of the dataset including sample count, feature count,
            target count, and gender distribution.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, genders: np.ndarray, config: dict):
        self.features = features
        self.targets = targets
        self.genders = genders
        self.skf = StratifiedKFold(
            n_splits=config["data_partition"]["n_splits"],
            shuffle=True,
            random_state=config["random_state"],
        )
        self.config = config

    def get_fold_indices(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Returns the indices for each fold in the stratified K-Folds."""
        indices = list(self.skf.split(np.zeros(len(self.genders)), self.genders))
        return indices

    def safe_collate(self, batch: list) -> torch.Tensor:
        """
        Collate function that filters out None samples from the batch.
        Args:
            batch (list): A list of samples to be collated.
        Returns:
            torch.Tensor: A collated tensor of the batch with None samples removed.
        """
        # drop None samples
        batch = [b for b in batch if b is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    def get_dataloader_fold(
        self,
        dataset: TensorDataset,
        fold_idx: int,
        fold_indices: list,
        batch_size: int = 32,  # shuffle=True
    ) -> tuple[DataLoader, DataLoader]:
        """
        Returns DataLoaders for the specified fold index using precomputed indices.
        Args:
            dataset (TensorDataset): The dataset to create DataLoaders from.
            fold_idx (int): The index of the fold to retrieve.
            fold_indices (list): A list of tuples containing train and test indices for each fold.
            batch_size (int, optional): The batch size for the DataLoaders. Defaults to 32.
        Returns:
            tuple[DataLoader, DataLoader]: A tuple containing the train and test DataLoaders
        """
        train_idx, test_idx = fold_indices[fold_idx]

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config["dataloaders"]["num_workers"],
            collate_fn=self.safe_collate,
        )
        test_loader = DataLoader(
            Subset(dataset, test_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config["dataloaders"]["num_workers"],
            collate_fn=self.safe_collate,
        )

        return train_loader, test_loader

    def get_arrays_from_indices(self, dataset: TensorDataset, fold_idx: int, fold_indices: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Given full arrays and fold index, return X, y, gender arrays for train/test sets.
        Args:
            dataset (TensorDataset): The dataset containing features, targets, and genders.
            fold_idx (int): The index of the fold to retrieve.
            fold_indices (list): A list of tuples containing train and test indices for each fold.
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing train features, train targets, train genders,
                test features, test targets, test genders.
        """
        train_idx, test_idx = fold_indices[fold_idx]
        features = dataset.features.numpy()
        targets = dataset.targets.numpy()
        genders = dataset.gender.numpy()

        return (
            features[train_idx],
            targets[train_idx],
            genders[train_idx],
            features[test_idx],
            targets[test_idx],
            genders[test_idx],
        )

    def _create_dataset(self, idx: np.ndarray) -> TensorDataset:
        """Create a TensorDataset for the given indices.
         Args:
            idx (np.ndarray): Indices to include in the dataset.
        Returns:
            TensorDataset: A TensorDataset containing the selected features, targets, and genders.
        """
        return TensorDataset(
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.genders[idx], dtype=torch.int64),
        )

    def get_folds_as_dataloaders(self, batch_size: int = 32, shuffle: bool = True) -> list[tuple[DataLoader, DataLoader]]:
        """Generates and returns DataLoaders for all folds.
        Args:
            batch_size (int, optional): The batch size for the DataLoaders. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the training DataLoader. Defaults to True.
        Returns:
            list[tuple[DataLoader, DataLoader]]: A list of tuples containing train and validation DataLoaders for each fold.
        """
        folds = []

        for train_idx, val_idx in self.skf.split(self.features, self.genders):
            train_dataset = self._create_dataset(train_idx)
            val_dataset = self._create_dataset(val_idx)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=shuffle
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            folds.append((train_loader, val_loader))

        return folds

    def get_folds_as_arrays(self) -> list[tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Generates and returns arrays for all folds.
        Returns:
            list[tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]]:
                A list of tuples containing train and validation arrays for each fold.
        """
        folds = []

        for train_idx, val_idx in self.skf.split(self.features, self.genders):
            train_data = (
                self.features[train_idx],
                self.targets[train_idx],
                self.genders[train_idx],
            )
            val_data = (
                self.features[val_idx],
                self.targets[val_idx],
                self.genders[val_idx],
            )
            folds.append((train_data, val_data))

        return folds

    def get_specs(self) -> DatasetSpecs:
        """Returns specifications of the dataset including sample count, feature count, and gender distribution.
        Returns:
            DatasetSpecs: An object containing dataset specifications.
        """
        gender_dist = dict(Counter(self.genders))
        return DatasetSpecs(
            num_samples=len(self.features),
            num_features=self.features.shape[1],
            num_targets=self.targets.shape[1] if self.targets.ndim > 1 else 1,
            gender_distribution=gender_dist,
        )
