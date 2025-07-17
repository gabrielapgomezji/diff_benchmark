from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, TensorDataset

from diff_benchmark.dataloaders.base import DatasetSpecs  # AbstractPreprocessedData


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

    def __init__(self, X, y, genders, n_splits=5, random_state=42):
        super().__init__(X, y, genders, n_splits)
        self.genders = genders
        self.skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

    def get_fold_indices(self):
        indices = list(self.skf.split(np.zeros(len(self.genders)), self.genders))
        return indices

    def get_dataloader_fold(
        self, dataset, fold_idx, fold_indices, batch_size=32, shuffle=True
    ):
        """
        Returns DataLoaders for the specified fold index using precomputed indices.
        """
        train_idx, test_idx = fold_indices[fold_idx]

        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size, shuffle=shuffle
        )
        test_loader = DataLoader(
            Subset(dataset, test_idx), batch_size=batch_size, shuffle=False
        )

        return train_loader, test_loader

    def get_arrays_from_indices(self, dataset, fold_idx, fold_indices):
        """
        Given full arrays and fold index, return X, y, gender arrays for train/test sets.
        """
        train_idx, test_idx = fold_indices[fold_idx]

        X = dataset.X.numpy()
        y = dataset.y.numpy()
        genders = dataset.gender.numpy()

        return (
            X[train_idx],
            y[train_idx],
            genders[train_idx],
            X[test_idx],
            y[test_idx],
            genders[test_idx],
        )

    def get_folds_as_dataloaders(self, batch_size=32, shuffle=True):
        folds = []

        for train_idx, val_idx in self.skf.split(self.X, self.genders):
            X_train, y_train, g_train = (
                self.X[train_idx],
                self.y[train_idx],
                self.genders[train_idx],
            )
            X_val, y_val, g_val = (
                self.X[val_idx],
                self.y[val_idx],
                self.genders[val_idx],
            )

            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(g_train, dtype=torch.int64),
            )
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
                torch.tensor(g_val, dtype=torch.int64),
            )

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=shuffle
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            folds.append((train_loader, val_loader))

        return folds

    def get_folds_as_arrays(self):
        folds = []

        for train_idx, val_idx in self.skf.split(self.X, self.genders):
            train_data = (self.X[train_idx], self.y[train_idx], self.genders[train_idx])
            val_data = (self.X[val_idx], self.y[val_idx], self.genders[val_idx])
            folds.append((train_data, val_data))

        return folds

    def get_specs(self) -> DatasetSpecs:
        gender_dist = dict(Counter(self.genders))
        return DatasetSpecs(
            num_samples=len(self.X),
            num_features=self.X.shape[1],
            num_targets=self.y.shape[1] if self.y.ndim > 1 else 1,
            gender_distribution=gender_dist,
        )
