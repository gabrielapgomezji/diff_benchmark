from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, TensorDataset


@dataclass
class DatasetSpecs:
    """Snapshot of dataset statistics returned by :meth:`PreprocessedData.get_specs`."""

    num_samples: int
    num_features: int
    num_targets: int
    gender_distribution: dict


class PreprocessedData:
    """Stratified K-fold container for preprocessed features, targets, and genders."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        genders: np.ndarray,
        config: dict,
    ):
        self.features = features
        self.targets = targets
        self.genders = genders
        self.skf = StratifiedKFold(
            n_splits=config.data.data_partition["n_splits"],
            shuffle=True,
            random_state=config.random_state,
        )
        self.config = config

    def get_fold_indices(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Returns the indices for each fold in the stratified K-Folds."""
        indices = list(self.skf.split(np.zeros(len(self.genders)), self.genders))
        train_size = self.config.data.data_partition.train_size

        # If using absolute number > 1 or percentage < 1 (where 1.0 means 100% so no change)
        if train_size != 1.0:
            print(f"Applying train_size: {train_size}")
            # Use a fixed random state for reproducibility and nested subsets independent of the main seed
            rng = np.random.RandomState(self.config.random_state)

            new_indices = []

            for fold_idx, (train_idx, test_idx) in enumerate(indices):
                shuffled_train_idx = train_idx.copy()
                rng.shuffle(shuffled_train_idx)

                n_train = len(train_idx)
                if train_size > 1:
                    n_samples = int(train_size)
                    if n_samples > n_train:
                        n_samples = n_train
                elif 0 < train_size <= 1:
                    n_samples = int(n_train * train_size)
                else:
                    raise ValueError(
                        f"Invalid train_size: {train_size}. Must be positive number."
                    )

                # nested subsets: smaller train_size is always a subset of larger train_size
                selected_train_idx = shuffled_train_idx[:n_samples]

                new_indices.append((selected_train_idx, test_idx))
                
                print(
                    f"Fold {fold_idx}: Reduced training set from {n_train} to {len(selected_train_idx)} samples"
                )

            indices = new_indices
        return indices

    def safe_collate(self, batch: list) -> torch.Tensor:
        """Collate function that filters out ``None`` samples from the batch."""
        batch = [b for b in batch if b is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    def get_dataloader_fold(
        self,
        dataset: TensorDataset,
        fold_idx: int,
        fold_indices: list,
        num_workers: int = 0,
        batch_size: int = 32, 
    ) -> tuple[DataLoader, DataLoader]:
        """
        Returns DataLoaders for the specified fold index using precomputed indices.

        Args:
            dataset: The dataset to create DataLoaders from.
            fold_idx: The index of the fold to retrieve.
            fold_indices: A list of tuples containing train and test indices for each fold.
            batch_size: Batch size for the DataLoaders.

        Returns:
            Tuple of (train_loader, test_loader).
        """
        train_idx, test_idx = fold_indices[fold_idx]

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.safe_collate,
        )
        test_loader = DataLoader(
            Subset(dataset, test_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.safe_collate,
        )

        return train_loader, test_loader

    def get_arrays_from_indices(
        self, dataset: TensorDataset, fold_idx: int, fold_indices: list
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return X, y, gender arrays for train and test sets of a given fold.

        Args:
            dataset: Dataset containing features, targets, and genders.
            fold_idx: Fold index to retrieve.
            fold_indices: List of (train_idx, test_idx) tuples per fold.

        Returns:
            Tuple of (X_train, y_train, g_train, X_test, y_test, g_test).
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
            idx: Indices to include in the dataset.

        Returns:
            TensorDataset containing the selected features, targets, and genders.
        """
        return TensorDataset(
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.genders[idx], dtype=torch.int64),
        )

    def get_folds_as_dataloaders(
        self, batch_size: int = 32, shuffle: bool = True
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Generates and returns DataLoaders for all folds.

        Args:
            batch_size: Batch size for the DataLoaders.
            shuffle: Whether to shuffle the training DataLoader.

        Returns:
            List of (train_loader, val_loader) tuples for each fold.
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

    def get_folds_as_arrays(
        self,
    ) -> list[
        tuple[
            tuple[np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ]
    ]:
        """Generates and returns arrays for all folds.

        Returns:
            List of ((X_train, y_train, g_train), (X_val, y_val, g_val)) tuples per fold.
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
        """Returns specifications of the dataset: sample count, feature count, and gender distribution."""
        gender_dist = dict(Counter(self.genders))
        return DatasetSpecs(
            num_samples=len(self.features),
            num_features=self.features.shape[1],
            num_targets=self.targets.shape[1] if self.targets.ndim > 1 else 1,
            gender_distribution=gender_dist,
        )
