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
    """Stratified K-fold container for preprocessed features, targets, and genders.

    Args:
        features: Feature array or DataFrame indexed by sample.
        targets: Target label array.
        genders: Gender label array used for stratification.
        config: OmegaConf config object providing ``data.data_partition``
            (``n_splits``, ``train_size``, ``val_size``) and ``random_state``.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        genders: np.ndarray,
        config,
    ):
        self.features = features
        self.targets = targets
        self.genders = genders
        self.config = config
        self.skf = StratifiedKFold(
            n_splits=config.data.data_partition["n_splits"],
            shuffle=True,
            random_state=config.random_state,
        )

    def get_fold_indices(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return stratified K-fold split indices, optionally subsampling training sets.

        When ``config.data.data_partition.train_size != 1.0``, each training
        split is truncated to the requested size.  Nested subsets are
        deterministic: a smaller ``train_size`` is always a subset of a larger
        one (reproducible via ``random_state``).

        Returns:
            List of ``(train_idx, test_idx)`` arrays, one tuple per fold.

        Raises:
            ValueError: If ``train_size`` is not a positive number.
        """
        indices = list(self.skf.split(np.zeros(len(self.genders)), self.genders))
        train_size = self.config.data.data_partition.train_size

        if train_size == 1.0:
            return indices

        print(f"Applying train_size: {train_size}")
        rng = np.random.RandomState(self.config.random_state)
        new_indices = []

        for fold_idx, (train_idx, test_idx) in enumerate(indices):
            shuffled = train_idx.copy()
            rng.shuffle(shuffled)

            n_total = len(train_idx)
            if train_size > 1:
                n_select = min(int(train_size), n_total)
            elif 0 < train_size <= 1:
                n_select = int(n_total * train_size)
            else:
                raise ValueError(f"Invalid train_size: {train_size}. Must be a positive number.")

            selected = shuffled[:n_select]
            new_indices.append((selected, test_idx))
            print(f"Fold {fold_idx}: training set reduced from {n_total} to {len(selected)} samples")

        return new_indices

    def safe_collate(self, batch: list) -> torch.Tensor:
        """Collate function that filters out ``None`` samples.

        Args:
            batch: List of samples from :meth:`Dataset.__getitem__`.

        Returns:
            Default-collated batch after removing ``None`` entries.
        """
        batch = [b for b in batch if b is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    def get_dataloader_fold(
        self,
        dataset,
        fold_idx: int,
        fold_indices: list,
        num_workers: int = 0,
        batch_size: int = 32,
    ) -> tuple[DataLoader, DataLoader]:
        """Build train and test :class:`DataLoader` objects for one fold.

        Args:
            dataset: The full dataset to subset.
            fold_idx: Index of the desired fold.
            fold_indices: Precomputed fold index list from :meth:`get_fold_indices`.
            num_workers: Number of DataLoader worker processes.
            batch_size: Batch size for both loaders.

        Returns:
            Tuple of ``(train_loader, test_loader)``.
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
        self, dataset, fold_idx: int, fold_indices: list
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return raw numpy arrays for train and test sets of a given fold.

        Args:
            dataset: Dataset with ``features``, ``targets``, and ``gender`` tensor attributes.
            fold_idx: Fold index to retrieve.
            fold_indices: Precomputed fold index list.

        Returns:
            Tuple of ``(X_train, y_train, g_train, X_test, y_test, g_test)``.
        """
        train_idx, test_idx = fold_indices[fold_idx]
        features = dataset.features.numpy()
        targets = dataset.targets.numpy()
        genders = dataset.gender.numpy()

        return (
            features[train_idx], targets[train_idx], genders[train_idx],
            features[test_idx], targets[test_idx], genders[test_idx],
        )

    def _create_dataset(self, idx: np.ndarray) -> TensorDataset:
        """Build a :class:`TensorDataset` from a subset of indices.

        Args:
            idx: Integer index array.

        Returns:
            TensorDataset of ``(features, targets, genders)`` for the selected samples.
        """
        return TensorDataset(
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.genders[idx], dtype=torch.int64),
        )

    def get_folds_as_dataloaders(
        self, batch_size: int = 32, shuffle: bool = True
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Generate DataLoaders for every fold.

        Args:
            batch_size: Batch size for all loaders.
            shuffle: Whether to shuffle training loaders.

        Returns:
            List of ``(train_loader, val_loader)`` tuples, one per fold.
        """
        folds = []
        for train_idx, val_idx in self.skf.split(self.features, self.genders):
            train_loader = DataLoader(
                self._create_dataset(train_idx), batch_size=batch_size, shuffle=shuffle
            )
            val_loader = DataLoader(
                self._create_dataset(val_idx), batch_size=batch_size, shuffle=False
            )
            folds.append((train_loader, val_loader))
        return folds

    def get_folds_as_arrays(
        self,
    ) -> list[tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Return raw arrays for every fold.

        Returns:
            List of ``((X_train, y_train, g_train), (X_val, y_val, g_val))`` per fold.
        """
        folds = []
        for train_idx, val_idx in self.skf.split(self.features, self.genders):
            train_data = (self.features[train_idx], self.targets[train_idx], self.genders[train_idx])
            val_data = (self.features[val_idx], self.targets[val_idx], self.genders[val_idx])
            folds.append((train_data, val_data))
        return folds

    def get_specs(self) -> DatasetSpecs:
        """Return a summary of dataset dimensions and gender balance.

        Returns:
            :class:`DatasetSpecs` with sample count, feature count, target count,
            and gender distribution.
        """
        return DatasetSpecs(
            num_samples=len(self.features),
            num_features=self.features.shape[1],
            num_targets=self.targets.shape[1] if self.targets.ndim > 1 else 1,
            gender_distribution=dict(Counter(self.genders)),
        )
