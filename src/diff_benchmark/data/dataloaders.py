from collections import Counter
from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pandas as pd
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
        
        prediction_task = self.config.pred_head.prediction_task
        if prediction_task == "binary_classification":
            self.stratify_labels = self.targets.ravel()
        else:
            self.stratify_labels = self.genders

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
        # indices = list(self.skf.split(np.zeros(len(self.genders)), self.genders))
        indices = list(self.skf.split(np.zeros(len(self.genders)), self.stratify_labels))
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

    def safe_collate(self, batch: list) -> Any:
        """Collate function that filters ``None`` samples and loads mesh data.

        All samples carry a three-tuple ``(X, target, gender)`` where ``X`` is
        either a float tensor (array / image pipelines) or a dict
        ``{"nodes_path": Path, "edges_path": Path}`` (mesh pipeline).

        Collation rules:

        - ``None`` batch entries (failed I/O reads) are silently dropped.
        - If ``X`` is a **tensor** the full batch is collated with PyTorch's
          default collate: ``(X_tensor, y_tensor, gender_tensor)``.
        - If ``X`` is a **dict** (mesh mode) both ``nodes.parquet`` and
          ``edges.parquet`` are loaded here for each sample. ``X_batch`` is
          returned as a **list of mesh dicts**, one per sample::

              {
                  "node_features":  FloatTensor (N, F),
                  "vertices":       FloatTensor (N, 3),
                  "parcel_labels":  LongTensor  (N,),
                  "hemisphere":     LongTensor  (N,),   # 0=LH, 1=RH
                  "edge_index":     LongTensor  (2, E),
              }
                  "vertices":       FloatTensor (N, 3),
                  "parcel_labels":  LongTensor  (N,),
                  "edge_index":     LongTensor  (2, E),
              }

          ``y`` and ``gender`` are collated into tensors normally.

        Args:
            batch: List of samples from :meth:`Dataset.__getitem__`.

        Returns:
            Three-tuple ``(X, y, gender)`` where ``X`` is either a stacked
            tensor (array/image) or a list of mesh dicts (mesh pipeline).
        """
        from torch.utils.data.dataloader import default_collate

        # Drop failed samples
        batch = [b for b in batch if b is not None]
        if not batch:
            return default_collate([])

        # All samples are 3-tuples: (X, y, gender)
        first_x = batch[0][0]

        if isinstance(first_x, dict):
            # Mesh mode — load both parquets and build a rich dict per sample
            mesh_batch: List[dict] = []

            for sample in batch:
                paths = sample[0]

                # ---- nodes parquet ----
                nodes_df = pd.read_parquet(paths["nodes_path"], engine="pyarrow")
                feat_cols = sorted(
                    [c for c in nodes_df.columns if c.startswith("feature_")],
                    key=lambda c: int(c.split("_")[1]),
                )
                node_features = torch.tensor(
                    nodes_df[feat_cols].to_numpy(dtype="float32"),
                    dtype=torch.float32,
                )  # (N, F)
                vertices = torch.tensor(
                    nodes_df[["x", "y", "z"]].to_numpy(dtype="float32"),
                    dtype=torch.float32,
                )  # (N, 3)
                parcel_labels = torch.tensor(
                    nodes_df["parcel_label"].to_numpy(dtype="int32"),
                    dtype=torch.long,
                )  # (N,)

                # Hemisphere indicator: 0 = LH, 1 = RH.
                # Present in newly-exported parquets; fall back to midpoint split
                # for legacy files that pre-date the hemisphere column.
                if "hemisphere" in nodes_df.columns:
                    hemisphere = torch.tensor(
                        nodes_df["hemisphere"].to_numpy(dtype="int8"),
                        dtype=torch.long,
                    )  # (N,)
                else:
                    n_total = len(nodes_df)
                    hemi_np = np.zeros(n_total, dtype=np.int8)
                    hemi_np[n_total // 2 :] = 1
                    hemisphere = torch.from_numpy(hemi_np).long()  # (N,)

                # --- Ensure RH parcel labels are globally unique --------
                # Parquets written before the build_parcel_label_vector fix
                # store RH labels in 1–K (same range as LH), causing _pool_parcels
                # to collapse 1000 Schaefer parcels into 500.  Detect this by
                # checking whether max(RH) > max(LH); if not, apply the offset
                # here so the model always sees globally unique label IDs.
                rh_mask = hemisphere == 1
                lh_mask = hemisphere == 0
                if rh_mask.any() and lh_mask.any():
                    max_lh = int(parcel_labels[lh_mask].max())
                    max_rh = int(parcel_labels[rh_mask & (parcel_labels > 0)].max()) if (rh_mask & (parcel_labels > 0)).any() else 0
                    if max_rh > 0 and max_rh <= max_lh:
                        # Old encoding — apply offset in-place on a clone
                        parcel_labels = parcel_labels.clone()
                        parcel_labels[rh_mask & (parcel_labels > 0)] += max_lh

                # ---- edges parquet ----
                edges_df = pd.read_parquet(paths["edges_path"], engine="pyarrow")
                edge_index = torch.tensor(
                    edges_df[["src", "dst"]].to_numpy(dtype="int64").T,
                    dtype=torch.long,
                )  # (2, E)

                mesh_batch.append({
                    "node_features": node_features,
                    "vertices":      vertices,
                    "parcel_labels": parcel_labels,
                    "hemisphere":    hemisphere,
                    "edge_index":    edge_index,
                })

            y_batch = default_collate([b[1] for b in batch])
            gender_batch = default_collate([b[2] for b in batch])
            return mesh_batch, y_batch, gender_batch

        # Array / image mode — default collate handles everything
        return default_collate(batch)

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
        # for train_idx, val_idx in self.skf.split(self.features, self.genders):
        for train_idx, val_idx in self.skf.split(self.features, self.stratify_labels):
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
        # for train_idx, val_idx in self.skf.split(self.features, self.genders):
        for train_idx, val_idx in self.skf.split(self.features, self.stratify_labels):
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