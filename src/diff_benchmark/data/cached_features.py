"""
Feature caching system for frozen backbone models.

This module provides functionality to pre-compute and cache features from frozen
backbone models (like ViT and DinoV2) to dramatically speed up training.

Storage format: Parquet files with flat table structure
- subject_id: str - Unique identifier for each subject
- augmentation_idx: int - Index of augmentation (0-9)
- feature_0, feature_1, ..., feature_N: float - Embedding dimensions

Data normalization architecture:
1. Raw MRI data [0, 1] → Normalize(mean=0.5, std=0.5) → Apply augmentations → Cache
2. At training/inference: Cached normalized data → Backbone unnormalizes in forward() → Model processes

Each backbone handles its own input preprocessing in its forward() method:
- ViT/DINOv2/CURIA: Unnormalize to [0, 1] (x = x * 0.5 + 0.5), then HuggingFace processor applies ImageNet normalization
- Future models needing [0, 255]: Can add x = x * 0.5 + 0.5 then x = x * 255.0
- Models working with normalized data: Can keep as-is

This ensures:
- Consistent cached data across all models (normalized with mean=0.5, std=0.5)
- No PIL conversion errors (works on tensors)
- Each backbone adapts input in its forward() method (2 lines of code)
- Easy to add new models with different input requirements
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_cache_key(
    model_name: str,
    model_config: dict,
    dataset_name: str,
    num_augmentations: int,
) -> str:
    """
    Compute a unique cache key based on model and dataset configuration.

    Args:
        model_name: Name of the model (e.g., "vit", "dinov2")
        model_config: Dictionary with model configuration
        dataset_name: Name of the dataset
        num_augmentations: Number of augmented versions per sample

    Returns:
        A unique hash string identifying this cache configuration
    """
    cache_dict = {
        "model_name": model_name,
        "model_config": model_config,
        "dataset_name": dataset_name,
        "num_augmentations": num_augmentations,
    }
    cache_str = json.dumps(cache_dict, sort_keys=True)
    cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:12]
    return f"{model_name}_{dataset_name}_{cache_hash}"


def get_deterministic_rotation_transforms(
    num_rotations: int = 10,
    image_size: Optional[Tuple[int, int]] = None,
    norm_mean: float = 0.5,
    norm_std: float = 0.5,
) -> List:
    """
    Create a list of deterministic transformation (rotations + flips) with normalization.

    Pipeline for cached features:
    1. Raw data [0, 1] → Normalize (mean, std) → Optional Resize → Rotation/Flip
    2. Cache stores normalized data
    3. At inference: Backbone unnormalizes in forward() (x = x * std + mean)
    4. Finally → Features

    This allows:
    - Consistent cached data across all models (normalized)
    - Each backbone adapts input by unnormalizing in its forward method
    - No PIL conversion errors (works on tensors)
    - Normalization parameters configurable via config file

    Args:
        num_rotations: Number of augmentations to generate (including base)
        image_size: Optional (H, W) to resize slices to. Useful for HCP (145×174→256×256)
        norm_mean: Mean for normalization (default 0.5, from config)
        norm_std: Std for normalization (default 0.5, from config)

    Returns:
        List of callable transforms (functions), each applying a different fixed transformation
    """
    transform_list = []

    # Normalization transform (mandatory, applied to all augmentations)
    normalize_transform = transforms.Normalize(mean=[norm_mean], std=[norm_std])

    # First transform: identity (no rotation) with optional resize and normalization
    def make_base_transform(resize_to=image_size, norm=normalize_transform):
        def transform_fn(x):
            # Normalize first (mandatory)
            x = norm(x)
            # Then resize if needed
            if resize_to is not None:
                x = transforms.functional.resize(x, size=resize_to, antialias=True)
            return x

        return transform_fn

    transform_list.append(make_base_transform())

    # Additional transforms with deterministic rotations and flips
    # We use a fixed seed to ensure deterministic behavior across runs
    rng = np.random.RandomState(42)

    for _ in range(num_rotations - 1):
        # Generate random parameters
        angle = rng.uniform(0, 360)
        h_flip = rng.choice([True, False])
        v_flip = rng.choice([True, False])

        # Create closure to capture the values
        def make_combined_transform(
            rotation_angle,
            do_h_flip,
            do_v_flip,
            resize_to=image_size,
            norm=normalize_transform,
        ):
            def transform_fn(x):
                # Normalize first (mandatory)
                x = norm(x)
                # Resize if needed
                if resize_to is not None:
                    x = transforms.functional.resize(x, size=resize_to, antialias=True)
                # Rotate
                x = transforms.functional.rotate(x, angle=rotation_angle)
                # Flips
                if do_h_flip:
                    x = transforms.functional.hflip(x)
                if do_v_flip:
                    x = transforms.functional.vflip(x)
                return x

            return transform_fn

        transform_list.append(make_combined_transform(angle, h_flip, v_flip))

    return transform_list


class CachedFeatureDataset(Dataset):
    """
    Dataset that caches backbone features with multiple augmentations.

    On initialization:
    1. Checks if cache_path exists
    2. If YES: Loads features from disk (fast)
    3. If NO: Runs backbone on source_dataloader, saves features to disk

    Storage format: Parquet file with flat table structure
        - subject_id: str
        - augmentation_idx: int (0 to num_augmentations-1)
        - feature_0, feature_1, ..., feature_N: float (embedding dimensions)

    Each subject will have num_augmentations rows (one per augmentation).
    Labels and genders are fetched from the source dataset at runtime.
    """

    def __init__(
        self,
        cache_path: Path | str,
        backbone: Optional[nn.Module] = None,
        source_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        num_augmentations: int = 10,
        image_size: Optional[Tuple[int, int]] = None,
        norm_mean: float = 0.5,
        norm_std: float = 0.5,
        force_recompute: bool = False,
    ):
        """
        Initialize cached feature dataset.

        Args:
            cache_path: Path where features should be cached (.parquet file)
            backbone: Backbone model (required if cache doesn't exist)
            source_dataloader: DataLoader with original data (required if cache doesn't exist)
            device: Device for computation (required if cache doesn't exist)
            num_augmentations: Number of augmented versions per sample (including base)
            image_size: Optional (H, W) to resize slices. Recommended for HCP: (256, 256)
            norm_mean: Mean for normalization (from config, default 0.5)
            norm_std: Std for normalization (from config, default 0.5)
            force_recompute: If True, recompute features even if cache exists
        """
        self.cache_path = Path(cache_path)
        self.num_augmentations = num_augmentations
        # Convert image_size to plain tuple/list for JSON serialization
        if image_size is not None and hasattr(image_size, "__iter__"):
            self.image_size = (
                tuple(image_size)
                if not isinstance(image_size, (list, tuple))
                else image_size
            )
        else:
            self.image_size = image_size
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.source_dataloader = (
            source_dataloader  # Keep reference for label/gender lookup
        )
        # Ensure .parquet extension
        if self.cache_path.suffix != ".parquet":
            self.cache_path = self.cache_path.with_suffix(".parquet")

        if self.cache_path.exists() and not force_recompute:
            logger.info(f"\n--- Loading Features from Cache: {self.cache_path} ---")
            self._load_from_cache()
        else:
            if backbone is None or source_dataloader is None or device is None:
                raise ValueError(
                    "Cache not found. backbone, source_dataloader, and device are required to generate it."
                )
            logger.info(f"\n--- Cache miss. Computing Features (The Slow Part) ---")
            self._compute_and_cache(backbone, source_dataloader, device)

    def _load_from_cache(self):
        """Load pre-computed features from parquet file."""
        start = time.time()

        df = pd.read_parquet(self.cache_path)

        # Ensure subject_id is string
        df["subject_id"] = df["subject_id"].astype(str)

        # Determine expected subjects from source_dataloader if available
        expected_subjects_list = None
        if self.source_dataloader is not None:
            dataset = self.source_dataloader.dataset
            if hasattr(dataset, "subject_ids"):
                expected_subjects_list = [str(s) for s in dataset.subject_ids]
            elif hasattr(dataset, "_subject_ids"):
                expected_subjects_list = [str(s) for s in dataset._subject_ids]

        # If we have expected subjects, filter and order by them
        if expected_subjects_list is not None:
            # Check for missing subjects
            cached_subjects_set = set(df["subject_id"].unique())
            missing_subjects = [
                s for s in expected_subjects_list if s not in cached_subjects_set
            ]

            if missing_subjects:
                logger.error(
                    f"Cache is missing {len(missing_subjects)} subjects required by the source dataset."
                )
                logger.error(f"Missing: {missing_subjects[:5]}...")
                raise ValueError(
                    f"Cache is missing {len(missing_subjects)} subjects required by the source dataset."
                )

            # Use expected subjects list (preserves order)
            unique_subjects = expected_subjects_list
            # Pre-filter dataframe for faster access
            df = df[df["subject_id"].isin(set(unique_subjects))]
        else:
            # No source dataloader or no subject_ids, use whatever is in cache
            unique_subjects = df["subject_id"].unique().tolist()

        self.subject_ids = unique_subjects

        # Get feature columns
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        self.metadata = {
            "num_samples": len(unique_subjects),
            "num_augmentations": self.num_augmentations,
            "embedding_dim": len(feature_cols),
        }

        # Reorganize into List[List[Tensor]] for efficient access
        # features[sample_idx][aug_idx] = Tensor of shape (embedding_dim,)
        self.features = []

        # Create a dictionary for faster lookup if loading many subjects
        # Group by subject_id first
        df_grouped = df.groupby("subject_id")

        for subject_id in unique_subjects:
            if subject_id not in df_grouped.groups:
                # Should have been caught by missing_subjects check above
                raise ValueError(f"Subject {subject_id} not found in cache group keys!")

            subject_df = df_grouped.get_group(subject_id).sort_values(
                "augmentation_idx"
            )

            # Extract features for all augmentations
            aug_features = []
            for _, row in subject_df.iterrows():
                feature_values = row[feature_cols].values
                # Convert to float64 first to handle object dtype, then to float32 tensor
                feature_values = feature_values.astype(np.float64)
                aug_features.append(torch.tensor(feature_values, dtype=torch.float32))

            self.features.append(aug_features)

        elapsed = time.time() - start
        file_size_mb = os.path.getsize(self.cache_path) / (1024**2)

        logger.info(f"Loaded {len(self.subject_ids)} samples in {elapsed:.4f} seconds.")
        logger.info(f"File size: {file_size_mb:.2f} MB")
        logger.info(f"Each sample has {self.num_augmentations} augmented versions")
        # Verify alignment with source dataset
        self._check_alignment_with_source()

    def _check_alignment_with_source(self):
        """Check alignment between cached subject IDs and source dataset."""
        if self.source_dataloader is None:
            return

        dataset = self.source_dataloader.dataset

        # Check if source dataset has subject_ids attribute
        source_subjects = None
        if hasattr(dataset, "subject_ids"):
            source_subjects = dataset.subject_ids
        elif hasattr(dataset, "_subject_ids"):
            source_subjects = dataset._subject_ids

        if source_subjects is None:
            logger.warning(
                "Could not verify alignment: source dataset has no subject_ids attribute"
            )
            return

        # Standardize to string for comparison
        cached_subjects = [str(s) for s in self.subject_ids]
        source_subjects = [str(s) for s in source_subjects]

        if len(cached_subjects) != len(source_subjects):
            logger.error(
                f"Length mismatch: Cache ({len(cached_subjects)}) vs Source ({len(source_subjects)})"
            )
            raise ValueError(
                f"Cache length mismatch: {len(cached_subjects)} vs {len(source_subjects)}. "
                f"Delete cache at {self.cache_path} to recompute."
            )

        mismatches = []
        for i, (cached, source) in enumerate(zip(cached_subjects, source_subjects)):
            if cached != source:
                mismatches.append((i, cached, source))

        if mismatches:
            logger.error(f"Found {len(mismatches)} subject ID mismatches!")
            for i, c, s in mismatches[:5]:
                logger.error(f"Index {i}: Cache={c}, Source={s}")
            raise ValueError(
                "Subject ID mismatch between cache and source dataset! "
                "The cache file has different subjects or order than the current dataset. "
                "Delete the cache file to recompute."
            )

        logger.info("✓ Verified: Cache subject order matches source dataset order.")

    def _compute_and_cache(
        self,
        backbone: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ):
        """Compute features using backbone and cache them."""
        backbone.eval()
        backbone.to(device)

        # Create augmentation transforms with normalization from config
        aug_transforms = get_deterministic_rotation_transforms(
            self.num_augmentations,
            image_size=self.image_size,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
        )

        all_features = []  # List of lists: [num_samples][num_augmentations]
        all_subject_ids = []

        start_time = time.time()
        total_batches = len(dataloader)

        resize_info = f" with resize to {self.image_size}" if self.image_size else ""
        logger.info(
            f"Computing features with {self.num_augmentations} augmentations per sample{resize_info}..."
        )

        # processed_sample_count = 0  # tracks true offset across dropped/partial batches
        with torch.no_grad():
            with torch.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx % 10 == 0:
                        elapsed = time.time() - start_time
                        logger.info(
                            f"Processing batch {batch_idx}/{total_batches}... (elapsed: {elapsed:.1f}s)"
                        )

                    # # Skip None batches produced by safe_collate when all samples failed
                    # if batch is None:
                    #     logger.warning(f"Skipping empty batch at index {batch_idx} (all samples were None)")
                    #     continue

                    # Unpack batch (might have 2 or 3 elements)
                    if len(batch) >= 2:
                        x = batch[0]
                    else:
                        x = batch

                    # Get subject IDs — use a running counter, not batch_idx * batch_size,
                    # because earlier batches may have been smaller (dropped None samples).
                    if hasattr(dataloader.dataset, "_subject_ids"):
                        batch_start = batch_idx * dataloader.batch_size
                        # batch_start = processed_sample_count
                        batch_end = min(
                            batch_start + len(x), len(dataloader.dataset._subject_ids)
                        )
                        subject_ids = dataloader.dataset._subject_ids[
                            batch_start:batch_end
                        ]
                    else:
                        subject_ids = [
                            f"sample_{batch_idx * dataloader.batch_size + i}"
                            for i in range(len(x))
                        ]
                    #     subject_ids = [f"sample_{processed_sample_count + i}" for i in range(len(x))]

                    # processed_sample_count += len(x)

                    # Process each sample in batch
                    for sample_idx in range(len(x)):
                        sample_x = x[sample_idx : sample_idx + 1].to(device)

                        # Squeeze channel dim if present (N, 1, D, H, W) -> (N, D, H, W)
                        if sample_x.ndim == 5 and sample_x.shape[1] == 1:
                            sample_x = sample_x.squeeze(1)

                        # Apply each augmentation and compute features
                        sample_features = []
                        for aug_idx, transform in enumerate(aug_transforms):
                            if hasattr(backbone, "collate_with_augmentation"):
                                # If backbone has custom collation (e.g., MedicalNet), use it
                                # It expects a list of (x, y, g) tuples and handles transforms + stacking

                                # We need to pass unwrapped tensor (D, H, W) not (1, D, H, W)
                                x_input = (
                                    sample_x[0] if sample_x.ndim == 4 else sample_x
                                )

                                # Create dummy batch with single sample
                                dummy_batch = [
                                    (x_input, torch.tensor(0), torch.tensor(0))
                                ]

                                # Use backbone's collate function
                                augmented_x, _, _ = backbone.collate_with_augmentation(
                                    dummy_batch, transform=transform
                                )
                                augmented_x = augmented_x.to(device)

                            else:
                                # Apply transformation slice-wise for 3D volumes
                                if sample_x.ndim == 4:  # (1, D, H, W)
                                    D, H, W = sample_x.shape[1:]
                                    augmented_slices = []
                                    for d in range(D):
                                        slice_2d = sample_x[0, d, :, :].unsqueeze(
                                            0
                                        )  # (1, H, W)
                                        aug_slice = transform(
                                            slice_2d
                                        )  # Should output (1, H', W')
                                        # Ensure it's 3D (1, H, W) not 4D
                                        if aug_slice.ndim == 4:
                                            aug_slice = aug_slice.squeeze(
                                                1
                                            )  # Remove extra channel: (1, 1, H, W) -> (1, H, W)
                                        augmented_slices.append(aug_slice)
                                    augmented_x = torch.stack(
                                        augmented_slices, dim=1
                                    )  # (1, D, H', W')
                                else:
                                    # Fallback for other dimensions
                                    augmented_x = transform(sample_x)

                            # Compute features with backbone (backbone handles normalization)
                            features = backbone(augmented_x)  # (1, embedding_dim)
                            sample_features.append(
                                features.squeeze(0).cpu()
                            )  # (embedding_dim,)

                        all_features.append(sample_features)
                        all_subject_ids.append(subject_ids[sample_idx])

        self.subject_ids = all_subject_ids
        self.features = all_features  # Keep as list of lists

        # Create metadata
        self.metadata = {
            "num_samples": len(all_features),
            "num_augmentations": self.num_augmentations,
            "embedding_dim": all_features[0][0].shape[0],
            "image_size": self.image_size,
            "timestamp": time.time(),
        }

        elapsed = time.time() - start_time
        logger.info(
            f"\nFeature extraction finished in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)."
        )

        # Save to disk as parquet
        logger.info(f"Saving features to {self.cache_path}...")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_to_parquet(all_features, all_subject_ids)

        file_size_mb = os.path.getsize(self.cache_path) / (1024**2)
        logger.info(
            f"Saved. Future runs will be instant. (Cache size: {file_size_mb:.2f} MB)"
        )

    def _save_to_parquet(self, all_features, all_subject_ids):
        """Save features to parquet file (columnar format, compressed)."""
        # Flatten the structure into a table
        rows = []
        embedding_dim = all_features[0][0].shape[0]

        for sample_idx, (sample_features, subject_id) in enumerate(
            zip(all_features, all_subject_ids)
        ):
            for aug_idx, features in enumerate(sample_features):
                row = {
                    "subject_id": subject_id,
                    "augmentation_idx": aug_idx,
                }
                # Add feature columns
                for feat_idx, feat_val in enumerate(features.numpy()):
                    row[f"feature_{feat_idx}"] = float(feat_val)

                rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Save with compression (snappy is fast and gives good compression)
        df.to_parquet(
            self.cache_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        # Store metadata separately for quick access
        metadata_path = self.cache_path.with_suffix(".meta.json")

        # Convert image_size to plain list for JSON serialization (handles ListConfig from Hydra)
        image_size_json = None
        if self.image_size is not None:
            if isinstance(self.image_size, (list, tuple)):
                image_size_json = list(self.image_size)
            elif hasattr(
                self.image_size, "__iter__"
            ):  # Handle ListConfig or other iterables
                image_size_json = list(self.image_size)
            else:
                image_size_json = self.image_size

        metadata = {
            "num_samples": len(all_features),
            "num_augmentations": self.num_augmentations,
            "embedding_dim": embedding_dim,
            "image_size": image_size_json,
            "timestamp": time.time(),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def __len__(self) -> int:
        return len(self.features)

    @property
    def subject_ids(self) -> list:
        """List of subject IDs aligned with dataset indices."""
        if hasattr(self, "_subject_ids"):
            return self._subject_ids
        else:
            return [f"subject_{i}" for i in range(len(self))]

    @subject_ids.setter
    def subject_ids(self, value: list):
        """Set subject IDs."""
        self._subject_ids = value

    @property
    def targets(self):
        """
        Targets/labels for the dataset.
        Always returns torch tensors for consistency with CustomDataset.
        Automatically converts numpy arrays to tensors if needed.
        """
        if hasattr(self, "_targets"):
            # If it's a numpy array, convert to tensor
            if isinstance(self._targets, np.ndarray):
                return torch.tensor(self._targets, dtype=torch.float32)
            # If already a tensor, return as-is
            return self._targets
        else:
            return None

    @targets.setter
    def targets(self, value):
        """Set targets (stores as-is, conversion happens in getter)."""
        self._targets = value

    @property
    def gender(self):
        """
        Gender labels for the dataset.
        Always returns torch tensors for consistency with CustomDataset.
        Automatically converts numpy arrays to tensors if needed.
        """
        if hasattr(self, "_genders"):
            # If it's a numpy array, convert to tensor
            if isinstance(self._genders, np.ndarray):
                return torch.tensor(self._genders, dtype=torch.long)
            # If already a tensor, return as-is
            return self._genders
        else:
            return None

    @gender.setter
    def gender(self, value):
        """Set gender (stores as-is, conversion happens in getter)."""
        self._genders = value

    def set_augmentation_indices(
        self, indices: Optional[np.ndarray] = None, mode: str = "random"
    ):
        """
        Set augmentation indices for consistent training.

        Args:
            indices: Optional array of augmentation indices per sample. If None, randomly generated.
            mode: "random" (randomly select one aug per sample), "fixed" (always use aug_idx=0),
                  or "custom" (use provided indices)

        Usage:
            # For training: randomly select one augmentation per subject, keep consistent
            dataset.set_augmentation_indices(mode="random")

            # For validation/testing: always use original (non-transformed)
            dataset.set_augmentation_indices(mode="fixed")
        """
        if mode == "fixed":
            # Always use augmentation_idx=0 (original/non-transformed)
            self.aug_indices = np.zeros(len(self), dtype=int)
        elif mode == "random":
            # Randomly select one augmentation per sample (for training)
            self.aug_indices = np.random.randint(
                0, self.num_augmentations, size=len(self)
            )
        elif mode == "custom" and indices is not None:
            # Use provided indices
            assert len(indices) == len(self), "indices must match dataset length"
            self.aug_indices = indices
        else:
            # Default: random selection per __getitem__ call (original behavior)
            self.aug_indices = None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample with augmentation.

        If set_augmentation_indices() was called, uses the pre-selected augmentation.
        Otherwise, randomly selects an augmentation (original behavior).

        Returns:
            features: (embedding_dim,)
            label: scalar tensor
            gender: scalar tensor
        """
        # Use pre-selected augmentation index if available, otherwise random
        if hasattr(self, "aug_indices") and self.aug_indices is not None:
            aug_idx = self.aug_indices[idx]
        else:
            # Original behavior: random selection per call
            aug_idx = torch.randint(0, self.num_augmentations, (1,)).item()

        features = self.features[idx][aug_idx]

        # Fetch label and gender - prefer stored attributes over source_dataloader
        # Handle both numpy arrays (from DatasetPreparation) and tensors
        if hasattr(self, "_targets") and hasattr(self, "_genders"):
            # Use pre-stored numpy arrays (set by DatasetPreparation)
            # Convert to tensors lazily per-sample
            label = torch.tensor(self._targets[idx], dtype=torch.float32)
            gender = torch.tensor(self._genders[idx], dtype=torch.long)
        elif hasattr(self, "targets") and hasattr(self, "genders"):
            # Legacy: if already tensors
            label = self.targets[idx]
            gender = self.genders[idx]
        elif self.source_dataloader is not None and hasattr(
            self.source_dataloader.dataset, "__getitem__"
        ):
            # Fallback: fetch from source dataset
            source_sample = self.source_dataloader.dataset[idx]
            if len(source_sample) >= 2:
                label = source_sample[1]
                gender = (
                    source_sample[2] if len(source_sample) >= 3 else torch.tensor(0)
                )
            else:
                label = torch.tensor(0.0)
                gender = torch.tensor(0)
        else:
            # Last resort fallback
            label = torch.tensor(0.0)
            gender = torch.tensor(0)

        return features, label, gender

    def get_specific_augmentation(
        self,
        idx: int,
        aug_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample with a specific augmentation index.

        Args:
            idx: Sample index
            aug_idx: Augmentation index (0 = original normalized, 1-N = rotations)

        Returns:
            features, label, gender
        """
        features = self.features[idx][aug_idx]

        # Fetch label and gender - same logic as __getitem__
        if hasattr(self, "_targets") and hasattr(self, "_genders"):
            label = torch.tensor(self._targets[idx], dtype=torch.float32)
            gender = torch.tensor(self._genders[idx], dtype=torch.long)
        elif hasattr(self, "targets") and hasattr(self, "genders"):
            label = self.targets[idx]
            gender = self.genders[idx]
        elif self.source_dataloader is not None and hasattr(
            self.source_dataloader.dataset, "__getitem__"
        ):
            source_sample = self.source_dataloader.dataset[idx]
            if len(source_sample) >= 2:
                label = source_sample[1]
                gender = (
                    source_sample[2] if len(source_sample) >= 3 else torch.tensor(0)
                )
            else:
                label = torch.tensor(0.0)
                gender = torch.tensor(0)
        else:
            label = torch.tensor(0.0)
            gender = torch.tensor(0)

        return features, label, gender


def get_cache_path(
    model_name: str,
    dataset_name: str,
    cache_dir: Path | str = "data/cached_features",
    image_size: Optional[Tuple[int, int]] = None,
    tissue_type: Optional[str] = None,
    metric_to_compute: Optional[str] = None,
) -> Path:
    """
    Get the path for cached features.

    Args:
        model_name: Name of the model
        dataset_name: Dataset name
        cache_dir: Base directory for caching
        image_size: Optional (H, W) for resizing. Included in filename to separate versions.
        tissue_type: Optional tissue type (e.g., "gray", "white"). Included in filename.
        metric_to_compute: Optional microstructure metric (e.g., "md", "fa", "sh"). Included in filename.

    Returns:
        Path to cache file (.parquet)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build filename with tissue_type, metric_to_compute, image_size
    parts = [model_name, dataset_name]

    # Add tissue type if specified
    if tissue_type is not None:
        parts.append(tissue_type)

    # Add microstructure metric if specified
    if metric_to_compute is not None:
        parts.append(metric_to_compute)

    # Add image size
    if image_size is not None:
        parts.append(f"{image_size[0]}x{image_size[1]}")
    else:
        parts.append("original")

    filename = "_".join(parts) + "_features.parquet"
    return cache_dir / filename


def append_augmentations_to_cache(
    cache_path: Path | str,
    backbone: nn.Module,
    source_dataloader: DataLoader,
    device: torch.device,
    new_transforms: List,
    start_aug_idx: Optional[int] = None,
    image_size: Optional[Tuple[int, int]] = None,
):
    """
    Append new augmentations to an existing cache file without recomputing existing ones.

    Args:
        cache_path: Path to existing cache file
        backbone: Backbone model for feature extraction
        source_dataloader: DataLoader with original data
        device: Device for computation
        new_transforms: List of new transform functions to apply
        start_aug_idx: Starting index for new augmentations (auto-detected if None)
        image_size: Optional (H, W) for resizing slices (should match original cache)
    """
    cache_path = Path(cache_path)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    logger.info(f"Appending {len(new_transforms)} new augmentations to {cache_path}")

    # Load existing cache
    df_existing = pd.read_parquet(cache_path)

    # Determine starting augmentation index
    if start_aug_idx is None:
        start_aug_idx = df_existing["augmentation_idx"].max() + 1

    logger.info(f"Existing augmentations: 0-{start_aug_idx-1}")
    logger.info(
        f"New augmentations: {start_aug_idx}-{start_aug_idx + len(new_transforms) - 1}"
    )

    # Get subject IDs from existing cache
    existing_subject_ids = df_existing["subject_id"].unique().tolist()

    backbone.eval()
    backbone.to(device)

    new_rows = []
    start_time = time.time()
    total_batches = len(source_dataloader)
    # processed_sample_count = 0  # running offset, not batch_idx * batch_size

    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            for batch_idx, batch in enumerate(source_dataloader):
                if batch_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Processing batch {batch_idx}/{total_batches}... (elapsed: {elapsed:.1f}s)"
                    )

                # # Skip None batches produced by safe_collate
                # if batch is None:
                #     logger.warning(f"Skipping empty batch at index {batch_idx}")
                #     continue

                # Unpack batch
                if len(batch) >= 2:
                    x = batch[0]
                else:
                    x = batch

                # Get subject IDs using running counter
                if hasattr(source_dataloader.dataset, "_subject_ids"):
                    batch_start = batch_idx * source_dataloader.batch_size
                    # batch_start = processed_sample_count
                    batch_end = min(
                        batch_start + len(x),
                        len(source_dataloader.dataset._subject_ids),
                    )
                    subject_ids = source_dataloader.dataset._subject_ids[
                        batch_start:batch_end
                    ]
                else:
                    subject_ids = [
                        f"sample_{batch_idx * source_dataloader.batch_size + i}"
                        for i in range(len(x))
                    ]
                #     subject_ids = [f"sample_{processed_sample_count + i}" for i in range(len(x))]

                # processed_sample_count += len(x)

                # Process each sample
                for sample_idx in range(len(x)):
                    sample_x = x[sample_idx : sample_idx + 1].to(device)
                    subject_id = subject_ids[sample_idx]

                    # Skip if not in original cache (in case dataset changed)
                    if subject_id not in existing_subject_ids:
                        continue

                    # Squeeze channel dim if present
                    if sample_x.ndim == 5 and sample_x.shape[1] == 1:
                        sample_x = sample_x.squeeze(1)

                    # Apply new transformations only
                    for transform_idx, transform in enumerate(new_transforms):
                        aug_idx = start_aug_idx + transform_idx

                        if hasattr(backbone, "collate_with_augmentation"):
                            # If backbone has custom collation (e.g., MedicalNet), use it
                            x_input = sample_x[0] if sample_x.ndim == 4 else sample_x
                            dummy_batch = [(x_input, torch.tensor(0), torch.tensor(0))]
                            augmented_x, _, _ = backbone.collate_with_augmentation(
                                dummy_batch, transform=transform
                            )
                            augmented_x = augmented_x.to(device)
                        elif sample_x.ndim == 4:  # (1, D, H, W)
                            # Apply transformation slice-wise for 3D volumes
                            D, H, W = sample_x.shape[1:]
                            augmented_slices = []
                            for d in range(D):
                                slice_2d = sample_x[0, d, :, :].unsqueeze(
                                    0
                                )  # (1, H, W)
                                # Apply transform (includes normalization and optional resize/rotation)
                                aug_slice = transform(
                                    slice_2d
                                )  # Should output (1, H', W')
                                # Ensure it's 3D (1, H, W) not 4D
                                if aug_slice.ndim == 4:
                                    aug_slice = aug_slice.squeeze(
                                        1
                                    )  # Remove extra channel: (1, 1, H, W) -> (1, H, W)
                                augmented_slices.append(aug_slice)
                            augmented_x = torch.stack(
                                augmented_slices, dim=1
                            )  # (1, D, H', W')
                        else:
                            # Fallback for other dimensions
                            augmented_x = transform(sample_x)

                        # Compute features
                        features = backbone(augmented_x)
                        features_np = features.squeeze(0).cpu().numpy()

                        # Create row
                        row = {
                            "subject_id": subject_id,
                            "augmentation_idx": aug_idx,
                        }
                        for feat_idx, feat_val in enumerate(features_np):
                            row[f"feature_{feat_idx}"] = float(feat_val)

                        new_rows.append(row)

    # Create DataFrame with new augmentations
    df_new = pd.DataFrame(new_rows)

    # Concatenate with existing data
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # Save back to file
    df_combined.to_parquet(
        cache_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    # Update metadata
    metadata_path = cache_path.with_suffix(".meta.json")
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata.update(
        {
            "num_augmentations": start_aug_idx + len(new_transforms),
            "last_updated": time.time(),
        }
    )

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    new_size_mb = cache_path.stat().st_size / (1024**2)

    logger.info(f"\n✓ Appended {len(new_transforms)} augmentations")
    logger.info(f"✓ Total augmentations: {start_aug_idx + len(new_transforms)}")
    logger.info(f"✓ Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    logger.info(f"✓ New cache size: {new_size_mb:.1f} MB")
