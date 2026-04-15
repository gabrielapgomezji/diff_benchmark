from pathlib import Path
from typing import List, Tuple, Union

import bids
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from diff_benchmark.data.cached_features import CachedFeatureDataset, get_cache_path
from diff_benchmark.data.dataloaders import PreprocessedData
from diff_benchmark.data.generate_dataset import CustomDataset
from diff_benchmark.models.model_configurations import get_model
from diff_benchmark.preprocessing.brain_feature_extraction import (
    DefaultPipeline,
    ImagePipeline,
    MeshPipeline,
)
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.preprocessing.preparation_pipeline import (
    BrainDataPreparationPipeline,
    DemographicsPreparationPipeline,
)
from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


def _parse_image_size(resize_shape) -> Union[Tuple[int, int], None]:
    """
    Parse image_size from config, handling Hydra's string conversion issues.

    Args:
        resize_shape: Value from config (can be None, list, tuple, or string)

    Returns:
        Tuple of (H, W) or None
    """
    if resize_shape is None:
        return None

    if isinstance(resize_shape, (list, tuple)):
        # Check if it's a list containing None or string "None"
        if len(resize_shape) == 1 and (
            resize_shape[0] is None or resize_shape[0] == "None"
        ):
            return None
        return tuple(resize_shape)

    # If it's a string "None", return None
    if isinstance(resize_shape, str) and resize_shape == "None":
        return None

    return resize_shape


def get_data_pipeline(
    data_type: str, dataset: DatasetConfig
) -> BrainDataPreparationPipeline:
    """Factory returning the appropriate data pipeline for *data_type*.

    Args:
        data_type: One of ``'images'``, ``'array'``, or ``'mesh'``.
        dataset: Dataset configuration.

    Returns:
        An instance of the selected :class:`BrainDataPreparationPipeline`.

    Raises:
        ValueError: If *data_type* is not recognized.
    """
    if data_type == "images":
        logger.info("Using Image Pipeline")
        print("Using Image Pipeline")
        brain_preparator = ImagePipeline(dataset)
    elif data_type == "array":
        logger.info("Using Default Array Pipeline")
        print("Using Default Array Pipeline")
        brain_preparator = DefaultPipeline(dataset)
    elif data_type == "mesh":
        logger.info("Using Mesh Pipeline (surface graph representation)")
        print("Using Mesh Pipeline (surface graph representation)")
        surface_type = getattr(dataset, "mesh_surface_type", "midthickness")
        brain_preparator = MeshPipeline(dataset, surface_type=surface_type)
    else:
        raise ValueError(
            f"Unknown data_type '{data_type}'. Must be one of ['images', 'array', 'mesh']."
        )

    return brain_preparator


class DatasetPreparation:
    """
    End-to-end data preparation:
    - Extract microstructure features
    - Extract and preprocess demographics
    - Align subjects
    - Build dataset and preprocessed objects
    """

    def __init__(
        self,
        cfg: DictConfig,
        source_dataset: DatasetConfig,
    ):
        self.cfg = cfg
        self.model_name = cfg.model.name
        self.source_dataset = source_dataset

    def _should_use_cache(self) -> bool:
        """Return ``True`` if this model is cacheable and ``freeze_backbone=True``."""
        # Only cache for heavy pretrained models
        cacheable_models = ["vit", "dinov2", "curia", "pointnet"]  # , "medicalnet"]

        if self.model_name not in cacheable_models:
            return False

        # Check if freeze_backbone is True
        freeze_backbone = self.cfg.model.backbone.get("freeze_backbone", False)

        # Special handling for medicalnet: treat as cacheable even if freeze_backbone missing
        # (user can override by explicitly setting freeze_backbone=False)
        if (
            self.model_name == "medicalnet"
            and "freeze_backbone" not in self.cfg.model.backbone
            and self.cfg.model.backbone.pretrained == True
        ):
            freeze_backbone = True

        # PointNet caching follows the same frozen-backbone workflow as vision backbones.
        # If the key is absent in config, default to cache-enabled behavior.
        if self.model_name == "pointnet" and "freeze_backbone" not in self.cfg.model.backbone:
            freeze_backbone = True

        if not freeze_backbone:
            logger.debug(
                f"Model {self.model_name} has freeze_backbone=False, not using cache"
            )
            return False

        logger.info(
            f"Model {self.model_name} with freeze_backbone=True → will use cache"
        )
        return True

    def _get_cache_info(self) -> Tuple[Path, bool, int, int]:
        """Return ``(cache_path, exists, required_augs, cached_augs)``."""
        cache_dir = Path(self.cfg.cluster.paths.cache_dir) / "dl_features"

        # Get image_size from config to ensure separate caches for resized/non-resized
        image_size = _parse_image_size(self.cfg.data.get("resize_shape"))
        if self.model_name in {"pointnet", "pointnet_pp", "region_pointnet"}:
            image_size = None

        # Get tissue_type from dataset config
        tissue_type = (
            self.source_dataset.tissue_type
            if hasattr(self.source_dataset, "tissue_type")
            else None
        )

        # Get microstructure metric from dataset config
        metric_to_compute = (
            self.source_dataset.metric_to_compute
            if hasattr(self.source_dataset, "metric_to_compute")
            else None
        )

        # Determine model name for cache key (handle variants)
        model_name_for_cache = self.model_name
        if self.model_name == "medicalnet":
            # Append depth to model name for unique caching
            depth = self.cfg.model.backbone.get("depth")
            if depth:
                model_name_for_cache = f"{self.model_name}_depth{depth}"

        cache_path = get_cache_path(
            model_name_for_cache,
            self.source_dataset.name,
            cache_dir,
            image_size=image_size,
            tissue_type=tissue_type,
            metric_to_compute=metric_to_compute,
        )

        # Backward compatibility: older mesh caches included image-size suffix.
        if (
            self.model_name in {"pointnet", "pointnet_pp", "region_pointnet"}
            and not cache_path.exists()
        ):
            legacy_image_size = _parse_image_size(self.cfg.data.get("resize_shape"))
            legacy_parts = [model_name_for_cache, self.source_dataset.name]
            if tissue_type is not None:
                legacy_parts.append(tissue_type)
            if metric_to_compute is not None:
                legacy_parts.append(metric_to_compute)
            if legacy_image_size is not None:
                legacy_parts.append(f"{legacy_image_size[0]}x{legacy_image_size[1]}")
            else:
                legacy_parts.append("original")
            legacy_cache_path = cache_dir / ("_".join(legacy_parts) + "_features.parquet")
            if legacy_cache_path.exists():
                logger.info(
                    "Using legacy mesh cache path with resize suffix: %s",
                    legacy_cache_path,
                )
                cache_path = legacy_cache_path

        required_augs = self.cfg.data.num_augmentations
        if self.model_name in {"pointnet"}:
            required_augs = 1

        if not cache_path.exists():
            return cache_path, False, required_augs, 0

        # Check how many augmentations are in the cache
        import json

        meta_path = cache_path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            cached_augs = metadata.get("num_augmentations", 0)
        else:
            # Fallback: read from parquet
            df = pd.read_parquet(cache_path)
            cached_augs = df["augmentation_idx"].max() + 1

        return cache_path, True, required_augs, cached_augs

    def _compute_or_update_cache(
        self,
        cache_path: Path,
        cache_exists: bool,
        required_augs: int,
        cached_augs: int,
        regular_dataset: CustomDataset,
    ):
        """
        Compute cache if missing, or update if more augmentations needed.

        Args:
            cache_path: Path to cache file
            cache_exists: Whether cache currently exists
            required_augs: Number of augmentations required
            cached_augs: Number of augmentations already cached
            regular_dataset: The regular dataset (for computing features)
        """
        from diff_benchmark.data.cached_features import append_augmentations_to_cache

        cache_batch_size = self.cfg.data.get("cache_batch_size", self.cfg.data.batch_size)
        if self.model_name in {"pointnet"}:
            cache_batch_size = self.cfg.data.get("mesh_cache_batch_size", 1)

        collate_fn = None
        if self.model_name in {"pointnet", "pointnet_pp", "region_pointnet"}:
            # Reuse mesh-safe collate to materialize parquet paths into tensors.
            mesh_collator = PreprocessedData(
                np.zeros((1, 1), dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=np.int64),
                config=self.cfg,
            )
            collate_fn = mesh_collator.safe_collate

        if not cache_exists:
            # Need to compute from scratch
            logger.info(f"Cache not found. Computing {required_augs} augmentations...")
            print(
                f"⚠️  Cache not found for {self.model_name} on {self.source_dataset.name}"
            )
            print(
                f"    Computing {required_augs} augmentations (this will take some time)..."
            )

            # Create backbone model
            model = get_model(
                self.model_name,
                OmegaConf.to_container(self.cfg, resolve=True),
            )
            if hasattr(model, "model"):
                model = model.model
            backbone = model.backbone if hasattr(model, "backbone") else model

            # Create dataloader
            dataloader = DataLoader(
                regular_dataset,
                batch_size=cache_batch_size,
                shuffle=False,
                num_workers=0,  # No multiprocessing for caching
                collate_fn=collate_fn,
            )

            # Get device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Get normalization parameters
            norm_mean = self.cfg.data.normalization.mean
            norm_std = self.cfg.data.normalization.std
            image_size = _parse_image_size(self.cfg.data.get("resize_shape"))
            if self.model_name in {"pointnet", "pointnet_pp", "region_pointnet"}:
                image_size = None

            # Create cached dataset (which computes features)
            _ = CachedFeatureDataset(
                cache_path=cache_path,
                backbone=backbone,
                source_dataloader=dataloader,
                device=device,
                num_augmentations=required_augs,
                image_size=image_size,
                norm_mean=norm_mean,
                norm_std=norm_std,
                force_recompute=False,
            )

            logger.info(f"✓ Cache created with {required_augs} augmentations")
            print(f"✓ Cache created successfully")

        elif cached_augs < required_augs:
            # Need to add more augmentations
            missing_augs = required_augs - cached_augs
            logger.info(f"Cache has {cached_augs} augmentations, need {required_augs}")
            logger.info(f"Computing {missing_augs} additional augmentations...")
            print(
                f"⚠️  Cache has {cached_augs} augmentations, but {required_augs} required"
            )
            print(f"    Computing {missing_augs} additional augmentations...")

            # Create backbone model
            model = get_model(
                self.model_name,
                OmegaConf.to_container(self.cfg, resolve=True),
            )
            if hasattr(model, "model"):
                model = model.model
            backbone = model.backbone if hasattr(model, "backbone") else model

            # Create dataloader
            dataloader = DataLoader(
                regular_dataset,
                batch_size=cache_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image_size = _parse_image_size(self.cfg.data.get("resize_shape"))

            # Create new transforms for additional augmentations
            from diff_benchmark.data.cached_features import (
                get_deterministic_rotation_transforms,
            )

            norm_mean = self.cfg.data.normalization.mean
            norm_std = self.cfg.data.normalization.std

            all_transforms = get_deterministic_rotation_transforms(
                num_rotations=required_augs,
                image_size=image_size,
                norm_mean=norm_mean,
                norm_std=norm_std,
            )
            new_transforms = all_transforms[cached_augs:]  # Only the new ones

            # Append to cache
            append_augmentations_to_cache(
                cache_path=cache_path,
                backbone=backbone,
                source_dataloader=dataloader,
                device=device,
                new_transforms=new_transforms,
                start_aug_idx=cached_augs,
                image_size=image_size,
            )

            logger.info(
                f"✓ Added {missing_augs} augmentations (now {required_augs} total)"
            )
            print(f"✓ Cache updated successfully")
        else:
            # Cache is complete
            logger.info(
                f"✓ Cache found with {cached_augs} augmentations (using {required_augs})"
            )
            print(f"✓ Using cached features ({cached_augs} augmentations available)")

    def _extract_participants_files_from_layouts(
        self,
        layouts: List[bids.BIDSLayout],
    ) -> Union[str, List[str]]:
        """
        Returns a single participants.tsv path or a list (multicenter).
        Args:
            layouts (List[bids.BIDSLayout]): List of BIDS layouts to search.
        Returns:
            Union[str, List[str]]: Path(s) to participants.tsv file(s).
        Raises:
            RuntimeError: If no participants.tsv files are found.
        """
        participants_files = []

        for layout in layouts:
            participants = layout.get_file("participants.tsv")
            if participants is not None:
                participants_files.append(participants.path)

        if not participants_files:
            raise RuntimeError("No participants.tsv found in any BIDS layout")

        return (
            participants_files if len(participants_files) > 1 else participants_files[0]
        )

    def _get_brain_df(self) -> pd.DataFrame:
        """Load and return the brain features DataFrame.

        For the mesh pipeline the DataFrame contains only a ``subject_id`` index
        column (mesh objects live in ``self.brain_preparator.results``); for all
        other pipelines the full feature matrix is returned.
        """
        # -------- MODEL & PIPELINE --------
        model = get_model(
            self.model_name,
            OmegaConf.to_container(self.cfg, resolve=True),
        )

        if hasattr(model, "model"):  # trainer wrapper
            model = model.model
        data_type = model.data_type
        
        self.brain_preparator = get_data_pipeline(data_type, self.source_dataset)
        brain_df = self.brain_preparator.load_features().reset_index()
        return brain_df

    def _get_demographics_df(self) -> pd.DataFrame:
        """Load and return the demographics DataFrame for the configured target column."""
        # -------- DEMOGRAPHICS --------
        if self.source_dataset.name == "hcp":
            cog_file = self.cfg.cluster.paths[self.source_dataset.name].csv_file

        else:
            cog_file = self._extract_participants_files_from_layouts(
                self.brain_preparator.layouts
            )
        preprocessor = DemographicsPreparationPipeline(cog_file)
        demographics_df = preprocessor.preprocess(self.cfg.target.target_column)
        return demographics_df

    def _get_full_demographics_df(
        self, available_subjects: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Get full demographics DataFrame without column filtering.

        Args:
            available_subjects: Optional list of subject IDs to filter by (e.g., subjects with brain data)

        Returns:
            pd.DataFrame: Full demographics DataFrame with all columns
        """
        if not hasattr(self, "brain_preparator"):
            brain_df = self._get_brain_df()

        # Get demographics file path
        if self.source_dataset.name == "hcp":
            cog_file = self.cfg.cluster.paths[self.source_dataset.name].csv_file
        else:
            cog_file = self._extract_participants_files_from_layouts(
                self.brain_preparator.layouts
            )

        # Load full demographics without filtering columns
        preprocessor = DemographicsPreparationPipeline(cog_file)
        demographics_df = preprocessor.get_full_demographics(available_subjects)

        return demographics_df

    def _filter_dfs(
        self, brain_df: pd.DataFrame, demographics_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align and filter brain and demographics DataFrames.
        Args:
            brain_df (pd.DataFrame): DataFrame containing brain features.
            demographics_df (pd.DataFrame): DataFrame containing demographics data.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Aligned and filtered brain and demographics Data
        """
        # -------- SUBJECT ALIGNMENT --------
        brain_df["subject_id"] = brain_df["subject_id"].astype(str)
        demographics_df["Subject"] = demographics_df["Subject"].astype(str)
        common_subjects = set(brain_df["subject_id"]) & set(demographics_df["Subject"])

        brain_filtered = brain_df[brain_df["subject_id"].isin(common_subjects)]
        demographics_filtered = demographics_df[
            demographics_df["Subject"].isin(common_subjects)
        ]
        return brain_filtered, demographics_filtered

    def _create_torch_dataset(
        self, brain_filtered: pd.DataFrame, demographics_filtered: pd.DataFrame
    ) -> Tuple[CustomDataset, PreprocessedData]:
        """
        Create CustomDataset and PreprocessedData objects.

        When the active pipeline is :class:`~diff_benchmark.preprocessing.brain_feature_extraction.MeshPipeline`,
        the mesh objects stored in ``self.brain_preparator.results`` are passed
        to :class:`~diff_benchmark.data.generate_dataset.CustomDataset` so each
        sample additionally returns a mesh tensor dict.

        Args:
            brain_filtered: Filtered brain features DataFrame.
            demographics_filtered: Filtered demographics DataFrame.
        Returns:
            Tuple[CustomDataset, PreprocessedData]: Created dataset and preprocessed data.
        """
        use_cache = self._should_use_cache()
        # -------- DATASET CREATION --------
        X = brain_filtered
        y = np.asarray(demographics_filtered[self.cfg.target.target_column[0]])
        gender = np.asarray(demographics_filtered["Gender"])
        subject_ids = demographics_filtered["Subject"].values

        # Check whether the active pipeline holds mesh results
        mesh_data = None
        if hasattr(self, "brain_preparator") and isinstance(
            self.brain_preparator, MeshPipeline
        ):
            mesh_data = self.brain_preparator.get_mesh_parquet_paths()
            logger.info(
                "Attaching mesh parquet paths for %d subjects", len(mesh_data)
            )

        if use_cache:
            # Load cached features
            cache_path, cache_exists, required_augs, cached_augs = (
                self._get_cache_info()
            )

            logger.info(f"Loading cached features from {cache_path}")

            # Load cache with normalization parameters
            norm_mean = self.cfg.data.normalization.mean
            norm_std = self.cfg.data.normalization.std
            image_size = _parse_image_size(self.cfg.data.get("resize_shape"))
            if self.model_name in {"pointnet", "pointnet_pp", "region_pointnet"}:
                image_size = None

            # Create a reference regular dataset to verify subject alignment
            regular_dataset = CustomDataset(X, y, gender, mesh_data=mesh_data)
            reference_loader = DataLoader(
                regular_dataset,
                batch_size=self.cfg.data.batch_size,
                shuffle=False,
                num_workers=0,
            )

            cached_dataset = CachedFeatureDataset(
                cache_path=cache_path,
                num_augmentations=required_augs,
                image_size=image_size,
                norm_mean=norm_mean,
                norm_std=norm_std,
                source_dataloader=reference_loader,
            )

            cached_dataset.subject_ids = subject_ids.tolist()
            cached_dataset.targets = y
            cached_dataset.gender = gender

            X_dummy = pd.DataFrame({"subject_id": subject_ids})
            preprocessed = PreprocessedData(X_dummy, y, gender, config=self.cfg)

            return cached_dataset, preprocessed
        else:
            # Regular dataset (with optional mesh data)
            torch_dataset = CustomDataset(X, y, gender, mesh_data=mesh_data)
            preprocessed = PreprocessedData(X, y, gender, config=self.cfg)
            return torch_dataset, preprocessed

    def pipeline(self) -> Tuple[CustomDataset, PreprocessedData]:
        """
        Orchestrates the data preparation pipeline.

        Automatically handles cached features for cacheable models (vit, dinov2, curia)
        with freeze_backbone=True. Computes cache if missing or incomplete.

        Returns:
            Tuple[Union[CustomDataset, CachedFeatureDataset], PreprocessedData]:
                The prepared dataset and preprocessed data.
        """
        use_cache = self._should_use_cache()
        print("Preparing demographics data...")
        brain_df = self._get_brain_df()
        demographics_df = self._get_demographics_df()

        print("Aligning and filtering data...")
        brain_filtered, demographics_filtered = self._filter_dfs(
            brain_df, demographics_df
        )

        if use_cache:
            # Check cache status
            cache_path, cache_exists, required_augs, cached_augs = (
                self._get_cache_info()
            )

            # Compute or update cache if needed
            if not cache_exists or cached_augs < required_augs:
                print("Creating regular dataset for cache computation...")
                mesh_data = None
                if hasattr(self, "brain_preparator") and isinstance(
                    self.brain_preparator, MeshPipeline
                ):
                    mesh_data = self.brain_preparator.get_mesh_parquet_paths()
                regular_dataset = CustomDataset(
                    brain_filtered,
                    np.asarray(demographics_filtered[self.cfg.target.target_column[0]]),
                    np.asarray(demographics_filtered["Gender"]),
                    mesh_data=mesh_data,
                )

                self._compute_or_update_cache(
                    cache_path,
                    cache_exists,
                    required_augs,
                    cached_augs,
                    regular_dataset,
                )

            # Now create cached dataset
            print("Creating cached feature dataset...")
            torch_dataset, preprocessed = self._create_torch_dataset(
                brain_filtered, demographics_filtered
            )
        else:
            # Regular path (no caching)
            print("Creating dataset...")
            torch_dataset, preprocessed = self._create_torch_dataset(
                brain_filtered, demographics_filtered
            )

        return torch_dataset, preprocessed
