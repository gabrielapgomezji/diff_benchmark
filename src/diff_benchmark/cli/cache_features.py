"""
CLI command to pre-compute and cache features for datasets.

This command computes features from frozen backbone models and saves them to disk,
dramatically speeding up subsequent training runs.
"""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from diff_benchmark.data.cached_features import CachedFeatureDataset, get_cache_path
from diff_benchmark.data.prepare_data import DatasetPreparation
from diff_benchmark.models.model_configurations import create_model
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_features_for_dataset(
    model_name: str,
    model_config: dict,
    dataset_config: dict,
    cache_dir: str,
    num_augmentations: int = 10,
    image_size: tuple = None,
    tissue_type: str = None,
    metric_to_compute: str = None,
    norm_mean: float = 0.5,
    norm_std: float = 0.5,
    force_recompute: bool = False,
):
    """
    Compute and cache features for a single dataset.

    Args:
        model_name: Name of the model (e.g., "vit", "dinov2")
        model_config: Model configuration dictionary
        dataset_config: Dataset configuration dictionary
        cache_dir: Directory to store cache files
        num_augmentations: Number of augmented versions per sample
        image_size: Optional (H, W) tuple for resizing slices. Recommended for HCP: (256, 256)
        tissue_type: Optional tissue type (e.g., "gray", "white")
        metric_to_compute: Optional microstructure metric (e.g., "md", "fa", "sh")
        norm_mean: Mean for normalization (from config)
        norm_std: Std for normalization (from config)
        force_recompute: If True, recompute even if cache exists
    """
    logger.info(f"Computing features for {model_name} on {dataset_config['name']}")

    # Convert image_size to tuple if it's a list from config
    # Handle Hydra's string "None" conversion issue
    if image_size is not None and isinstance(image_size, (list, tuple)):
        # Check if it's a list containing None or string "None"
        if len(image_size) == 1 and (image_size[0] is None or image_size[0] == "None"):
            image_size = None
        else:
            image_size = tuple(image_size)

    if image_size is not None:
        logger.info(f"Will resize slices to {image_size}")
    else:
        logger.info("Using original image sizes (no resizing)")

    logger.info(f"Normalization: mean={norm_mean}, std={norm_std}")
    # Create dataset
    dataset_obj = DatasetConfig(
        **dataset_config,
        base_dir=Path(
            model_config["cluster"]["paths"][dataset_config["name"]]["base_dir"]
        ),
        results_dir=Path(
            model_config["cluster"]["paths"][dataset_config["name"]]["results_dir"]
        ),
    )

    dataset_preparator = DatasetPreparation(
        cfg=OmegaConf.create(model_config),
        source_dataset=dataset_obj,
    )

    torch_dataset, preprocessed = dataset_preparator.pipeline()

    logger.info(f"Dataset loaded: {len(torch_dataset)} samples")

    # Create backbone model with frozen weights
    model_kwargs = model_config["model"]["backbone"].copy()
    model_kwargs["freeze_backbone"] = True  # Ensure backbone is frozen

    logger.info(f"Creating {model_name} backbone (frozen)...")
    model = create_model(
        model_name=model_name,
        model_kwargs=model_kwargs,
        pred_head={"prediction_task": "binary_classification"},  # Dummy head
    )

    # Extract just the backbone if it's a TaskModel
    if hasattr(model, "backbone"):
        backbone = model.backbone

    # Verify backbone is frozen
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in backbone.parameters())
    logger.info(f"Backbone parameters: {trainable_params}/{total_params} trainable")

    if trainable_params > 0:
        logger.warning(
            f"WARNING: Backbone has {trainable_params} trainable parameters!"
        )

    # Create dataloader
    dataloader = DataLoader(
        torch_dataset,
        batch_size=model_config["data"]["batch_size"],
        shuffle=False,
        num_workers=0,  # No multiprocessing for feature computation
    )

    # Determined cache model name (handle variants like medicalnet depth)
    cache_model_name = model_name
    if model_name == "medicalnet" and "depth" in model_config["model"]["backbone"]:
        depth = model_config["model"]["backbone"]["depth"]
        cache_model_name = f"{model_name}_depth{depth}"

    # Get cache path
    cache_path = get_cache_path(
        model_name=cache_model_name,
        dataset_name=dataset_config["name"],
        cache_dir=cache_dir,
        image_size=image_size,
        tissue_type=tissue_type,
        metric_to_compute=metric_to_compute,
    )

    logger.info(f"Cache will be saved to: {cache_path}")

    # Check if cache already exists
    if cache_path.exists() and not force_recompute:
        logger.info(f"Cache already exists at {cache_path}")
        logger.info("Use force_recompute=True to recompute")

        # Load metadata
        import json

        metadata_path = cache_path.with_suffix(".meta.json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            # Fallback: read from parquet
            import pandas as pd

            df = pd.read_parquet(cache_path)
            metadata = {
                "num_samples": len(df["subject_id"].unique()),
                "num_augmentations": len(df["augmentation_idx"].unique()),
                "embedding_dim": len(
                    [c for c in df.columns if c.startswith("feature_")]
                ),
            }

        logger.info(f"Cache contains:")
        logger.info(f"  - Samples: {metadata.get('num_samples', 'unknown')}")
        logger.info(
            f"  - Augmentations: {metadata.get('num_augmentations', 'unknown')}"
        )
        logger.info(f"  - Embedding dim: {metadata.get('embedding_dim', 'unknown')}")

        # Show image_size if available
        if "image_size" in metadata:
            img_size = metadata["image_size"]
            if img_size is None:
                logger.info(f"  - Image size: original (no resizing)")
            else:
                logger.info(f"  - Image size: {img_size}")

        # Show file size
        cache_size_mb = cache_path.stat().st_size / (1024**2)
        logger.info(f"  - Size: {cache_size_mb:.1f} MB")
        return

    # Compute and cache features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cached_dataset = CachedFeatureDataset(
        cache_path=cache_path,
        backbone=backbone,
        source_dataloader=dataloader,
        device=device,
        num_augmentations=num_augmentations,
        image_size=image_size,
        norm_mean=norm_mean,
        norm_std=norm_std,
        force_recompute=force_recompute,
    )

    # Show file size
    cache_size_mb = cache_path.stat().st_size / (1024**2)

    logger.info(f"\n✓ Successfully cached features for {len(cached_dataset)} samples")
    logger.info(f"✓ Cache saved to: {cache_path}")
    logger.info(f"✓ Size: {cache_size_mb:.1f} MB")
    logger.info(f"✓ Future training runs will be ~10-20x faster!\n")


@hydra.main(
    version_base="1.3",
    config_path="pkg://diff_benchmark.configs",
    config_name="main",
)
def main(cfg: DictConfig) -> None:
    """
    Main function for feature caching CLI.

    Normalization parameters (mean, std) are read from data.normalization in config.

    Usage with Hydra multirun for cross products:
        # Cache features for specific model and dataset
        python -m diff_benchmark.cli.cache_features model.name=vit datasets.name=abide

        # Cache with custom normalization (override config defaults)
        python -m diff_benchmark.cli.cache_features model.name=vit data.normalization.mean=0.5 data.normalization.std=0.5

        # Cache with resizing (recommended for HCP with 145×174 slices)
        python -m diff_benchmark.cli.cache_features model.name=vit datasets.name=hcp

        # Multirun: cache for multiple models and datasets (cross product)
        python -m diff_benchmark.cli.cache_features -m model.name=vit,dinov2 datasets.name=abide,aomic

        # Force recompute
        python -m diff_benchmark.cli.cache_features model.name=vit force=true

        # Custom cache directory and augmentations
        python -m diff_benchmark.cli.cache_features cache_dir=my_cache num_augmentations=20
    """
    # Get parameters from single configuration (Hydra handles cross products via multirun)
    model_name = cfg.model.name
    cache_dir = Path(cfg.cluster.paths.cache_dir) / "dl_features"
    num_augmentations = cfg.data.num_augmentations
    image_size = cfg.data.resize_shape
    tissue_type = cfg.dataset.get(
        "tissue_type", None
    )  # Get tissue type from dataset config
    metric_to_compute = cfg.dataset.get(
        "metric_to_compute", None
    )  # Get microstructure metric
    force = cfg.get("force", False)

    # Get normalization parameters from config
    norm_mean = cfg.data.normalization.mean
    norm_std = cfg.data.normalization.std

    # Only cache for models that benefit from caching (frozen backbones)
    if model_name not in ["vit", "dinov2", "curia", "medicalnet"]:
        logger.info(f"Skipping {model_name} (not a frozen backbone model)")
        return

    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {cfg.dataset.name}")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Augmentations per sample: {num_augmentations}")
    logger.info(f"Tissue type: {tissue_type if tissue_type else 'N/A'}")
    logger.info(
        f"Microstructure metric: {metric_to_compute if metric_to_compute else 'N/A'}"
    )
    logger.info(f"Normalization: mean={norm_mean}, std={norm_std}")
    logger.info(f"Force recompute: {force}")
    logger.info("")

    # Convert config to dict for compatibility
    config = OmegaConf.to_container(cfg, resolve=True)

    # Workaround for learning_rate being a dict (e.g. from search space config)
    # which causes TypeError in TorchTrainer/Adam initialization
    if "backend" in config and isinstance(
        config["backend"].get("learning_rate"), (dict, list)
    ):
        logger.warning(
            f"Found complex type for learning_rate: {config['backend']['learning_rate']}. Using default 1e-4 for feature caching."
        )
        config["backend"]["learning_rate"] = 1e-4

    # Extract dataset config
    dataset_config = config.get("dataset", {})
    try:
        compute_features_for_dataset(
            model_name=model_name,
            model_config=config,
            dataset_config=dataset_config,
            cache_dir=cache_dir,
            num_augmentations=num_augmentations,
            image_size=image_size,
            tissue_type=tissue_type,
            metric_to_compute=metric_to_compute,
            norm_mean=norm_mean,
            norm_std=norm_std,
            force_recompute=force,
        )

        # logger.info("\n" + "=" * 80)
        logger.info("Feature caching complete!")
        # logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error computing features for {model_name}: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
