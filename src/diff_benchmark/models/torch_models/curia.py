import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel, BitImageProcessor
from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


class CuriaBackbone(nn.Module):
    """
    CURIA medical foundation model adapted for 3D volumes
    via slice-wise processing.
    
    Note: Expects input data normalized with mean=0.5, std=0.5 (from cache).
    This will be unnormalized back to [0, 1] before passing to HuggingFace processor.
    """

    data_type = "images"

    def __init__(
        self,
        model_name: str = "raidium/curia",
        freeze_backbone: bool = False,
        slice_axis: int = 0,  # axial by default
        pooling: str = "mean",  # mean | max | cls
    ):
        super().__init__()

        self.slice_axis = slice_axis
        self.pooling = pooling

        # Check for local pretrained model first, same as vit.py and dinov2.py
        model_dir = Path(__file__).parent.parent.parent.parent.parent / "pretrain" / model_name
        if model_dir.exists():
            source = str(model_dir)
            local_only = True
            logger.info(f"Loading CURIA from local directory: {source}")
        else:
            logger.info(f"Pretrained model directory {model_dir} does not exist. Using model name {model_name} from HuggingFace Hub if possible.")
            source = model_name
            local_only = False
        
        # Handle snapshot structure if config.json not in root
        source_path = Path(source)
        if source_path.exists() and not (source_path / "config.json").exists():
             # Look for config.json in subdirectories (either snapshots/hash or just hash)
             logger.info(f"No config.json in {source}, searching subdirectories...")
             candidates = list(source_path.glob("**/config.json"))
             # Filter out hidden directories like .git
             candidates = [p for p in candidates if ".git" not in str(p)]
             
             if candidates:
                # Pick the most recently modified config
                target_config = max(candidates, key=lambda p: p.stat().st_mtime)
                snapshot_dir = target_config.parent
                logger.info(f"Redirecting source to found model directory: {snapshot_dir}")
                source = str(snapshot_dir)

        # Move backbone loading before processor to access config
        self.backbone = AutoModel.from_pretrained(source, local_files_only=local_only)
        self.embedding_dim = self.backbone.config.hidden_size
        
        num_channels = getattr(self.backbone.config, "num_channels", 3)
        self.num_channels = num_channels
        logger.info(f"Model expects {num_channels} channel(s)")

        try:
            # Use trust_remote_code=True to load local custom processor (CuriaImageProcessor)
            self.processor = AutoImageProcessor.from_pretrained(source, local_files_only=local_only, trust_remote_code=True)
        except (OSError, ValueError) as e:
            # If preprocessor_config.json is missing or invalid, fallback
            logger.info(f"Could not load image processor from {source}: {e}")
            
            # Get parameters from model config
            image_size = getattr(self.backbone.config, "image_size", 512)
            logger.info(f"Configuring fallback BitImageProcessor using model config: image_size={image_size}")
            
            # Configure normalization based on channels
            if num_channels == 1:
                image_mean = [0.5]
                image_std = [0.5]
            else:
                image_mean = [0.485, 0.456, 0.406]
                image_std = [0.229, 0.224, 0.225]

            # Fallback for curia/dinov2 if preprocessor_config.json is missing
            self.processor = BitImageProcessor(
                do_resize=True,
                size={"shortest_edge": image_size},
                do_center_crop=True,
                crop_size={"height": image_size, "width": image_size},
                resample=3,  # BICUBIC
                do_rescale=True,
                rescale_factor=0.00392156862745098,
                do_normalize=True,
                image_mean=image_mean,
                image_std=image_std,
            )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        logger.info(
            f"Loaded CURIA backbone {model_name} "
            f"(embedding_dim={self.embedding_dim})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, D, H, W) or (B, 1, D, H, W) - normalized data from cache (mean=0.5, std=0.5)
        Returns:
            Tensor of shape (B, embedding_dim)
        """
        # Unnormalize: convert from normalized [-1, 1] back to [0, 1]
        # x_original = x_normalized * std + mean = x * 0.5 + 0.5
        x = x * 0.5 + 0.5
        # Check if input is already features (from cache) or raw images
        if x.ndim == 2:
            # Already processed features from cache: (B, embedding_dim)
            # No processing needed, return as-is
            return x
        
        # Handle both (B, D, H, W) and (B, 1, D, H, W) formats
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(1)  # (B, D, H, W)
        
        B, D, H, W = x.shape

        # Slice selection
        if self.slice_axis == 0:
            slices = x
        elif self.slice_axis == 1:
            slices = x.permute(0, 2, 1, 3)
        else:
            slices = x.permute(0, 3, 1, 2)

        num_slices = slices.shape[1]

        # Flatten slices into batch
        slices = slices.reshape(B * num_slices, H, W)

        # NOTE: We do NOT add a channel dimension here because the custom CuriaImageProcessor
        # interprets 3D inputs (C, H, W) as 3D volumes (H, W, D) and tries to process them slice-by-slice.
        # By passing 2D tensors (H, W), the processor treats them as single slices as intended.

        # Processor expects a list of images (tensors or arrays)
        inputs = self.processor(
            images=list(slices), 
            return_tensors="pt",
            do_rescale=False,
        )

        # Fix: Ensure output is 4D (B, C, H, W)
        if "pixel_values" in inputs:
            p_vals = inputs["pixel_values"]
            if p_vals.ndim == 5 and p_vals.shape[1] == 1:
                inputs["pixel_values"] = p_vals.squeeze(1)

        inputs = {k: v.to(x.device) for k, v in inputs.items()}

        # outputs = self.backbone(**inputs)
        # tokens = outputs.last_hidden_state  # (B*S, N, C)
        with torch.no_grad():
            outputs = self.backbone(**inputs)
        tokens = outputs.last_hidden_state  # (B*S, N, C)

        # Slice-level pooling
        if self.pooling == "cls":
            slice_embeds = tokens[:, 0]
        else:
            slice_embeds = tokens.mean(dim=1)

        # Volume-level aggregation
        slice_embeds = slice_embeds.view(B, num_slices, -1)

        if self.pooling == "max":
            volume_embed = slice_embeds.max(dim=1).values
        else:
            volume_embed = slice_embeds.mean(dim=1)

        return volume_embed
