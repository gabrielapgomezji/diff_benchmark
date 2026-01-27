import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


class GoogleViTBackbone(nn.Module):
    """
    Google ViT backbone adapted for 3D volumes via slice-wise processing.
    """

    data_type = "images"

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        freeze_backbone: bool = False,
        slice_axis: int = 2,
        pooling: str = "mean",  # mean | max | cls
    ):
        super().__init__()

        self.slice_axis = slice_axis
        self.pooling = pooling
        model_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "pretrain" / model_name
        if model_dir.exists():
            source = str(model_dir)
            local_only = True
        else:
            print(f"Pretrained model directory {model_dir} does not exist. Using model name {model_name} from HuggingFace Hub if possible.")
            source = model_name
            local_only = False
        self.processor = AutoImageProcessor.from_pretrained(source, local_files_only=local_only)
        self.backbone = AutoModel.from_pretrained(source, local_files_only=local_only)

        self.embedding_dim = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        logger.info(
            f"Loaded Google ViT backbone {model_name} "
            f"(embedding_dim={self.embedding_dim})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 1, D, H, W)
        Returns:
            Tensor of shape (B, embedding_dim)
        """
        B, D, H, W = x.shape
        # assert C == 1, "Expected single-channel volumes"

        # # Remove channel dim
        # x = x.squeeze(1)  # (B, D, H, W)

        # Slice selection
        if self.slice_axis == 0:
            slices = x
        elif self.slice_axis == 1:
            slices = x.permute(0, 2, 1, 3)
        else:
            slices = x.permute(0, 3, 1, 2)
        
        num_slices = slices.shape[1]

        # (B*S, H, W)
        slices = slices.reshape(B * num_slices, H, W)

        # Grayscale → RGB
        slices = slices.unsqueeze(1).repeat(1, 3, 1, 1)

        inputs = self.processor(
            images=slices,
            return_tensors="pt",
            do_rescale=False,
        )
        inputs = {k: v.to(x.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.backbone(**inputs)
        tokens = outputs.last_hidden_state  # (B*S, N, C)

        if self.pooling == "cls":
            slice_embeds = tokens[:, 0]
        else:
            slice_embeds = tokens.mean(dim=1)

        slice_embeds = slice_embeds.view(B, num_slices, -1)

        if self.pooling == "max":
            volume_embed = slice_embeds.max(dim=1).values
        else:
            volume_embed = slice_embeds.mean(dim=1)

        return volume_embed
