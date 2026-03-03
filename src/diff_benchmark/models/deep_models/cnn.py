from typing import Callable

import torch
from torch import nn
from torchvision import models


class ResNet18Backbone(nn.Module):
    """ResNet-18 feature extractor with the final FC layer removed."""

    def __init__(self, pretrained: bool = True, trainable_blocks: int = 0, **kwargs):
        super().__init__()
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Remove final FC
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # up to avgpool
        self.out_dim = 512

        # freeze all by default
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # unfreeze last N blocks
        if trainable_blocks > 0:
            blocks = [resnet.layer4, resnet.layer3, resnet.layer2, resnet.layer1]
            for block in blocks[:trainable_blocks]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)  # (B, 512, 1, 1)
        return feats.view(feats.size(0), -1)  # (B, 512)


class ResNet3SliceMultihead(nn.Module):
    """ResNet-based model that processes 3D volumes as grouped 2D slice triplets."""

    data_type = "images"

    def __init__(
        self,
        input_slices: int,
        freeze_backbone: bool = True,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.backbone = ResNet18Backbone(**kwargs)
        self.num_subvols = input_slices // 3
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.aggregate_weights = nn.Parameter(
            torch.ones(self.num_subvols, dtype=torch.float32)
        )
        self.out_dim = self.backbone.out_dim
        self.mean = 0.5
        self.std = 0.5
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch, slice, height, width = x.shape
        # assert S % 3 == 0, "Slice dimension must be divisible by 3"
        x = x.squeeze(1)

        subvols = x.unfold(dimension=1, size=3, step=3)
        subvols = subvols.permute(0, 1, 4, 2, 3)  # (B, num_subvols, 3, H, W)

        B, N, C, H, W = subvols.shape  # N = num_subvols

        # Merge (B, N) into a big batch: (B*N, 3, H, W)
        subvols = subvols.reshape(B * N, C, H, W)

        # Run backbone ONCE on all subvolumes in parallel
        feats = self.backbone(subvols)  # (B*N, 512)

        # Reshape back to (B, N, 512)
        feats = feats.reshape(B, N, -1)

        # Concatenate subvolume features: (B, N*512)
        # feats = feats.reshape(B, -1)
        w = torch.softmax(self.aggregate_weights, dim=0)  # (N,)
        w = w.view(1, N, 1)  # (1, N, 1)

        feats = (feats * w).sum(dim=1)  # (B, 512)

        # feats scalar product with weights. 1 embedding per features (B, 512).
        feats = self.dropout(feats)
        return feats

    def collate_with_augmentation(
        self, batch: list, transform: Callable = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate a batch and optionally apply 2D augmentation slice-by-slice.

        Args:
            batch (list): List of ``(x, y, g)`` tuples; ``x`` has shape ``(D, H, W)``.
            transform (callable | None): Per-slice 2D transform. No-op if ``None``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(xs_aug, ys, gs)`` with
                ``xs_aug`` of shape ``(B, D, H, W)``.
        """
        xs, ys, gs = zip(*batch)
        xs_aug = []
        for x in xs:  # x shape: (D,H,W)
            slices = []
            for i in range(x.shape[0]):
                slice_2d = x[i, :, :].unsqueeze(0)  # (1,H,W)
                if transform:
                    slice_2d = transform(slice_2d)
                slices.append(slice_2d)
            x_aug = torch.stack(slices, dim=0)  # (D,1,H,W)
            x_aug = x_aug.permute(1, 0, 2, 3)  # (C=1,D,H,W)
            xs_aug.append(x_aug)

        xs_aug = torch.stack(xs_aug, dim=0)
        ys = torch.stack(ys)
        gs = torch.stack(gs)
        return xs_aug.squeeze(1), ys, gs
