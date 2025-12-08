import torch
from torch import nn
from torchvision import models

from diff_benchmark.models.base import TorchPipeline


class ResNet18Backbone(nn.Module):
    """ResNet18Backbone is a PyTorch neural network module that utilizes a pre-trained ResNet-18 model
    as a feature extractor. It removes the final fully connected layer to output feature vectors
    of a specified dimension.
    Attributes:
        feature_extractor (nn.Sequential): A sequential container that holds the layers of the
        ResNet-18 model up to the average pooling layer.
        out_dim (int): The output dimension of the feature vectors, which is 512 for ResNet-18.
    Args:
        pretrained (bool): If True, initializes the model with pre-trained weights. Defaults to True.
        **kwargs: Additional keyword arguments to be passed to the parent class.
    Methods:
        forward(x):
            Takes an input tensor and returns the extracted feature vector.
            Args:
                x (torch.Tensor): Input tensor of shape (B, 3, H, W), where B is the batch size,
                3 is the number of channels (RGB), H is the height, and W is the width of the image.
            Returns:
                torch.Tensor: A tensor of shape (B, 512) containing the extracted features.
    """

    def __init__(self, pretrained=True, trainable_blocks=0, **kwargs):
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

    def forward(self, x):
        """
        x: (B, 3, H, W)
        returns: (B, 512)
        """
        feats = self.feature_extractor(x)  # (B, 512, 1, 1)
        return feats.view(feats.size(0), -1)  # (B, 512)


class ResNet3SliceMultihead(nn.Module):
    def __init__(
        self, input_slices, num_classes=2, freeze_backbone=True, dropout=0.5, **kwargs
    ):
        super().__init__()
        self.backbone = ResNet18Backbone(**kwargs)
        self.num_subvols = input_slices // 3
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        # self.fc = nn.Linear(self.num_subvols * self.backbone.out_dim, num_classes)
        # Aggregate subvolume embeddings into a single embedding (B, 512)
        # learnable per-subvolume scalar weights (will be normalized via softmax in forward)
        self.aggregate_weights = nn.Parameter(
            torch.ones(self.num_subvols, dtype=torch.float32)
        )
        self.prediction_task = kwargs.get("prediction_task", None)
        if self.prediction_task == "classification":
            self.fc = nn.Linear(self.backbone.out_dim, num_classes)
        # elif self.prediction_task is None:
        #     raise ValueError("prediction_task must be specified as 'classification' or 'regression'")
        else:
            # self.fc = nn.Linear(self.backbone.out_dim, 1)
            self.fc = nn.Sequential(
                nn.Linear(self.backbone.out_dim, 256),
                nn.ReLU(),
                self.dropout,
                nn.Linear(256, 1),
            )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        x: (batch, Slice, Height, Width) where Slice is slice dimension
        """
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
        # print(w)
        feats = (feats * w).sum(dim=1)  # (B, 512)
        # print(feats)
        # feats scalar product with weights. 1 embedding per features (B, 512).
        feats = self.dropout(feats)
        out = self.fc(feats)  # (B, num_classes)
        return out

def collate_with_augmentation(batch, transform=None):
    """Custom collate function that applies 2D augmentations to each slice of 3D volumes in the batch."""
    xs, ys, gs = zip(*batch)  # separate batch components
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

class CNNRegTorchTrainModel(TorchPipeline):
    """CNN Torch Train Model class inheriting from TorchPipeline."""

    data_type = "images"

    def _build_model(
        self, input_slices, num_classes, freeze_backbone, dropout, **kwargs
    ):
        pretrained = kwargs.get("pretrained", False)
        trainable_blocks = kwargs.get("trainable_blocks", None)
        prediction_task = kwargs.get("prediction_task", None)
        model = ResNet3SliceMultihead(
            input_slices=input_slices,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
            pretrained=pretrained,
            trainable_blocks=trainable_blocks,
            prediction_task=prediction_task,
        )
        model.collate_with_augmentation = collate_with_augmentation
        model.std = 0.5
        model.mean = 0.5
        
        return model
