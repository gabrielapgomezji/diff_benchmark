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

    def __init__(self, pretrained: bool =True, trainable_blocks: int =0, **kwargs):
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
        """
        Defines the forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where
                B is the batch size, C is the number of channels, 
                H is the height, and W is the width.
        Returns:
            torch.Tensor: Output tensor of shape (B, 512), where 512 
                represents the flattened feature dimension extracted 
                by the feature extractor.
        """
        
        feats = self.feature_extractor(x)  # (B, 512, 1, 1)
        return feats.view(feats.size(0), -1)  # (B, 512)


class ResNet3SliceClassifier(nn.Module):
    """ResNet3SliceClassifier is a neural network model that classifies input slices using a ResNet18 backbone.
    Attributes:
        backbone (nn.Module): The ResNet18 backbone used for feature extraction.
        num_subvols (int): The number of subvolumes derived from the input slices.
        fc (nn.Linear): Fully connected layer for classification.
    Args:
        input_slices (int): The total number of input slices.
        num_classes (int, optional): The number of output classes. Default is 2.
        freeze_backbone (bool, optional): If True, the backbone parameters are frozen during training. Default is True.
        **kwargs: Additional keyword arguments passed to the ResNet18 backbone.
    Methods:
        forward(x):
            Forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor of shape (batch, Slice, Height, Width) where batch is batch size,
                                  Slice is the number of slices, Height is height, and Width is width.
            Returns:
                torch.Tensor: Output tensor of shape (batch, num_classes) representing class scores.
    """

    def __init__(
        self, input_slices: int, num_classes: int = 2, freeze_backbone: bool = True, dropout: float = 0.5, **kwargs
    ):
        super().__init__()
        self.backbone = ResNet18Backbone(**kwargs)
        self.num_subvols = input_slices // 3
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.num_subvols * self.backbone.out_dim, num_classes)
        # # Aggregate subvolume embeddings into a single embedding (B, 512)
        # # learnable per-subvolume scalar weights (will be normalized via softmax in forward)
        # self.aggregate_weights = nn.Parameter(
        #     torch.ones(self.num_subvols, dtype=torch.float32)
        # )
        # self.fc = nn.Linear(self.backbone.out_dim, num_classes)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.attention = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where:
                - B is the batch size,
                - C is the number of channels,
                - H and W are the height and width of the input.
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes), where:
                - B is the batch size,
                - num_classes is the number of output classes.
        Process:
            1. The input tensor is divided into subvolumes along the channel dimension.
            2. Subvolumes are reshaped and processed in parallel through the backbone network.
            3. Features from all subvolumes are concatenated and passed through a dropout layer.
            4. The final output is computed using a fully connected layer.
        """
        
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
        feats = feats.reshape(B, -1)

        feats = self.dropout(feats)
        out = self.fc(feats)  # (B, num_classes)
        return out


def collate_with_augmentation(batch: list, transform: callable = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate function that applies 2D augmentations to each slice of 3D volumes in the batch.
    Args:
        batch (list): A list of tuples, where each tuple contains (x, y, g) for a single sample.
                      x is a 3D tensor (D, H, W), y is the label tensor, and g is additional info tensor.
        transform (callable, optional): A function that applies 2D augmentations to a single slice.
                                        If None, no augmentation is applied. Defaults to None.
    Returns:
        tuple: A tuple containing:
            - xs_aug (torch.Tensor): A tensor of shape (batch_size, C, D, H, W) with augmented slices.
            - ys (torch.Tensor): A tensor of shape (batch_size,) containing labels.
            - gs (torch.Tensor): A tensor of shape (batch_size,) containing additional info.
    """
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


class CNNTorchTrainModel(TorchPipeline):
    """CNN Torch Train Model class inheriting from TorchPipeline.
    Attributes:
        name (str): The name of the model.
        data_type (str): The type of data the model processes.
    Methods:
        _build_model(input_slices, num_classes, freeze_backbone, dropout, **kwargs):
            Builds and returns the ResNet3SliceClassifier model.
    """

    data_type = "images"

    def _build_model(
        self, input_slices: int, num_classes: int, freeze_backbone: bool, dropout: float, **kwargs
    ):
        pretrained = kwargs.get("pretrained", False)
        trainable_blocks = kwargs.get("trainable_blocks", None)
        model = ResNet3SliceClassifier(
            input_slices=input_slices,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
            pretrained=pretrained,
            trainable_blocks=trainable_blocks,
        )
        model.collate_with_augmentation = collate_with_augmentation
        model.std = 0.5
        model.mean = 0.5
        return model
