import torch
from torch import nn
from torchvision import models

from typing import Any


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
                represents the flattened feature vector extracted 
                by the feature extractor.
        """
        
        feats = self.feature_extractor(x)  # (B, 512, 1, 1)
        return feats.view(feats.size(0), -1)  # (B, 512)


class ResNet3SliceMultihead(nn.Module):
    """A PyTorch module for processing 3D medical image data using a ResNet-based backbone.
    This model processes input slices in groups of three (subvolumes) and aggregates their features
    using learnable scalar weights. It supports both classification and regression tasks.
    Args:
        input_slices (int): The number of input slices. Must be divisible by 3.
        num_classes (int, optional): The number of output classes for classification tasks. Default is 2.
        freeze_backbone (bool, optional): Whether to freeze the backbone's parameters during training. Default is True.
        dropout (float, optional): Dropout probability for regularization. If 0, dropout is disabled. Default is 0.5.
        **kwargs: Additional keyword arguments, including:
            - prediction_task (str): The type of prediction task, either "classification" or "regression".
    Attributes:
        backbone (nn.Module): The ResNet18-based backbone for feature extraction.
        num_subvols (int): The number of subvolumes derived from the input slices.
        dropout (nn.Module): Dropout layer for regularization.
        aggregate_weights (nn.Parameter): Learnable scalar weights for aggregating subvolume features.
        fc (nn.Module): Fully connected layer(s) for the final prediction, configured based on the task.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the model. Processes the input tensor and returns the output tensor.
            Args:
                x (torch.Tensor): Input tensor of shape (batch, Slice, Height, Width), where Slice is the slice dimension.
            Returns:
                torch.Tensor: Output tensor of shape (batch, num_classes) for classification tasks or
                (batch, 1) for regression tasks.
    """
    
    data_type = "images"
    
    def __init__(
        self, input_slices: int, num_classes: int = 2, freeze_backbone: bool = True, dropout: float = 0.5, **kwargs
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
        self.out_dim = self.backbone.out_dim
        # self.fc = PredictionHead(
        #     embedding_dim=self.backbone.out_dim,
        #     prediction_task=self.prediction_task,
        #     num_classes=num_classes, # for regression is specified to 1
        #     hidden_dims=[256],
        #     dropout=dropout,
        # )
        self.mean = 0.5
        self.std = 0.5
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        This method processes the input tensor `x` through the following steps:
        1. Removes the singleton dimension from the input tensor.
        2. Splits the input tensor into subvolumes along the slice dimension.
        3. Permutes and reshapes the subvolumes for parallel processing.
        4. Passes the subvolumes through the backbone network to extract features.
        5. Aggregates the features using learned weights and computes a single
           embedding per batch.
        6. Applies dropout to the aggregated features.
        7. Passes the processed features through a fully connected layer to
           produce the final output.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, S, H, W), where
            B is the batch size,
            S is the number of slices (must be divisible by 3),
            H is the height, and
            W is the width.
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes), where
            num_classes is the number of output classes.
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
        # out = self.fc(feats)  # (B, num_classes)
        # return out
        return feats
        
        ###
        # feats = []
        # for i in range(num_subvols):
        #     sv = subvols[:, i]  # (Batch, 3, Height, Width)
        #     f = self.backbone(sv)  # (Batch, 512)
        #     feats.append(f)

        # feats = torch.cat(feats, dim=1)  # (Batch, num_subvols*512)
        # feats = self.dropout(feats)
        # # Normalization layer
        # out = self.fc(feats)  # (Batch, num_classes)
        # return out

    def collate_with_augmentation(self, batch: list, transform: callable = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Custom collate function that applies 2D augmentations to each slice of 3D volumes in the batch.
        Args:
            batch (list): A list of tuples, where each tuple contains (x, y, g) for a single sample.
                        x is a 3D tensor (D, H, W), y is the label tensor, and g is the group tensor.
            transform (callable, optional): A function that applies 2D augmentations to a single slice.
                                            If None, no augmentation is applied. Default is None.
        Returns:
            tuple: A tuple containing:
                - xs_aug (torch.Tensor): A tensor of shape (batch_size, C, D, H, W) with augmented slices.
                - ys (torch.Tensor): A tensor of shape (batch_size,) containing the labels.
                - gs (torch.Tensor): A tensor of shape (batch_size,) containing the group identifiers.
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

def collate_with_augmentation(batch: list, transform: callable = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Custom collate function that applies 2D augmentations to each slice of 3D volumes in the batch.
        Args:
            batch (list): A list of tuples, where each tuple contains (x, y, g) for a single sample.
                        x is a 3D tensor (D, H, W), y is the label tensor, and g is the group tensor.
            transform (callable, optional): A function that applies 2D augmentations to a single slice.
                                            If None, no augmentation is applied. Default is None.
        Returns:
            tuple: A tuple containing:
                - xs_aug (torch.Tensor): A tensor of shape (batch_size, C, D, H, W) with augmented slices.
                - ys (torch.Tensor): A tensor of shape (batch_size,) containing the labels.
                - gs (torch.Tensor): A tensor of shape (batch_size,) containing the group identifiers.
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


class ResNet3SliceBackbone(nn.Module):
    """
    Backend-agnostic CNN architecture.
    """

    data_type = "images"

    def __init__(
        self,
        input_slices: int,
        num_classes: int,
        freeze_backbone: bool = True,
        dropout: float = 0.5,
        pretrained: bool = False,
        trainable_blocks: Any =None,
        prediction_task: str |None = None,
        **kwargs: Any,
    ):
        super().__init__()

        self.net = ResNet3SliceMultihead(
            input_slices=input_slices,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
            pretrained=pretrained,
            trainable_blocks=trainable_blocks,
            prediction_task=prediction_task,
        )

        # dataset-specific metadata (OK here)
        self.collate_with_augmentation = collate_with_augmentation
        self.mean = 0.5
        self.std = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
