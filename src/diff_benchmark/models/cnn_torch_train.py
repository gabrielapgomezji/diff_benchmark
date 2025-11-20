import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (  # roc_auc_score,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from tqdm import tqdm

from diff_benchmark.models.base import TorchPipeline
from diff_benchmark.utils.logger import TrainLogger  # MetricsManager


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
        self, input_slices, num_classes=2, freeze_backbone=True, dropout=0.5, **kwargs
    ):
        super().__init__()
        self.backbone = ResNet18Backbone(**kwargs)
        self.num_subvols = input_slices // 3
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.num_subvols * self.backbone.out_dim, num_classes)

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
        feats = feats.reshape(B, -1)

        feats = self.dropout(feats)
        out = self.fc(feats)  # (B, num_classes)
        return out

class CNNTorchTrainModel(TorchPipeline):
    
    data_type = "images"

    def _build_model(
        self,
        input_slices,
        num_classes,
        freeze_backbone,
        dropout,
        **kwargs
    ):
        model = ResNet3SliceClassifier(
            input_slices=input_slices,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )
        return model