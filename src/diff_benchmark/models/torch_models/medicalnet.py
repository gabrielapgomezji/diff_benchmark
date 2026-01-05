import csv
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from diff_benchmark.models.base import TorchPipeline, LightningModel
from diff_benchmark.models.utils import create_trainer
from diff_benchmark.models.utils_models.prediction_head import PredictionHead
from typing import Any

__all__ = [
    "ResNet",
    "resnet10",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet200",
]

def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv3d:
    """
    Creates a 3D convolutional layer with a 3x3x3 kernel.
    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.
        dilation (int, optional): Dilation rate for the convolution. Default is 1.
    Returns:
        nn.Conv3d: A 3D convolutional layer with the specified parameters.
    """
    
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False,
    )


def downsample_basic_block(x: torch.Tensor, planes: int, stride: int, no_cuda: bool = False) -> torch.Tensor:
    """
    Downsamples a 3D tensor using average pooling and zero-padding.
    Args:
        x (torch.Tensor): The input tensor with shape (N, C, D, H, W), where
            N is the batch size, C is the number of channels, and D, H, W are
            the spatial dimensions.
        planes (int): The target number of channels after downsampling.
        stride (int): The stride for the average pooling operation.
        no_cuda (bool, optional): If True, ensures the operation is performed
            on the CPU even if the input tensor is on the GPU. Defaults to False.
    Returns:
        torch.Tensor: The downsampled tensor with shape (N, planes, D', H', W'),
        where D', H', W' are the spatial dimensions after downsampling.
    """
    
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    ).zero_()
    if not no_cuda:
        if out.is_cuda and out.dtype == torch.float32:
            zero_pads = zero_pads.cuda()

    out = torch.cat([out.data, zero_pads], dim=1)

    return out


class BasicBlock(nn.Module):
    """
    A BasicBlock module for a 3D convolutional neural network.
    This block is a fundamental building block for constructing residual networks.
    It consists of two 3D convolutional layers, each followed by batch normalization
    and a ReLU activation. The block also supports downsampling and dilation for
    adjusting the spatial dimensions of the input.
    """

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W, D), where
                N is the batch size, C is the number of channels, and H, W, D
                are the spatial dimensions.
        Returns:
            torch.Tensor: Output tensor after applying the convolutional layers,
            batch normalization, ReLU activation, and residual connection.
        """
        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for a 3D convolutional neural network.
    This class implements a bottleneck block, which is a building block for
    deep residual networks. It uses three convolutional layers with Batch
    Normalization and ReLU activation. The block supports downsampling and
    dilated convolutions.
    Attributes:
        expansion (int): Expansion factor for the output channels of the third
            convolutional layer. Default is 4.
        conv1 (nn.Conv3d): First 1x1x1 convolutional layer.
        bn1 (nn.BatchNorm3d): Batch normalization for the first convolutional layer.
        conv2 (nn.Conv3d): Second 3x3x3 convolutional layer.
        bn2 (nn.BatchNorm3d): Batch normalization for the second convolutional layer.
        conv3 (nn.Conv3d): Third 1x1x1 convolutional layer.
        bn3 (nn.BatchNorm3d): Batch normalization for the third convolutional layer.
        relu (nn.ReLU): ReLU activation function.
        downsample (callable, optional): Downsampling layer to match the dimensions
            of the input and output. Default is None.
        stride (int): Stride for the second convolutional layer. Default is 1.
        dilation (int): Dilation rate for the second convolutional layer. Default is 1.
    Methods:
        forward(x):
            Performs the forward pass of the bottleneck block. Applies three
            convolutional layers with Batch Normalization and ReLU activation,
            adds the residual connection, and applies the final ReLU activation.
    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels for the first and second
            convolutional layers. The third convolutional layer outputs
            `planes * expansion` channels.
        stride (int, optional): Stride for the second convolutional layer. Default is 1.
        dilation (int, optional): Dilation rate for the second convolutional layer. Default is 1.
        downsample (callable, optional): Downsampling layer to match the dimensions
            of the input and output. Default is None.
    """

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W), where
                N is the batch size, C is the number of channels, and D, H, W
                are the depth, height, and width of the input tensor, respectively.
        Returns:
            torch.Tensor: Output tensor after applying the convolutional layers,
            batch normalization, ReLU activation, and residual connection.
        """
        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet is a 3D convolutional neural network model designed for processing volumetric data.
    It is based on the ResNet architecture and supports custom configurations for the number
    of layers, blocks, and other parameters.
    Args:
        block (nn.Module): A block class that defines the building block of the ResNet model.
        layers (list of int): A list specifying the number of blocks in each layer of the network.
        num_classes (int): The number of output classes for the final fully connected layer.
        shortcut_type (str, optional): The type of shortcut connection to use ("A" or "B").
            Defaults to "B".
        no_cuda (bool, optional): If True, disables the use of CUDA for the model. Defaults to False.
    Attributes:
        conv1 (nn.Conv3d): The initial 3D convolutional layer.
        bn1 (nn.BatchNorm3d): Batch normalization layer for the initial convolutional layer.
        relu (nn.ReLU): ReLU activation function.
        maxpool (nn.MaxPool3d): Max pooling layer after the initial convolution.
        layer1 (nn.Sequential): The first residual layer.
        layer2 (nn.Sequential): The second residual layer.
        layer3 (nn.Sequential): The third residual layer with dilation.
        layer4 (nn.Sequential): The fourth residual layer with increased dilation.
        avgpool (nn.AdaptiveAvgPool3d): Adaptive average pooling layer to reduce spatial dimensions.
        fc (nn.Linear): Fully connected layer for classification.
    Methods:
        forward(x):
            Defines the forward pass of the ResNet model.
            Args:
                x (torch.Tensor): Input tensor of shape (N, C, D, H, W), where N is the batch size,
                    C is the number of channels, and D, H, W are the depth, height, and width of the
                    input volume.
            Returns:
                torch.Tensor: Output tensor of shape (N, num_classes), where N is the batch size and
                    num_classes is the number of output classes.
    """

    def __init__(self, block: nn.Module, layers: list[int], num_classes: int, prediction_task: str, shortcut_type: str = "B", no_cuda: bool = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super().__init__()
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False
        )

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4
        )
        self.out_dim = 512 * block.expansion
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = PredictionHead(
        #     embedding_dim=512 * block.expansion,
        #     prediction_task=prediction_task,
        #     num_classes=num_classes, # for regression is specified to 1
        #     hidden_dims=None,
        #     dropout=0.0,
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block: nn.Module, planes: int, blocks: int, shortcut_type: str, stride: int = 1, dilation: int = 1) -> nn.Sequential:
        """
        Creates a sequential layer consisting of multiple blocks.
        Args:
            block (nn.Module): The building block module to be used in the layer.
            planes (int): The number of output channels for the blocks.
            blocks (int): The number of blocks to include in the layer.
            shortcut_type (str): The type of shortcut connection to use. 
                Options are "A" for basic block downsampling or other types for 
                convolutional downsampling.
            stride (int, optional): The stride to use for the first block. Defaults to 1.
            dilation (int, optional): The dilation rate for the convolutional layers. Defaults to 1.
        Returns:
            nn.Sequential: A sequential container of the constructed blocks.
        """
        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where
                              N is the batch size, C is the number of channels,
                              H is the height, and W is the width.
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        if x.ndim == 4:  # (B, D, H, W)
            x = x.unsqueeze(1)  # → (B, 1, D, H, W)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x



# class MedicalNet(ResNet):

#     data_type = "images"

#     def __init__(self, depth: int, num_classes, prediction_task, shortcut_type = "B", no_cuda = False):
#         if depth == 10:
#             block = BasicBlock
#             layers = [1, 1, 1, 1]
#         elif depth == 18:
#             block = BasicBlock
#             layers = [2, 2, 2, 2]
#         elif depth == 34:
#             block = BasicBlock
#             layers = [3, 4, 6, 3]
#         elif depth == 50:
#             block = Bottleneck
#             layers = [3, 4, 6, 3]
#         elif depth == 101:
#             block = Bottleneck
#             layers = [3, 4, 23, 3]
#         elif depth == 152:
#             block = Bottleneck
#             layers = [3, 8, 36, 3]
#         elif depth == 200:
#             block = Bottleneck
#             layers = [3, 24, 36, 3]
#         super().__init__(block, layers, num_classes, prediction_task, shortcut_type, no_cuda)


#     def collate_with_augmentation(batch, transform: callable =None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Collates a batch of data with optional augmentation and normalization.
#         Args:
#             batch (list of tuples): A batch of data where each element is a tuple 
#                 containing three tensors (x, y, g). `x` represents the input data, 
#                 `y` represents the labels, and `g` represents additional metadata.
#             transform (callable, optional): A callable transformation function to 
#                 apply to the input data `x`. Defaults to None.
#         Returns:
#             tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
#                 - xs (torch.Tensor): The stacked and normalized input data tensor 
#                     with shape (B, 1, D, H, W), where B is the batch size.
#                 - ys (torch.Tensor): The stacked labels tensor.
#                 - gs (torch.Tensor): The stacked metadata tensor.
#         Notes:
#             - The input data `x` is normalized using a mean of 0.5 and a standard 
#                 deviation of 0.5.
#             - If a transformation function is provided, it should be applied to 
#                 the input data before stacking.
#         """
        
#         mean = 0.5
#         std = 0.5
#         xs, ys, gs = zip(*batch)
#         # xs = torch.stack(xs, dim=0)   # default stacking: (B, 1, D, H, W)
#         xs = torch.stack([x.unsqueeze(0) for x in xs], dim=0)
#         ys = torch.stack(ys)
#         gs = torch.stack(gs)

#         # Normalize: (x - mean) / std
#         xs = (xs - mean) / std

#         return xs, ys, gs

# First, define a mapping for depth → (block, layers)
RESNET_DEPTHS = {
    10:  (BasicBlock,  [1, 1, 1, 1]),
    18:  (BasicBlock,  [2, 2, 2, 2]),
    34:  (BasicBlock,  [3, 4, 6, 3]),
    50:  (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3]),
    200: (Bottleneck, [3, 24, 36, 3]),
}


class MedicalNet(ResNet):
    """
    MedicalNet backbone.
    Backend-agnostic (Torch / Lightning safe).
    Encapsulates a 3D ResNet architecture based on the specified depth.
    """

    data_type = "images"

    def __init__(
        self,
        depth: int,
        num_classes: int,
        prediction_task: str | None = None,
        shortcut_type: str = "B",
        no_cuda: bool = False,
        **kwargs: Any,
    ):
        if depth not in RESNET_DEPTHS:
            raise ValueError(
                f"Unsupported depth {depth}. Available depths: {sorted(RESNET_DEPTHS.keys())}"
            )

        block, layers = RESNET_DEPTHS[depth]

        super().__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            prediction_task=prediction_task,
            shortcut_type=shortcut_type,
            no_cuda=no_cuda,
        )
        self.collate_with_augmentation = self.collate_with_augmentation
        self.mean = 0.5
        self.std = 0.5
        
    @staticmethod
    def collate_with_augmentation(batch, transform: callable = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collates a batch of data with optional augmentation and normalization.
        Args:
            batch (list of tuples): Each element is (x, y, g).
            transform (callable, optional): Transformation function to apply to x.
        Returns:
            tuple of torch.Tensor: (xs, ys, gs)
        """
        mean = 0.5
        std = 0.5

        xs, ys, gs = zip(*batch)
        if transform:
            xs = [transform(x) for x in xs]
        
        xs = torch.stack([x.unsqueeze(0) for x in xs], dim=0)  # Add channel dim
        # xs = torch.stack([x.unsqueeze(0) if x.ndim == 3 else x for x in xs], dim=0)
        ys = torch.stack(ys)
        gs = torch.stack(gs)

        xs = (xs - mean) / std
        return xs, ys, gs
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = x.unsqueeze(1)  # Add channel dimension
    #     # if x.ndim == 4:
    #     #     x = x.unsqueeze(1)  # (B, 1, D, H, W)
    #     return x

    
# class ResNet3DModelLite(LightningModel, nn.Module):
#     """
#     ResNet3DModel
#     A wrapper around a 3D ResNet (resnet10) for medical-volume classification built on PyTorch.
#     This class combines model construction, optional pretrained-weight loading, training (with
#     a simple inner-loop validation and early stopping), prediction, and lightweight logging.
#     - input_volumes: int, default 1
#         Kept for API compatibility; currently unused. The implementation expects input volumes
#         as 3D tensors and will add a channel dimension before forwarding (unsqueeze(1)).
#     - num_classes: int, default 2
#         Number of output classes for classification (final linear layer output dimension).
#     - device: str or torch.device, default "cuda"
#         Device used for model and tensor transfers.
#     - **kwargs: optional keyword arguments
#         - run_id: str, default "unnamed_run" -- identifier used to name saved model & logs.
#         - epochs: int, default 100 -- number of training epochs to iterate (outer loop).
#         - pretrain_path: str or Path, optional -- path to a checkpoint file; if provided, the
#           checkpoint is loaded (torch.load) and, if it contains a "state_dict" key that mapping
#           is used. load_state_dict(..., strict=False) is used to allow partial matches.
#         - learning_rate: float, default 1e-5 -- Adam optimizer learning rate.
#         - weight_decay: float, default 1e-4 -- Adam optimizer weight decay (L2).
#         (Other kwargs are ignored by the implementation.)
#     """

#     data_type = "images"

#     def __init__(
#         self, device: torch.device, num_classes: int = 2, input_channels: int = 1, model_depth: int = 10, **kwargs
#     ):
#         super().__init__(
#             learning_rate=kwargs.get("learning_rate", 1e-5),
#             weight_decay=kwargs.get("weight_decay", 1e-4),
#             scheduler_type=kwargs.get("scheduler_type", "plateau"),
#             optimizer_type=kwargs.get("optimizer_type", "adamw"),
#             prediction_task = kwargs.get("prediction_task", None)
#         )
#         self.run_id = kwargs.get("run_id", "unnamed_run")
#         self.num_classes = num_classes
#         self.input_channels = input_channels
#         self.model_depth = model_depth
#         self.device_str = device
#         self.fold_idx = kwargs.get("fold_idx", -1)
#         self.epochs = kwargs.get("epochs", 100)
#         self.prediction_task = kwargs.get("prediction_task", None)

#         self.save_hyperparameters()

#         self.build_model()  # required by parent
#         # criterion already set in LightningModel

#     def build_model(self):
#         """
#         Build the ResNet model based on the specified depth.
#         This method initializes the `self.model` attribute with a ResNet model
#         corresponding to the specified `self.model_depth`. The number of output
#         classes is determined by `self.num_classes`.
#         Supported ResNet depths:
#             - 10: Initializes a ResNet-10 model.
#             - 18: Initializes a ResNet-18 model.
#             - 34: Initializes a ResNet-34 model.
#             - 50: Initializes a ResNet-50 model.
#         Raises:
#             ValueError: If `self.model_depth` is not one of the supported depths.
#         """
        
#         if self.model_depth == 10:
#             self.model = resnet10(num_classes=self.num_classes, prediction_task=self.prediction_task)
#         elif self.model_depth == 18:
#             self.model = resnet18(num_classes=self.num_classes, prediction_task=self.prediction_task)
#         elif self.model_depth == 34:
#             self.model = resnet34(num_classes=self.num_classes, prediction_task=self.prediction_task)
#         elif self.model_depth == 50:
#             self.model = resnet50(num_classes=self.num_classes, prediction_task=self.prediction_task)
#         elif self.model_depth == 101:
#             self.model = resnet101(num_classes=self.num_classes, prediction_task=self.prediction_task)
#         elif self.model_depth == 152:
#             self.model = resnet152(num_classes=self.num_classes, prediction_task=self.prediction_task)
#         elif self.model_depth == 200:
#             self.model = resnet200(num_classes=self.num_classes, prediction_task=self.prediction_task)
#         else:
#             raise ValueError(f"Unsupported ResNet depth: {self.model_depth}")
#         self.model.collate_with_augmentation = collate_with_augmentation
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the model.
#         Args:
#             x (torch.Tensor): Input tensor. Expected shape is either (B, C, D, H, W) 
#                               where C is the channel dimension, or (B, D, H, W) 
#                               if the channel dimension is omitted.
#         Returns:
#             torch.Tensor: Output tensor after passing through the model.
#         """
        
#         # If user supplies input without channel dim → add it
#         if x.ndim == 4:
#             x = x.unsqueeze(1)  # (B, 1, D, H, W)
#         return self.model(x)


# class ResNet3DModel(TorchPipeline):
#     """
#     ResNet3DModel
#     A wrapper around a 3D ResNet (resnet10) for medical-volume classification built on PyTorch.
#     This class combines model construction, optional pretrained-weight loading, training (with
#     a simple inner-loop validation and early stopping), prediction, and lightweight logging.
#     - input_volumes: int, default 1
#         Kept for API compatibility; currently unused. The implementation expects input volumes
#         as 3D tensors and will add a channel dimension before forwarding (unsqueeze(1)).
#     - num_classes: int, default 2
#         Number of output classes for classification (final linear layer output dimension).
#     - device: str or torch.device, default "cuda"
#         Device used for model and tensor transfers.
#     - **kwargs: optional keyword arguments
#         - run_id: str, default "unnamed_run" -- identifier used to name saved model & logs.
#         - epochs: int, default 100 -- number of training epochs to iterate (outer loop).
#         - pretrain_path: str or Path, optional -- path to a checkpoint file; if provided, the
#           checkpoint is loaded (torch.load) and, if it contains a "state_dict" key that mapping
#           is used. load_state_dict(..., strict=False) is used to allow partial matches.
#         - learning_rate: float, default 1e-5 -- Adam optimizer learning rate.
#         - weight_decay: float, default 1e-4 -- Adam optimizer weight decay (L2).
#         (Other kwargs are ignored by the implementation.)
#     """

#     data_type = "images"

#     def _build_model(self, num_classes: int, model_depth=10, **kwargs) -> ResNet:
#         """
#         Build a ResNet model with the specified depth and number of classes.
#         Args:
#             num_classes (int): The number of output classes for the model.
#             model_depth (int, optional): The depth of the ResNet model. Supported values are 
#                 10, 18, 34, 50, 101, 152, and 200. Defaults to 10.
#             **kwargs: Additional keyword arguments. Supported keys:
#                 - prediction_task (str, optional): Specifies the prediction task for the model.
#         Returns:
#             ResNet: An instance of the ResNet model with the specified configuration.
#         Raises:
#             ValueError: If an unsupported ResNet depth is provided.
#         """
        
#         prediction_task = kwargs.get("prediction_task", None)
#         # model = resnet10(num_classes=num_classes)
#         if model_depth == 10:
#             model = resnet10(num_classes=num_classes, prediction_task=prediction_task)
#         elif model_depth == 18:
#             model = resnet18(num_classes=num_classes, prediction_task=prediction_task)
#         elif model_depth == 34:
#             model = resnet34(num_classes=num_classes, prediction_task=prediction_task)
#         elif model_depth == 50:
#             model = resnet50(num_classes=num_classes, prediction_task=prediction_task)
#         elif model_depth == 101:
#             model = resnet101(num_classes=num_classes, prediction_task=prediction_task)
#         elif model_depth == 152:
#             model = resnet152(num_classes=num_classes, prediction_task=prediction_task)
#         elif model_depth == 200:
#             model = resnet200(num_classes=num_classes, prediction_task=prediction_task)
#         else:
#             raise ValueError(f"Unsupported ResNet depth: {model_depth}")
#         model.collate_with_augmentation = collate_with_augmentation
#         model.mean = 0.5
#         model.std = 0.5
#         return model
