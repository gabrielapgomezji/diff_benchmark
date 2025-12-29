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

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = PredictionHead(
            embedding_dim=512 * block.expansion,
            prediction_task=prediction_task,
            num_classes=num_classes, # for regression is specified to 1
            hidden_dims=None,
            dropout=0.0,
        )

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
        x = self.fc(x)

        return x


def resnet10(**kwargs) -> ResNet:
    """
    Constructs a ResNet-10 model.
    This function creates a ResNet model with 10 layers using the BasicBlock
    building block. The layer configuration is defined as [1, 1, 1, 1], which
    specifies the number of blocks in each of the four layers of the network.
    Args:
        **kwargs: Additional keyword arguments passed to the ResNet constructor.
    Returns:
        ResNet: An instance of the ResNet-10 model.
    """
    
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs) -> ResNet:
    """
    Constructs a ResNet-18 model.
    This function initializes a ResNet-18 model using the ResNet architecture
    with BasicBlock layers and a predefined layer configuration of [2, 2, 2, 2].
    Args:
        **kwargs: Additional keyword arguments to be passed to the ResNet constructor.
                  These can include parameters such as the number of input channels,
                  the number of classes for classification, etc.
    Returns:
        ResNet: An instance of the ResNet-18 model.
    """
    
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs) -> ResNet:
    """
    Constructs a ResNet-34 model.
    This function creates a ResNet-34 architecture using the ResNet class and 
    the BasicBlock building block. The ResNet-34 model is defined by the 
    layer configuration [3, 4, 6, 3], which specifies the number of blocks 
    in each of the four layers of the network.
    Args:
        **kwargs: Additional keyword arguments passed to the ResNet class 
                  constructor. These can include parameters such as the 
                  number of input channels, number of classes, etc.
    Returns:
        ResNet: An instance of the ResNet-34 model.
    """
    
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs) -> ResNet:
    """
    Constructs a ResNet-50 model.
    This function creates a ResNet-50 architecture using the Bottleneck block 
    and a predefined layer configuration of [3, 4, 6, 3]. Additional arguments 
    can be passed to customize the model.
    Args:
        **kwargs: Arbitrary keyword arguments passed to the ResNet constructor.
    Returns:
        ResNet: An instance of the ResNet-50 model.
    """
    
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs) -> ResNet:
    """
    Creates a ResNet-101 model.
    This function constructs a ResNet-101 architecture using the `ResNet` class and the `Bottleneck` block. 
    The ResNet-101 model is defined by the layer configuration [3, 4, 23, 3], which specifies the number 
    of blocks in each of the four layers of the network.
    Args:
        **kwargs: Additional keyword arguments to be passed to the `ResNet` class.
    Returns:
        ResNet: An instance of the ResNet-101 model.
    """
    
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs) -> ResNet:
    """
    Constructs a ResNet-152 model.
    ResNet-152 is a deep residual network architecture with 152 layers, 
    which is commonly used for image recognition tasks. This function 
    initializes the model using the Bottleneck block and a specific 
    layer configuration.
    Args:
        **kwargs: Additional keyword arguments to customize the ResNet model. 
                  These arguments are passed to the ResNet constructor.
    Returns:
        ResNet: An instance of the ResNet-152 model.
    """
    
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs) -> ResNet:
    """
    Constructs a ResNet-200 model.
    This function initializes a ResNet-200 architecture using the Bottleneck
    building block and a layer configuration of [3, 24, 36, 3]. The ResNet-200
    model is a deep residual network designed for image or feature extraction tasks.
    Args:
        **kwargs: Additional keyword arguments to be passed to the ResNet constructor.
                  These may include parameters such as the number of input channels,
                  number of classes, or other model-specific configurations.
    Returns:
        ResNet: An instance of the ResNet-200 model.
    """
    
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


def generate_model(opt: Any) -> ResNet:
    """
    Generates a ResNet model based on the provided options.
    Args:
        opt (Any): A configuration object containing the following attributes:
            - model (str): The type of model to generate. Must be "resnet".
            - model_depth (int): The depth of the ResNet model. Must be one of 
              [10, 18, 34, 50, 101, 152, 200].
            - input_W (int): The width of the input sample.
            - input_H (int): The height of the input sample.
            - input_D (int): The depth of the input sample.
            - resnet_shortcut (str): The type of shortcut connection to use in ResNet.
            - no_cuda (bool): Whether to disable CUDA (GPU) support.
            - n_seg_classes (int): The number of segmentation classes.
            - gpu_id (list[int]): List of GPU IDs to use for training.
            - phase (str): The phase of the model, e.g., "train" or "test".
            - pretrain_path (str): Path to the pretrained model file (if any).
            - new_layer_names (list[str]): List of layer names to treat as new parameters.
    Returns:
        Tuple[ResNet, Union[Iterable[torch.nn.Parameter], Dict[str, Iterable[torch.nn.Parameter]]]]:
            - The generated ResNet model.
            - The model parameters or a dictionary containing base and new parameters 
              (if pretrained model is loaded and new layers are specified).
    Notes:
        - If `opt.no_cuda` is False and multiple GPUs are specified in `opt.gpu_id`, 
          the model is wrapped in `torch.nn.DataParallel`.
        - If `opt.phase` is not "test" and `opt.pretrain_path` is provided, the model 
          is initialized with the pretrained weights. New parameters are separated 
          based on `opt.new_layer_names`.
    """

    assert opt.model in ["resnet"]

    if opt.model == "resnet":
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
        elif opt.model_depth == 18:
            model = resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
        elif opt.model_depth == 34:
            model = resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
        elif opt.model_depth == 50:
            model = resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
        elif opt.model_depth == 101:
            model = resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
        elif opt.model_depth == 152:
            model = resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
        elif opt.model_depth == 200:
            model = resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )

    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict()
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    # load pretrain
    if opt.phase != "test" and opt.pretrain_path:
        print(f"loading pretrained model {opt.pretrain_path}")
        pretrain = torch.load(opt.pretrain_path)
        pretrain_dict = {
            k: v for k, v in pretrain["state_dict"].items() if k in net_dict.keys()
        }

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(
            filter(lambda p: id(p) not in new_parameters_id, model.parameters())
        )
        parameters = {
            "base_parameters": base_parameters,
            "new_parameters": new_parameters,
        }

        return model, parameters

    return model, model.parameters()

def collate_with_augmentation(batch, transform: callable =None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates a batch of data with optional augmentation and normalization.
    Args:
        batch (list of tuples): A batch of data where each element is a tuple 
            containing three tensors (x, y, g). `x` represents the input data, 
            `y` represents the labels, and `g` represents additional metadata.
        transform (callable, optional): A callable transformation function to 
            apply to the input data `x`. Defaults to None.
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - xs (torch.Tensor): The stacked and normalized input data tensor 
                with shape (B, 1, D, H, W), where B is the batch size.
            - ys (torch.Tensor): The stacked labels tensor.
            - gs (torch.Tensor): The stacked metadata tensor.
    Notes:
        - The input data `x` is normalized using a mean of 0.5 and a standard 
            deviation of 0.5.
        - If a transformation function is provided, it should be applied to 
            the input data before stacking.
    """
    
    mean = 0.5
    std = 0.5
    xs, ys, gs = zip(*batch)
    # xs = torch.stack(xs, dim=0)   # default stacking: (B, 1, D, H, W)
    xs = torch.stack([x.unsqueeze(0) for x in xs], dim=0)
    ys = torch.stack(ys)
    gs = torch.stack(gs)

    # Normalize: (x - mean) / std
    xs = (xs - mean) / std

    return xs, ys, gs


class ResNet3DModelLite(LightningModel, nn.Module):
    """
    ResNet3DModel
    A wrapper around a 3D ResNet (resnet10) for medical-volume classification built on PyTorch.
    This class combines model construction, optional pretrained-weight loading, training (with
    a simple inner-loop validation and early stopping), prediction, and lightweight logging.
    - input_volumes: int, default 1
        Kept for API compatibility; currently unused. The implementation expects input volumes
        as 3D tensors and will add a channel dimension before forwarding (unsqueeze(1)).
    - num_classes: int, default 2
        Number of output classes for classification (final linear layer output dimension).
    - device: str or torch.device, default "cuda"
        Device used for model and tensor transfers.
    - **kwargs: optional keyword arguments
        - run_id: str, default "unnamed_run" -- identifier used to name saved model & logs.
        - epochs: int, default 100 -- number of training epochs to iterate (outer loop).
        - pretrain_path: str or Path, optional -- path to a checkpoint file; if provided, the
          checkpoint is loaded (torch.load) and, if it contains a "state_dict" key that mapping
          is used. load_state_dict(..., strict=False) is used to allow partial matches.
        - learning_rate: float, default 1e-5 -- Adam optimizer learning rate.
        - weight_decay: float, default 1e-4 -- Adam optimizer weight decay (L2).
        (Other kwargs are ignored by the implementation.)
    """

    data_type = "images"

    def __init__(
        self, device: torch.device, num_classes: int = 2, input_channels: int = 1, model_depth: int = 10, **kwargs
    ):
        super().__init__(
            learning_rate=kwargs.get("learning_rate", 1e-5),
            weight_decay=kwargs.get("weight_decay", 1e-4),
            scheduler_type=kwargs.get("scheduler_type", "plateau"),
            optimizer_type=kwargs.get("optimizer_type", "adamw"),
            prediction_task = kwargs.get("prediction_task", None)
        )
        self.run_id = kwargs.get("run_id", "unnamed_run")
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.model_depth = model_depth
        self.device_str = device
        self.fold_idx = kwargs.get("fold_idx", -1)
        self.epochs = kwargs.get("epochs", 100)
        self.prediction_task = kwargs.get("prediction_task", None)

        self.save_hyperparameters()

        self.build_model()  # required by parent
        # criterion already set in LightningModel

    def build_model(self):
        """
        Build the ResNet model based on the specified depth.
        This method initializes the `self.model` attribute with a ResNet model
        corresponding to the specified `self.model_depth`. The number of output
        classes is determined by `self.num_classes`.
        Supported ResNet depths:
            - 10: Initializes a ResNet-10 model.
            - 18: Initializes a ResNet-18 model.
            - 34: Initializes a ResNet-34 model.
            - 50: Initializes a ResNet-50 model.
        Raises:
            ValueError: If `self.model_depth` is not one of the supported depths.
        """
        
        if self.model_depth == 10:
            self.model = resnet10(num_classes=self.num_classes, prediction_task=self.prediction_task)
        elif self.model_depth == 18:
            self.model = resnet18(num_classes=self.num_classes, prediction_task=self.prediction_task)
        elif self.model_depth == 34:
            self.model = resnet34(num_classes=self.num_classes, prediction_task=self.prediction_task)
        elif self.model_depth == 50:
            self.model = resnet50(num_classes=self.num_classes, prediction_task=self.prediction_task)
        elif self.model_depth == 101:
            self.model = resnet101(num_classes=self.num_classes, prediction_task=self.prediction_task)
        elif self.model_depth == 152:
            self.model = resnet152(num_classes=self.num_classes, prediction_task=self.prediction_task)
        elif self.model_depth == 200:
            self.model = resnet200(num_classes=self.num_classes, prediction_task=self.prediction_task)
        else:
            raise ValueError(f"Unsupported ResNet depth: {self.model_depth}")
        self.model.collate_with_augmentation = collate_with_augmentation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor. Expected shape is either (B, C, D, H, W) 
                              where C is the channel dimension, or (B, D, H, W) 
                              if the channel dimension is omitted.
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        
        # If user supplies input without channel dim → add it
        if x.ndim == 4:
            x = x.unsqueeze(1)  # (B, 1, D, H, W)
        return self.model(x)


class ResNet3DModel(TorchPipeline):
    """
    ResNet3DModel
    A wrapper around a 3D ResNet (resnet10) for medical-volume classification built on PyTorch.
    This class combines model construction, optional pretrained-weight loading, training (with
    a simple inner-loop validation and early stopping), prediction, and lightweight logging.
    - input_volumes: int, default 1
        Kept for API compatibility; currently unused. The implementation expects input volumes
        as 3D tensors and will add a channel dimension before forwarding (unsqueeze(1)).
    - num_classes: int, default 2
        Number of output classes for classification (final linear layer output dimension).
    - device: str or torch.device, default "cuda"
        Device used for model and tensor transfers.
    - **kwargs: optional keyword arguments
        - run_id: str, default "unnamed_run" -- identifier used to name saved model & logs.
        - epochs: int, default 100 -- number of training epochs to iterate (outer loop).
        - pretrain_path: str or Path, optional -- path to a checkpoint file; if provided, the
          checkpoint is loaded (torch.load) and, if it contains a "state_dict" key that mapping
          is used. load_state_dict(..., strict=False) is used to allow partial matches.
        - learning_rate: float, default 1e-5 -- Adam optimizer learning rate.
        - weight_decay: float, default 1e-4 -- Adam optimizer weight decay (L2).
        (Other kwargs are ignored by the implementation.)
    """

    data_type = "images"

    def _build_model(self, num_classes: int, model_depth=10, **kwargs) -> ResNet:
        """
        Build a ResNet model with the specified depth and number of classes.
        Args:
            num_classes (int): The number of output classes for the model.
            model_depth (int, optional): The depth of the ResNet model. Supported values are 
                10, 18, 34, 50, 101, 152, and 200. Defaults to 10.
            **kwargs: Additional keyword arguments. Supported keys:
                - prediction_task (str, optional): Specifies the prediction task for the model.
        Returns:
            ResNet: An instance of the ResNet model with the specified configuration.
        Raises:
            ValueError: If an unsupported ResNet depth is provided.
        """
        
        prediction_task = kwargs.get("prediction_task", None)
        # model = resnet10(num_classes=num_classes)
        if model_depth == 10:
            model = resnet10(num_classes=num_classes, prediction_task=prediction_task)
        elif model_depth == 18:
            model = resnet18(num_classes=num_classes, prediction_task=prediction_task)
        elif model_depth == 34:
            model = resnet34(num_classes=num_classes, prediction_task=prediction_task)
        elif model_depth == 50:
            model = resnet50(num_classes=num_classes, prediction_task=prediction_task)
        elif model_depth == 101:
            model = resnet101(num_classes=num_classes, prediction_task=prediction_task)
        elif model_depth == 152:
            model = resnet152(num_classes=num_classes, prediction_task=prediction_task)
        elif model_depth == 200:
            model = resnet200(num_classes=num_classes, prediction_task=prediction_task)
        else:
            raise ValueError(f"Unsupported ResNet depth: {model_depth}")
        model.collate_with_augmentation = collate_with_augmentation
        model.mean = 0.5
        model.std = 0.5
        return model
