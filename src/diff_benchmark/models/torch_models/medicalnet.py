from functools import partial
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


def conv3x3x3(
    in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1
) -> nn.Conv3d:
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


def downsample_basic_block(
    x: torch.Tensor, planes: int, stride: int, no_cuda: bool = False
) -> torch.Tensor:
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

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module = None,
    ):
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

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module = None,
    ):
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

    def __init__(
        self,
        block: nn.Module,
        layers: list[int],
        shortcut_type: str = "B",
        no_cuda: bool = False,
        **kwargs: Any,
    ):
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

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        blocks: int,
        shortcut_type: str,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
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
        return x


# First, define a mapping for depth → (block, layers)
RESNET_DEPTHS = {
    10: (BasicBlock, [1, 1, 1, 1]),
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
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
        shortcut_type: str = "B",
        no_cuda: bool = False,
        pretrained: bool = False,
        pretrain_path: str | None = None,
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
            shortcut_type=shortcut_type,
            no_cuda=no_cuda,
        )

        if pretrained:
            if pretrain_path is None:
                raise ValueError("pretrained=True but no pretrain_path provided")

            state = torch.load(pretrain_path, map_location="cpu")

            # Handle common MedicalNet checkpoint formats
            if "state_dict" in state:
                state = state["state_dict"]

            # Remove possible Lightning prefixes
            new_state = {}
            for k, v in state.items():
                if k.startswith("module."):
                    new_state[k[len("module.") :]] = v
                else:
                    new_state[k] = v

            missing, unexpected = self.load_state_dict(new_state, strict=False)

            if missing:
                logger.warning(f"[MedicalNet] Missing keys: {missing}")
            if unexpected:
                logger.warning(f"[MedicalNet] Unexpected keys: {unexpected}")

        self.collate_with_augmentation = self.collate_with_augmentation
        self.mean = 0.5
        self.std = 0.5

    @staticmethod
    def collate_with_augmentation(
        batch, transform: Callable = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
