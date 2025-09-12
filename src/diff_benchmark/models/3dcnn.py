# import csv
# import json
# from pathlib import Path

# import numpy as np
# import torch
# from sklearn.model_selection import train_test_split
from torch import nn
# from torch.utils.data import DataLoader, Subset
from torchvision import models
from tqdm import tqdm

from diff_benchmark.models.base import TorchAbstractModel
class ResNet10Backbone(nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        resnet = models.resnet10
        self.out_dim = 512
    # def _make_layer():
        
class ResNetVolumeClassifier(nn.Module):
    def __init__(self, input_volumes, num_classes=2, **kwargs):
        super().__init__()
        self.backbone = ResNet10Backbone(**kwargs)
        self.fc = nn.Linear(1 * self.Model.out_dim, num_classes)
        
    def forward(self, x):
        feats = self.backbone(x)
        out = self.fc(feats)
        return out
class ResNetVolumeModel(TorchAbstractModel, nn.Module):
    def __init__(self, input_volume=1, num_classes=2, device="cuda", **kwargs):
        super(ResNetVolumeModel, self).__init__()
        self.device = device
        self.run_id = kwargs.get("run_id", "unnamed_run")
        self.epochs = kwargs.get("epochs", 100)
        self.model = ResNetVolumeClassifier()