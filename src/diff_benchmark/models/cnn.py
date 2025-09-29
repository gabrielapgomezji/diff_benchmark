import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

from diff_benchmark.models.base import TorchAbstractModel
from diff_benchmark.utils.logger import TrainLogger, MetricsManager


def collate_with_augmentation(batch, transform=None):
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
    return xs_aug, ys, gs


train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        # transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)
val_transforms = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


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
        subvols = x.unfold(
            dimension=1, size=3, step=3
        )  # (Batch, num_subvols, 3, Height, Width)
        subvols = subvols.permute(0, 1, 4, 2, 3)
        num_subvols = subvols.size(1)

        feats = []
        for i in range(num_subvols):
            sv = subvols[:, i]  # (Batch, 3, Height, Width)
            f = self.backbone(sv)  # (Batch, 512)
            feats.append(f)

        feats = torch.cat(feats, dim=1)  # (Batch, num_subvols*512)
        feats = self.dropout(feats)
        # Normalization layer
        out = self.fc(feats)  # (Batch, num_classes)
        return out


class ResNet3SliceModel(TorchAbstractModel, nn.Module):
    """
    ResNet3SliceModel is a PyTorch implementation of a 3-slice ResNet model for classification tasks.
    Attributes:
        device (str): The device to run the model on (e.g., "cuda" or "cpu").
        run_id (str): Identifier for the current run, defaults to "unnamed_run".
        epochs (int): Number of training epochs, defaults to 100.
        model (nn.Module): The ResNet3SliceClassifier model.
        criterion (nn.Module): Loss function used for training (CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        best_val_model (float): Best validation accuracy achieved during training.
        history (dict): Dictionary to store training and validation history.
    Methods:
        _dataloader_to_numpy(dataloader): Converts data from a DataLoader to NumPy arrays.
        _train_val_loader_split(train_loader, val_ratio=0.3): Splits the dataset into training and validation sets.
        _save_logs(history, save_path): Saves training history to a specified file format (JSON or CSV).
        fit(dataloader): Trains the model using the provided DataLoader.
        predict(dataloader): Generates predictions for the provided DataLoader.
    """

    def __init__(self, input_slices=145, num_classes=2, device="cuda", **kwargs):
        super(ResNet3SliceModel, self).__init__()
        self.device = device
        self.run_id = kwargs.get("run_id", "unnamed_run")
        self.epochs = kwargs.get("epochs", 100)
        self.model = ResNet3SliceClassifier(
            input_slices=input_slices, num_classes=num_classes, **kwargs
        ).to(device)
        self.criterion = nn.CrossEntropyLoss()
        lr = kwargs.get("learning_rate", 1e-5)
        weight_decay = kwargs.get("weight_decay", 1e-4)
        self.optimizer = torch.optim.Adam(
            (
                self.model.parameters()
                if not kwargs.get("freeze_backbone", True)
                else self.model.fc.parameters()
            ),
            lr=lr,
            weight_decay=weight_decay,
        )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        self.average = "binary"

        self.best_val_model = 0
        self.history = {
            "train": {"epoch": [], "batch": [], "loss": [], "metrics": []},
            "val": {
                "epoch": [],
                "loss": [],
                "metrics": [],
                "batch_train_idx": [],
            },
        }

    def _dataloader_to_numpy(self, dataloader):
        x, y, g = [], [], []
        for xb, yb, gb in dataloader:
            x.append(xb.numpy())
            y.append(yb.numpy())
            g.append(gb.numpy())
        return np.concatenate(x), np.concatenate(y), np.concatenate(g)

    def _train_val_loader_split(self, train_loader, val_ratio=0.3):
        dataset = train_loader.dataset  # access the underlying dataset
        n = len(dataset)
        genders = np.asarray(dataset.dataset.gender[dataset.indices])

        # for i in range(n):
        #     _, _, g = dataset[i]
        #     genders.append(g)
        # genders = np.array(genders)

        indices = np.arange(n)
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, stratify=genders, random_state=42
        )

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # train_transforms = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(15),
        #     # transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        #     transforms.Normalize(mean=[0.5], std=[0.5]),
        # ])
        # val_transforms = transforms.Compose([
        #     # transforms.Resize((224, 224)),
        #     transforms.Normalize(mean=[0.5], std=[0.5]),
        # ])

        # train_subset = TransformedDataset(train_subset, transform=train_transforms)
        # val_subset = TransformedDataset(val_subset, transform=val_transforms)

        train_loader_new = DataLoader(
            train_subset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=20,
            pin_memory=True,
            collate_fn=lambda batch: collate_with_augmentation(
                batch, transform=train_transforms
            ),
        )
        val_loader_new = DataLoader(
            val_subset,
            batch_size=128,
            shuffle=True,
            num_workers=20,
            pin_memory=True,
            collate_fn=lambda batch: collate_with_augmentation(
                batch, transform=val_transforms
            ),
        )
        return train_loader_new, val_loader_new

    def _save_logs(self, history, save_path):
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(history, f)
        elif path.suffix == ".csv":
            keys = history[0].keys()
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(history)
        else:
            raise ValueError("Save path must end with .json or .csv")

    def fit(self, dataloader):
        print(f"Device: {self.device}")
        self.model.train()
        train_loader, val_loader = self._train_val_loader_split(dataloader)
        logger = TrainLogger(run_id=self.run_id, save_dir="./data/results/logger", monitor="val_accuracy", mode="max")
        print(f"Dataloaders created")
        for epoch in tqdm(range(self.epochs)):

            print(f"Epoch {epoch}")
            for batch_train_idx, (xb, yb, _) in enumerate(train_loader):
                # print("Batch loaded")
                xb, yb = xb.to(self.device), yb.long().to(self.device)
                # print("Moved to device")
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                # print("Forward + Bakcward done")
                self.optimizer.step()

                y_true = yb.cpu().detach().numpy()
                y_pred = preds.argmax(dim=1).cpu().detach().numpy()
                
                metrics = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, average=self.average, zero_division="warn"),
                    "recall": recall_score(y_true, y_pred, average=self.average, zero_division="warn"),
                    "f1": f1_score(y_true, y_pred, average=self.average, zero_division="warn"),
                    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
                }
                self.history["train"]["epoch"].append(epoch)
                self.history["train"]["batch"].append(batch_train_idx)
                self.history["train"]["loss"].append(loss.item())
                self.history["train"]["metrics"].append(metrics)

                # print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
                if batch_train_idx % 10 == 0:
                    self.model.eval()
                    y_true = []
                    y_pred = []
                    val_loss = 0
                    with torch.no_grad():
                        for batch_val_idx, (xb, yb, _) in enumerate(val_loader):
                            print(f"Val: batch {batch_val_idx}")
                            xb, yb = xb.to(self.device), yb.long().to(self.device)
                            preds = self.model(xb)
                            loss = self.criterion(preds, yb)
                            val_loss += loss.item()

                            y_true.append(yb.cpu().numpy())
                            y_pred.append(preds.argmax(dim=1).cpu().numpy())
                    
                    val_loss /= len(val_loader)
                        
                    y_true = np.concatenate(y_true)
                    y_pred = np.concatenate(y_pred)
                    
                    metrics = {
                        "accuracy": accuracy_score(y_true, y_pred),
                        "precision": precision_score(y_true, y_pred, average=self.average, zero_division="warn"),
                        "recall": recall_score(y_true, y_pred, average=self.average, zero_division="warn"),
                        "f1": f1_score(y_true, y_pred, average=self.average, zero_division="warn"),
                        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
                    }
                    self.history["val"]["epoch"].append(epoch)
                    self.history["val"]["batch_train_idx"].append(batch_train_idx)
                    self.history["val"]["loss"].append(val_loss)
                    self.history["val"]["metrics"].append(metrics)

                    logger.save_checkpoint(self.model, epoch, metrics["accuracy"])                    
                    self.model.train()  
            self.scheduler.step(val_loss)
        logger.save_checkpoint(self.model, self.epochs, 0, is_last=True)
        logger.save_logs()
        self._save_logs(
            self.history, f"./data/results/logs/{self.run_id}_training_log.json"
        )

    def predict(self, dataloader):
        self.model.eval()
        preds_all = []
        with torch.no_grad():
            for xb, _, _ in dataloader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                preds = torch.argmax(logits, dim=1)
                preds_all.append(preds.cpu())
        return torch.cat(preds_all).numpy()


class AttentionPool(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, feats):
        # feats: (B, num_subvols, 512)
        attn_weights = torch.softmax(self.attn(feats), dim=1)  # (B, num_subvols, 1)
        pooled = torch.sum(attn_weights * feats, dim=1)  # (B, 512)
        return pooled
