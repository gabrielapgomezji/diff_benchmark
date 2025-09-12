import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import models
from tqdm import tqdm

from diff_benchmark.models.base import TorchAbstractModel


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
        # self.optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=1e-5)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.best_val_model = 0
        self.history = {
            "train": {"epoch": [], "batch": [], "loss": [], "accuracy": []},
            "val": {
                "epoch": [],
                "batch": [],
                "loss": [],
                "accuracy": [],
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

        genders = []
        for i in range(n):
            _, _, g = dataset[i]
            genders.append(g)
        genders = np.array(genders)

        indices = np.arange(n)
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, stratify=genders, random_state=42
        )

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader_new = DataLoader(
            train_subset, batch_size=train_loader.batch_size, shuffle=True
        )
        val_loader_new = DataLoader(
            val_subset, batch_size=train_loader.batch_size, shuffle=False
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
        for epoch in tqdm(range(self.epochs)):
            total_loss = 0
            train_accuracy = 0
            for batch_train_idx, (xb, yb, _) in enumerate(train_loader):
                xb, yb = xb.to(self.device), yb.long().to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                train_current_loss = loss.item()
                train_accuracy += (preds.argmax(dim=1) == yb).float().mean().item()
                train_current_accuracy = (
                    (preds.argmax(dim=1) == yb).float().mean().item()
                )
                # avg_train_accuracy = train_accuracy / len(train_loader) # NOT USED FOR THE MOMENT

                self.history["train"]["epoch"].append(epoch)
                self.history["train"]["batch"].append(batch_train_idx)
                self.history["train"]["loss"].append(loss.item())
                self.history["train"]["accuracy"].append(
                    (preds.argmax(dim=1) == yb).float().mean().item()
                )

                # print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
                if batch_train_idx % 3 == 0:
                    self.model.eval()
                    val_loss = 0
                    val_accuracy = 0
                    with torch.no_grad():
                        for batch_val_idx, (xb, yb, _) in enumerate(val_loader):
                            xb, yb = xb.to(self.device), yb.long().to(self.device)
                            preds = self.model(xb)
                            loss = self.criterion(preds, yb)
                            val_loss += loss.item()
                            val_current_loss = loss.item()
                            val_accuracy += (
                                (preds.argmax(dim=1) == yb).float().mean().item()
                            )
                            val_current_accuracy = (
                                (preds.argmax(dim=1) == yb).float().mean().item()
                            )

                            self.history["val"]["epoch"].append(epoch)
                            self.history["val"]["batch"].append(batch_val_idx)
                            self.history["val"]["loss"].append(loss.item())
                            self.history["val"]["accuracy"].append(
                                (preds.argmax(dim=1) == yb).float().mean().item()
                            )
                            self.history["val"]["batch_train_idx"].append(
                                batch_train_idx
                            )

                    # avg_val_loss = val_loss / len(val_loader)   # NOT USED FOR THE MOMENT
                    avg_val_accuracy = val_accuracy / len(val_loader)
                    if avg_val_accuracy > self.best_val_model:
                        save_path = Path("./data/models") / f"{self.run_id}_best.pth"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        self.best_val_model = avg_val_accuracy
                        torch.save(
                            self.model.state_dict(),
                            save_path,
                        )
                    self.model.train()  # switch back to train mode

                # avg_train_loss = total_loss / (batch_train_idx + 1)  # len(train_loader) #NOT USED FOR THE MOMENT

                # self.scheduler.step()
                # print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_accuracy:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_accuracy:.4f}")
                print(
                    f"Epoch {epoch+1}: Train Loss={train_current_loss:.4f}, Train Acc={train_current_accuracy:.4f}, Val Loss={val_current_loss:.4f}, Val Acc={val_current_accuracy:.4f}"
                )
        self._save_logs(self.history, f"./data/results/{self.run_id}_training_log.json")

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
