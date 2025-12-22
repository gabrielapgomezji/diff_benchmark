import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

from diff_benchmark.models.base import LightningModel
from diff_benchmark.models.utils import create_trainer
from typing import Callable


def collate_with_augmentation(batch: list, transform: Callable = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate function that applies 2D augmentations to each slice of 3D volumes in the batch.
    Args:
        batch (list): A list of tuples, where each tuple contains (x, y, g).
                      x is a 3D tensor (D, H, W), y is the label, and g is additional info.
        transform (Callable, optional): A torchvision transform to apply to each 2D slice. Defaults to None.
    Returns:
        tuple: A tuple containing:
            - xs_aug (torch.Tensor): Augmented 4D tensor of shape (B, C=1, D, H, W).
            - ys (torch.Tensor): Tensor of labels of shape (B,).
            - gs (torch.Tensor): Tensor of additional info of shape (B,).
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
                represents the flattened feature dimension.
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

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CNN model.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Slice, Height, Width).
                              If the input is a list or tuple, the first element is used.
        Returns:
            torch.Tensor: Output tensor of shape (Batch, num_classes).
        The forward pass performs the following steps:
        1. If the input is a list or tuple, extracts the first element.
        2. Removes the singleton dimension at index 1 using `squeeze`.
        3. Splits the input tensor into sub-volumes along the slice dimension using `unfold`.
           Each sub-volume has a size of 3 slices.
        4. Rearranges the dimensions of the sub-volumes for processing.
        5. Iterates over each sub-volume, passing it through the backbone network to extract features.
        6. Concatenates the features from all sub-volumes along the feature dimension.
        7. Applies dropout to the concatenated features.
        8. Passes the features through a fully connected layer to produce the final output.
        Note:
            - The slice dimension of the input tensor must be divisible by 3.
            - The backbone network is expected to output a feature vector of size 512 for each sub-volume.
        """
        
        # batch, slice, height, width = x.shape
        # assert S % 3 == 0, "Slice dimension must be divisible by 3"
        if isinstance(x, (list, tuple)):
            x = x[0]
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


class ResNet3SliceModel(LightningModel):
    """
    Lightning-based implementation of the 3-slice ResNet model.
    Retains the same API as the original TorchAbstractModel (fit, predict),
    but runs fully under the PyTorch Lightning training framework.
    Attributes:
        data_type (str): Type of data the model works with, set to "images".
    Args:
        input_slices (int): Number of input slices for the model.
        num_classes (int, optional): Number of output classes. Default is 2.
        device (str, optional): Device to run the model on. Default is "cuda".
        **kwargs: Additional keyword arguments for model configuration.
    Methods:
        build_model():
            Builds the ResNet3SliceClassifier model.
        forward(x):
            Forward pass of the model.
        _train_val_loader_split(train_loader, val_ratio):
            Splits the input dataloader into training and validation sets.
        _save_logs(history, save_path):
            Saves training logs to a specified path in JSON or CSV format.
        fit(dataloader):
            Fits the model using the provided dataloader.
        x_only_loader(dl):
            Creates a dataloader that yields only inputs (no labels).
        predict(dataloader):
            Predicts outputs using the provided dataloader.
    """

    data_type = "images"

    def __init__(self, input_slices: int =145, num_classes: int =2, device: str ="cuda", **kwargs):
        super().__init__(
            learning_rate=kwargs.get("learning_rate", 1e-5),
            weight_decay=kwargs.get("weight_decay", 1e-4),
            average="binary",
            scheduler_type=kwargs.get("weight_decay", "plateau"),
            optimizer_type=kwargs.get("optimizer_type", "adamw"),
        )

        self.device_str = device
        self.run_id = kwargs.get("run_id", "unnamed_run")
        self.fold_idx = kwargs.get("fold_idx", -1)
        self.epochs = kwargs.get("epochs", 100)
        self.input_slices = input_slices
        self.num_classes = num_classes
        self.freeze_backbone = kwargs.get("freeze_backbone", True)
        self.save_hyperparameters()

        # Build the model and loss
        self.build_model()
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------
    # Model definition
    # ------------------------------------------------------------
    def build_model(self):
        """Build the actual ResNet classifier."""
        # To avoid repeating args as input_slices
        model_kwargs = {
            k: v
            for k, v in vars(self.hparams).items()
            if k
            not in [
                "input_slices",
                "num_classes",
                "learning_rate",
                "weight_decay",
                "average",
            ]
        }
        self.model = ResNet3SliceClassifier(
            input_slices=self.input_slices,
            num_classes=self.num_classes,
            **model_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------
    def _train_val_loader_split(self, train_loader: DataLoader, val_ratio: float = 0.3) -> tuple[DataLoader, DataLoader]:
        """Split the incoming dataloader's dataset into train and validation subsets.
        Args:
            train_loader (DataLoader): The original training dataloader.
            val_ratio (float, optional): Proportion of data to use for validation. Defaults to 0.3.
        Returns:
            tuple: A tuple containing the new training and validation dataloaders.
        """
        dataset = train_loader.dataset  # access the underlying dataset
        n = len(dataset)
        genders = np.asarray(dataset.dataset.gender[dataset.indices])

        indices = np.arange(n)
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, stratify=genders, random_state=42
        )

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Transform-aware collation (optional)
        train_loader_new = DataLoader(
            train_subset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=19,  # 0,#
            pin_memory=False,
            collate_fn=lambda batch: collate_with_augmentation(
                batch, transform=train_transforms
            ),
        )
        val_loader_new = DataLoader(
            val_subset,
            batch_size=128,
            shuffle=False,
            num_workers=19,  # 0,#10,
            pin_memory=False,
            collate_fn=lambda batch: collate_with_augmentation(
                batch, transform=val_transforms
            ),
        )
        return train_loader_new, val_loader_new

    def _save_logs(self, history: list[dict], save_path: str):
        """Utility for saving training logs as JSON or CSV.
        Args:
            history (list): List of dictionaries containing training logs.
            save_path (str): Path to save the logs, must end with .json or .csv.
        """
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

    # ------------------------------------------------------------
    # Lightning-compatible fit/predict interface
    # ------------------------------------------------------------
    def fit(self, dataloader: DataLoader):
        """
        Lightning-based fit function to preserve compatibility with the old API.
        Splits the input dataloader into training and validation sets,
        sets up the trainer with early stopping and checkpointing,
        and runs Trainer.fit().
        Args:
            dataloader (DataLoader): The input dataloader for training.
        """
        print(f"Device: {self.device_str}")
        print(f"Fold index: {self.fold_idx}")

        train_loader, val_loader = self._train_val_loader_split(dataloader)
        print("Dataloaders created.")

        trainer = create_trainer(
            max_epochs=self.epochs,
            monitor="val_accuracy",
            mode="max",
            patience=10,
            accelerator="gpu" if "cuda" in self.device_str else "cpu",
            devices=1,
            save_dir=f"./data/results/checkpoints/{self.run_id}/fold_{self.fold_idx}",
        )

        trainer.fit(self, train_loader, val_loader)
        self.trainer = trainer  # store for predict later

        print(
            f"[INFO] Training finished. Best model: {trainer.checkpoint_callback.best_model_path}"
        )

    def x_only_loader(self, dl: DataLoader):
        """Utility to create a dataloader that yields only inputs (no labels).
        Args:
            dl (DataLoader): Original dataloader yielding (x, y, g).
        Yields:
            tuple: A tuple containing only the input tensor x.
        """
        for x, _, _ in dl:
            if isinstance(x, list):
                x = torch.stack(x)
            # Ensure it’s a 5D tensor (B, 1, D, H, W)
            if x.dim() == 4:
                x = x.unsqueeze(1)
            yield (x,)

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """
        Lightning-based predict function to preserve the old API.
        Automatically loads the best checkpoint from the Trainer.
        Args:
            dataloader (DataLoader): The input dataloader for prediction.
        Returns:
            np.ndarray: The predicted outputs as a NumPy array.
        """
        dataset = dataloader.dataset
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=19,  # 0,#10,
            pin_memory=False,
            collate_fn=lambda batch: collate_with_augmentation(
                batch, transform=val_transforms
            ),
        )
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            # If fit() hasn’t been run, create a default trainer
            trainer = create_trainer(
                accelerator="gpu" if "cuda" in self.device_str else "cpu",
                devices=1,
                max_epochs=1,
            )

        # Load best model checkpoint automatically
        best_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
        # if best_path and Path(best_path).exists():
        #     state_dict = torch.load(best_path, map_location=self.device_str)
        #     self.load_state_dict(state_dict["state_dict"], strict=False)
        #     print(f"[INFO] Loaded checkpoint from {best_path}")

        self.eval()
        # preds_all = trainer.predict(self, dataloaders=self.x_only_loader(dataloader))
        if best_path and Path(best_path).exists():
            preds_all = trainer.predict(
                self, dataloaders=dataloader, ckpt_path=best_path
            )
        else:
            preds_all = trainer.predict(
                self, dataloaders=self.x_only_loader(dataloader)
            )
        # preds_all = trainer.predict(self, dataloaders=dataloader)
        preds = torch.cat([p.cpu() for p in preds_all])
        return preds.numpy()
