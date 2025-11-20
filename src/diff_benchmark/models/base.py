from abc import ABC, abstractmethod
from typing import Optional
from tqdm import tqdm

import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import (  # confusion_matrix,; roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from diff_benchmark.utils.logger import TrainLogger
from sklearn.model_selection import train_test_split
from pathlib import Path
import json, csv

from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule, record_function
from torch.cuda.amp import autocast, GradScaler


def collate_with_augmentation(batch, transform=None):
    """Custom collate function that applies 2D augmentations to each slice of 3D volumes in the batch."""
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

class NumpyAbstractModel(ABC):
    """
    Abstract base class for all models in the diff_benchmark framework.
    Defines the interface that all models must implement.
    """

    @abstractmethod
    def _dataloader_to_numpy(self, dataloader):
        """
        Convert a DataLoader to numpy arrays.
        This method should be implemented by all subclasses to handle the conversion.
        """

    @abstractmethod
    def fit(self, dataloader):
        """
        Fit the model to the training data.
        """

    @abstractmethod
    def predict(self, dataloader):
        """
        Predict using the fitted model.
        """


class TorchAbstractModel(ABC):
    """
    Abstract base class for all models in the diff_benchmark framework.
    Defines the interface that all models must implement.
    """

    @abstractmethod
    def _dataloader_to_numpy(self, dataloader):
        """
        Convert a DataLoader to numpy arrays.
        This method should be implemented by all subclasses to handle the conversion.
        """

    @abstractmethod
    def fit(self, dataloader):
        """
        Fit the model to the training data.
        """

    @abstractmethod
    def predict(self, dataloader):
        """
        Predict using the fitted model.
        """

class TorchPipeline:
    """
    Abstract base class for Torch models that require training.
    Extends TorchAbstractModel to include training-specific methods.
    """

    def __init__(self, num_workers=10, device=None, dtype=None, **kwargs):

        self.num_workers = num_workers
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        self.model = self._build_model(**kwargs).to(self.device)

        self.fold_idx = kwargs.get("fold_idx", -1)
        self.run_id = kwargs.get("run_id", "unnamed_run")
        self.epochs = kwargs.get("epochs", 100)
        self.average = kwargs.get("average", "binary")

        self.learning_rate = kwargs.get("learning_rate", 1e-4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

    @abstractmethod
    def _build_model(self, **kwargs):
        raise NotImplementedError("_build_model must be implemented and return a torch model.")

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
        
    def _train_val_loader_split(self, train_loader, val_ratio=0.3):
        
        dataset = train_loader.dataset
        n = len(dataset)
        genders = np.asarray(dataset.dataset.gender[dataset.indices])

        indices = np.arange(n)
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, stratify=genders, random_state=42
        )

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader_new = DataLoader(
            train_subset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_with_augmentation(
                batch, transform=train_transforms
            ),
        )
        val_loader_new = DataLoader(
            val_subset,
            batch_size=128,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_with_augmentation(
                batch, transform=val_transforms
            ),
        )
        return train_loader_new, val_loader_new
    
    def fit(self, dataloader):

        print(f"Device: {self.device}")
        self.model.train()

        train_loader, val_loader = self._train_val_loader_split(dataloader)
        print(f"Fold index: {self.fold_idx}")

        self.logger = TrainLogger(
            fold_idx=self.fold_idx,
            run_id=self.run_id,
            save_dir="./data/results/logger",
            monitor="val_accuracy",
            mode="max",
        )

        scaler = GradScaler()
        
        prof =  profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=5, repeat=1),
            on_trace_ready=tensorboard_trace_handler('./profiler/test_cnn'),
            profile_memory=True,
            record_shapes=False, 
            with_stack=False,
            with_flops=False
        )

        with prof:

            print("Dataloaders created")
            for epoch in tqdm(range(self.epochs)):

                print(f"Epoch {epoch}")
                for batch_train_idx, (xb, yb, _) in enumerate(train_loader):

                    # print("Batch loaded")
                    xb, yb = xb.to(self.device, non_blocking=True), yb.long().to(self.device, non_blocking=True)

                    # print("Moved to device")
                    with record_function("optimizer zero grad"): self.optimizer.zero_grad()
                    with record_function("inference"): preds = self.model(xb)
                    with record_function("loss function"): loss = self.criterion(preds, yb)

                    with record_function("backward"): loss.backward()
                    
                    # print("Forward + Bakcward done")
                    self.optimizer.step()
                    # self.optimizer.zero_grad()
                    # with autocast():
                    #     preds = self.model(xb)
                    #     loss = self.criterion(preds, yb)
                    # scaler.scale(loss).backward()
                    # scaler.step(self.optimizer)
                    # scaler.update()
                    

                    # y_true = yb.cpu().detach().numpy()
                    # y_pred = preds.argmax(dim=1).cpu().detach().numpy()

                    # metrics = {
                    #     "accuracy": accuracy_score(y_true, y_pred),
                    #     "precision": precision_score(
                    #         y_true, y_pred, average=self.average, zero_division="warn"
                    #     ),
                    #     "recall": recall_score(
                    #         y_true, y_pred, average=self.average, zero_division="warn"
                    #     ),
                    #     "f1": f1_score(
                    #         y_true, y_pred, average=self.average, zero_division="warn"
                    #     ),
                    #     # "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
                    # }
                    # self.history["train"]["epoch"].append(epoch)
                    # self.history["train"]["batch"].append(batch_train_idx)
                    # self.history["train"]["loss"].append(loss.item())
                    # self.history["train"]["metrics"].append(metrics)

                    prof.step()

                    ############### VALIDATION ##############
                    # print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
                    if batch_train_idx % 10 == 0:
                        self.model.eval()
                        y_true = []
                        y_pred = []
                        val_loss = 0
                        with torch.no_grad():
                            for batch_val_idx, (xb, yb, _) in enumerate(val_loader):
                                print(f"Val: batch {batch_val_idx}")
                                xb, yb = xb.to(self.device, non_blocking=True), yb.long().to(self.device, non_blocking=True)

                                preds = self.model(xb)
                                loss = self.criterion(preds, yb)
                                val_loss += loss.item()

                                y_true.append(yb.cpu().numpy())
                                y_pred.append(preds.argmax(dim=1).cpu().numpy())

                        val_loss /= len(val_loader)

                        y_true = np.concatenate(y_true)
                        y_pred = np.concatenate(y_pred)

                        # metrics = {
                        #     "accuracy": accuracy_score(y_true, y_pred),
                        #     "precision": precision_score(
                        #         y_true, y_pred, average=self.average, zero_division="warn"
                        #     ),
                        #     "recall": recall_score(
                        #         y_true, y_pred, average=self.average, zero_division="warn"
                        #     ),
                        #     "f1": f1_score(
                        #         y_true, y_pred, average=self.average, zero_division="warn"
                        #     ),
                        #     # "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
                        # }
                        # self.history["val"]["epoch"].append(epoch)
                        # self.history["val"]["batch_train_idx"].append(batch_train_idx)
                        # self.history["val"]["loss"].append(val_loss)
                        # self.history["val"]["metrics"].append(metrics)

                        # # self.logger.save_checkpoint(self.model, epoch, metrics["accuracy"])
                        # self.logger.update_smooth_checkpoint(
                        #     self.model, epoch, metrics["accuracy"]
                        # )
                        self.model.train()
                self.scheduler.step(val_loss)

        self.logger.save_checkpoint(self.model, self.epochs, 0, is_last=True)
        self.logger.save_logs()
        # self._save_logs(
        #     self.history, f"./data/results/logs/{self.run_id}_training_log.json"
        # )
    
    def predict(self, dataloader):
        checkpoint_path = Path(self.logger.best_path)
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[INFO] Loaded checkpoint from {checkpoint_path}")

        self.model.eval()
        preds_all = []

        mean = 0.5
        std = 0.5

        with torch.no_grad():
            for xb, _, _ in dataloader:
                xb = (xb - mean) / std
                xb = xb.to(self.device)
                logits = self.model(xb)
                preds = torch.argmax(logits, dim=1)
                preds_all.append(preds.cpu())
        return torch.cat(preds_all).numpy()  


class LightningModel(pl.LightningModule, ABC):  # pylint: disable=too-many-ancestors
    """
    Abstract Lightning-based deep learning model class.
    Provides:
      - training/validation/test step loops
      - metric computation
      - early stopping and checkpointing support
    """

    def __init__(
        self,
        learning_rate=1e-4,
        weight_decay=1e-4,
        average="binary",
        scheduler_type="plateau",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.average = average
        self.scheduler_type = scheduler_type

        # Subclasses must define self.model and self.criterion
        self.model = None
        # self.criterion = None
        self.criterion = torch.nn.CrossEntropyLoss()

    @abstractmethod
    def build_model(self):
        """Define the network architecture and loss function."""

    @abstractmethod
    # def forward(self, x):
    def forward(self, *args, **kwargs):
        """Forward pass."""

    @abstractmethod
    def fit(self, dataloader):
        """Fit the model to the training data."""

    @abstractmethod
    def predict(self, dataloader):
        """Predict using the fitted model."""

    def compute_metrics(self, y_true, y_pred):
        """Compute classification metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average=self.average, zero_division="warn"
            ),
            "recall": recall_score(
                y_true, y_pred, average=self.average, zero_division="warn"
            ),
            "f1": f1_score(y_true, y_pred, average=self.average, zero_division="warn"),
        }

    def training_step(self, batch, batch_idx, *args, **kwargs):
        """Training step for a single batch."""
        _ = batch_idx
        x, y, _ = batch
        logits = self(x)
        y = y.long()
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        metrics = self.compute_metrics(y.cpu(), preds.cpu())

        self.log("train_loss", loss, prog_bar=True)
        self.log_dict({f"train_{k}": v for k, v in metrics.items()}, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        _ = batch_idx
        x, y, _ = batch
        logits = self(x)
        y = y.long()
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        metrics = self.compute_metrics(y.cpu(), preds.cpu())

        self.log("val_loss", loss, prog_bar=True)
        self.log_dict({f"val_{k}": v for k, v in metrics.items()}, prog_bar=True)
        return {"val_loss": loss, **metrics}

    def predict_step(self, batch, batch_idx, *args, dataloader_idx=0, **kwargs):
        # Unpack batch safely
        _, _ = batch_idx, dataloader_idx
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,  # When to call the scheduler
                },
            }
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.5
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        if self.scheduler_type == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
