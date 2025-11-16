from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
from sklearn.metrics import (  # confusion_matrix,; roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
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
