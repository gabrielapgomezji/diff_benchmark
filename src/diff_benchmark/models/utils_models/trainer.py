from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.base import BaseEstimator
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


class BaseTrainer(ABC):
    """
    Backend-agnostic trainer interface.

    main.py MUST depend only on this API.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @property
    def data_type(self):
        return getattr(self.model, "data_type", None)

    @abstractmethod
    def fit(self, dataloader):
        """Train the model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataloader):
        """Run inference and return predictions."""
        raise NotImplementedError


class SklearnModel(ABC, BaseEstimator):
    """
    Base class for sklearn models.
    Implements fit and predict methods.
    """

    data_type = "array"

    def __init__(self, **kwargs):
        super().__init__()
        self.model = self._build_model(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs) -> BaseEstimator:
        """
        Build and instantiate the machine learning model.
        This is an abstract method that must be implemented by subclasses to define
        the specific model architecture and configuration.
        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for model configuration and customization.
        Returns
        -------
        BaseEstimator
            The instantiated and configured model object that follows the scikit-learn
            estimator interface.
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """

        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to the training data.
        Parameters
        ----------
        X : np.ndarray
            Training feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target variable of shape (n_samples,).
        Returns
        -------
        self
            Returns the fitted trainer instance for method chaining.
        """

        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data using the trained model.
        Parameters
        ----------
        X : np.ndarray
            Input features for prediction. Shape should be (n_samples, n_features).
        Returns
        -------
        np.ndarray
            Predicted values from the model. Shape depends on the model type and task
            (e.g., (n_samples,) for regression/binary classification,
            (n_samples, n_classes) for multi-class classification).
        """

        return self.model.predict(X)


class SklearnTrainer(BaseTrainer):
    """
    Abstract base class for all models in the diff_benchmark framework.
    Defines the interface that all models must implement.
    """

    def __init__(self, model: SklearnModel, **kwargs):
        self.model = model
        self.output_dim = 1

    def _dataloader_to_numpy(
        self, dataloader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts a dataloader containing batches of data into NumPy arrays.
        Args:
            dataloader (iterable): An iterable that yields batches of data in the form
                                   (x_batch, y_batch, _), where x_batch and y_batch
                                   are the input and target tensors, respectively.
        Returns:
            tuple: A tuple containing two NumPy arrays:
                - features (np.ndarray): Concatenated array of input data.
                - targets (np.ndarray): Concatenated array of target data.
        """

        features_list = []
        targets_list = []
        for features_batch, targets_batch, _ in dataloader:
            features_list.append(features_batch.numpy())
            targets_list.append(targets_batch.numpy())
        features = np.concatenate(features_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        return features, targets

    def _reshape_data(self, dataloader: DataLoader):
        features, targets = self._dataloader_to_numpy(dataloader)
        features_reshaped = features.reshape(features.shape[0], -1)
        return features_reshaped, targets.flatten()

    def fit(self, dataloader: DataLoader):
        """
        Fit the model to the training data.
        Args:
            dataloader (DataLoader): PyTorch DataLoader with training data.
        """
        features, targets = self._reshape_data(dataloader)
        self.model.fit(features, targets)

    def predict(self, dataloader: DataLoader):
        """
        Predict using the fitted model.
        Args:
            dataloader (DataLoader): PyTorch DataLoader with data to predict.
        """
        features, _ = self._reshape_data(dataloader)
        preds = self.model.predict(features)
        if self.output_dim == 2:
            return preds.reshape(-1, 1)
        return preds


def split_loader(dataloader, val_ratio=0.2, seed=42):
    """
    Split a PyTorch DataLoader into training and validation subsets.
    This function takes an existing DataLoader and splits its underlying dataset
    into training and validation sets based on a specified ratio. It creates new
    DataLoaders with the same configuration as the input DataLoader while preserving
    reproducibility through a fixed random seed.
    Args:
        dataloader (DataLoader): The original PyTorch DataLoader to split.
        val_ratio (float, optional): Fraction of the dataset to use for validation.
            Must be between 0 and 1. Defaults to 0.2 (20%).
        seed (int, optional): Random seed for reproducibility of the split.
            Defaults to 42.
    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training subset with
              shuffling enabled.
            - val_loader (DataLoader): DataLoader for the validation subset with
              shuffling disabled.
    Example:
        >>> train_loader, val_loader = split_loader(dataloader, val_ratio=0.2)
        >>> for batch in train_loader:
        ...     # process training batch
        ...     pass
    """

    dataset = dataloader.dataset
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


class TorchTrainer(BaseTrainer):
    """
    Pure PyTorch training backend.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        prediction_task: str = "classification",
        device: str = "cuda",
        val_ratio: float = 0.2,
        **kwargs: Any,
    ):
        super().__init__(model)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.epochs = epochs
        self.val_ratio = val_ratio

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.criterion = (
            nn.CrossEntropyLoss()
            if prediction_task == "classification"
            else nn.MSELoss()
        )
        self.prediction_task = prediction_task

    def fit(self, dataloader):
        train_loader, val_loader = split_loader(dataloader, val_ratio=self.val_ratio)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{self.epochs}]"):
                x, y, *_ = batch
                x = x.to(self.device, non_blocking=True)
                if self.prediction_task == "classification":
                    y = y.long().to(self.device, non_blocking=True)
                else:
                    y = y.float().to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                preds = self.model(x)
                if self.prediction_task == "classification":
                    preds = preds
                else:
                    preds = preds.squeeze(1)
                loss = self.criterion(preds, y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            self._validate(val_loader)

    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x, y, *_ = batch
                x = x.to(self.device, non_blocking=True)
                if self.prediction_task == "classification":
                    y = y.long().to(self.device, non_blocking=True)
                else:
                    y = y.float().to(self.device, non_blocking=True)

                preds = self.model(x)
                if self.prediction_task == "classification":
                    preds = preds
                else:
                    preds = preds.squeeze(1)
                loss = self.criterion(preds, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

    def predict(self, dataloader):
        self.model.eval()
        outputs = []

        mean = self.model.mean
        std = self.model.std
        with torch.no_grad():
            for batch in dataloader:
                x, *_ = batch
                x = (x - mean) / std
                x = x.to(self.device)
                preds = self.model(x)
                if self.prediction_task == "classification":
                    preds = preds.argmax(dim=1)
                else:
                    preds = preds.squeeze(1)
                outputs.append(preds.cpu().detach())

        return torch.cat(outputs).numpy()

    def save(self, path: str):
        """
        Save the model's state dictionary to a file.
        Args:
            path (str): The file path where the model state dictionary will be saved.
        """

        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """
        Load model weights from a checkpoint file.
        Args:
            path (str): Path to the checkpoint file containing the model state dictionary.
        Returns:
            None
        Raises:
            FileNotFoundError: If the checkpoint file does not exist at the specified path.
            RuntimeError: If the checkpoint is incompatible with the current model architecture.
        """

        self.model.load_state_dict(torch.load(path, map_location=self.device))


class _LightningModuleAdapter(pl.LightningModule):
    """
    Thin Lightning wrapper around a pure nn.Module.
    """

    def __init__(
        self,
        model: nn.Module,
        prediction_task: str,
        learning_rate: float,
        weight_decay: float,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.criterion = (
            nn.CrossEntropyLoss()
            if prediction_task == "classification"
            else nn.MSELoss()
        )
        self.prediction_task = prediction_task

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y, *_ = batch
        preds = self(x)
        if self.prediction_task == "classification":
            y = y.long()
        else:
            y = y.float()
        loss = self.criterion(preds, y)
        # if self.prediction_task == "classification":
        #     preds = preds.argmax(dim=1)
        # else:
        #     preds = preds.squeeze(1)
        # metrics
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y, *_ = batch
        preds = self(x)
        if self.prediction_task == "classification":
            y = y.long()
        else:
            y = y.float()
        loss = self.criterion(preds, y)
        # if self.prediction_task == "classification":
        #     preds = preds.argmax(dim=1)
        # else:
        #     preds = preds.squeeze(1)
        # metrics
        self.log("val_loss", loss, prog_bar=True)

    def predict_step(self, batch, _, __=None):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        preds = self(x)
        if self.prediction_task == "classification":
            preds = preds.argmax(dim=1)
        else:
            preds = preds.squeeze(1)
        return preds

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


def x_only_loader(dl: DataLoader):
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


class LightningTrainer(BaseTrainer):
    """
    PyTorch Lightning backend.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        trainer_kwargs: dict,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        prediction_task: str = "classification",
        val_ratio: float = 0.2,
        **kwargs: Any,
    ):
        lightning_model = _LightningModuleAdapter(
            model=model,
            prediction_task=prediction_task,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        super().__init__(lightning_model)

        self.trainer = pl.Trainer(**trainer_kwargs)
        self.val_ratio = val_ratio

    def fit(self, dataloader):
        train_loader, val_loader = split_loader(dataloader, val_ratio=self.val_ratio)
        self.trainer.fit(self.model, train_loader, val_loader)

    def predict(self, dataloader):
        # preds = self.trainer.predict(self.model, dataloader)
        preds = self.trainer.predict(dataloaders=x_only_loader(dataloader))
        preds = torch.cat([p.cpu() for p in preds])
        return preds.numpy()

    # def save(self, path: str):
    #     torch.save(self.model.model.state_dict(), path)

    # def load(self, path: str):
    #     self.model.model.load_state_dict(torch.load(path))
