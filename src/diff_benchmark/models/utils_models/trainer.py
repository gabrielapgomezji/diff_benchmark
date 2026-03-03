import copy
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from diff_benchmark.utils.logger import (
    LightningDebugLogger,
    LightningPrintLogger,
    TorchDebugLogger,
    TrainerLogRecord,
    setup_logger,
    tqdm_if_enabled,
)
from diff_benchmark.utils.scores import compute_metrics


def configure_cached_dataset_augmentation(dataset, mode: str = "random"):
    """
    Configure augmentation mode for cached feature datasets.

    Args:
        dataset: Dataset or Subset wrapping a CachedFeatureDataset
        mode: "random" for training (consistent per subject), "fixed" for val/test (no augmentation)

    This function handles both direct CachedFeatureDataset and Subset wrappers.
    """
    # Unwrap Subset to get the actual dataset
    actual_dataset = dataset
    if isinstance(dataset, Subset):
        if hasattr(dataset.dataset, "dataset"):
            actual_dataset = dataset.dataset.dataset
        else:
            actual_dataset = dataset.dataset

    # Check if it's a CachedFeatureDataset
    if hasattr(actual_dataset, "set_augmentation_indices"):
        actual_dataset.set_augmentation_indices(mode=mode)
        return True
    return False


train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)
val_transforms = transforms.Compose(
    [
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


class BaseTrainer(ABC):
    """
    Backend-agnostic trainer interface.

    main.py MUST depend only on this API.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.fold_idx: int | None = None  # add fold placeholder
        self.target_mean = (
            0.0  # default values; will be set properly during training if needed
        )
        self.target_std = 1.0

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

    def set_fold(self, fold_idx: int):
        """Set the current fold for logging/tracking purposes."""
        self.fold_idx = fold_idx


class SklearnModel(ABC, BaseEstimator):
    """Base class for sklearn-compatible models; implements ``fit`` and ``predict``."""

    data_type = "array"

    def __init__(self, **kwargs):
        super().__init__()
        self.model = self._build_model(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs) -> BaseEstimator:
        """Build and return the sklearn estimator.

        Args:
            **kwargs: Model-specific configuration.

        Returns:
            BaseEstimator: Configured estimator.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to training data.

        Args:
            X (np.ndarray): Feature matrix of shape ``(n_samples, n_features)``.
            y (np.ndarray): Targets of shape ``(n_samples,)``.

        Returns:
            self
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for *X*.

        Args:
            X (np.ndarray): Input features of shape ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)


class SklearnTrainer(BaseTrainer):
    """Sklearn backend trainer wrapping a :class:`SklearnModel`."""

    def __init__(self, model: SklearnModel, **kwargs):
        self.model = model
        self.output_dim = 1

    def _dataloader_to_numpy(
        self, dataloader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a DataLoader to concatenated NumPy arrays.

        Args:
            dataloader: Yields ``(x_batch, y_batch, _)`` tuples.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(features, targets)``.
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

    def set_fold(self, fold_idx: int):
        super().set_fold(fold_idx)

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


def split_loader(dataloader, collate_fn: Callable | None, val_ratio=0.2, seed=42):
    """Split a DataLoader into stratified train and validation subsets.

    For cached feature datasets, configures augmentation automatically:
    training uses random per-subject augmentation; validation uses
    non-transformed features only.

    Args:
        dataloader (DataLoader): Source DataLoader to split.
        collate_fn (Callable | None): Per-batch collate function; if provided,
            training batches receive ``train_transforms`` and validation batches
            receive ``val_transforms``.
        val_ratio (float): Fraction reserved for validation.
        seed (int): Random seed for the split.

    Returns:
        tuple[DataLoader, DataLoader]: ``(train_loader, val_loader)``.
    """
    print(f"Val ratio: {val_ratio}, seed: {seed}")
    dataset = dataloader.dataset
    genders = np.asarray(dataset.dataset.gender[dataset.indices])
    idx = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_ratio,
        stratify=genders,
        random_state=42,
    )

    # Create distinct improved copies for validation to allow independent augmentation state
    # This allows setting different augmentation modes for train vs val while sharing heavy data
    dataset_val = copy.copy(dataset)
    if hasattr(dataset, "dataset"):
        dataset_val.dataset = copy.copy(dataset.dataset)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset_val, val_idx)

    # Configure augmentation for cached datasets
    is_cached_train = configure_cached_dataset_augmentation(train_ds, mode="random")

    # aug_indices inside the CahedFeaturesDataset is set to random
    # Validation: no augmentation (use original features)
    is_cached_val = configure_cached_dataset_augmentation(val_ds, mode="fixed")

    # the same aug_indices from the same CahedFeaturesDataset is set to fixed
    # so train_ds will use the fixed mode as well

    if is_cached_train or is_cached_val:
        print("✓ Detected cached feature dataset")
        if is_cached_train:
            print("  - Training: using random augmentation per subject (consistent)")
        if is_cached_val:
            print("  - Validation: using non-transformed features only")

    train_loader = DataLoader(
        train_ds,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
        collate_fn=(
            dataloader.collate_fn
            if collate_fn is None
            else lambda batch: collate_fn(batch, transform=train_transforms)
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        collate_fn=(
            dataloader.collate_fn
            if collate_fn is None
            else lambda batch: collate_fn(batch, transform=val_transforms)
        ),
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
        prediction_task: str,
        epochs: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = "cuda",
        val_ratio: float = 0.2,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(model)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.epochs = epochs
        self.val_ratio = val_ratio
        self.seed = seed

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.95,  # multiply LR by 0.95 every epoch
        )

        self.criterion = (
            nn.CrossEntropyLoss()
            if prediction_task == "binary_classification"
            else nn.MSELoss()
        )
        self.prediction_task = prediction_task
        self.run_id = kwargs["run_id"] if "run_id" in kwargs else "default_run"

        self.log = setup_logger(__name__)
        self.logger = TorchDebugLogger(
            enabled=kwargs.get("debug", False),
            run_id=self.run_id,
            output_dir=f"exp_outputs/experiments/exp_{self.run_id}/debug/",
            prediction_task=self.prediction_task,
        )

    def set_fold(self, fold_idx: int):
        super().set_fold(fold_idx)

    def fit(self, dataloader):
        train_loader, val_loader = split_loader(
            dataloader,
            collate_fn=self.model.collate_fn,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )

        print(f"Learning rate: {self.lr}, weight decay: {self.weight_decay}")

        if self.prediction_task != "binary_classification":
            targets = []
            for batch in train_loader:
                _, y, *_ = batch
                targets.append(y)
            all_targets = torch.cat(targets).float()
            self.target_mean = all_targets.mean().item()
            self.target_std = all_targets.std().item() + 1e-8
            self.log.info(
                f"Whitening targets: mean={self.target_mean:.4f}, std={self.target_std:.4f}"
            )

        show_progress = not self.logger.enabled or self.logger.enabled

        self.log.info(f"Starting training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            loss_window = deque(maxlen=20)  # smooth loss
            pbar = tqdm_if_enabled(
                enumerate(train_loader),
                desc=f"Epoch {epoch+1}/{self.epochs}",
                total=len(train_loader),
                enabled=show_progress,
            )
            self.log.info(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Epoch {epoch+1}/{self.epochs} of fold {self.fold_idx}")

            for batch_idx, batch in pbar:
                x, y, *_ = batch
                x = x.to(self.device, non_blocking=True)
                if self.prediction_task == "binary_classification":
                    y = y.long().to(self.device, non_blocking=True)
                else:
                    y = y.float().to(self.device, non_blocking=True)
                    y = (y - self.target_mean) / self.target_std

                self.optimizer.zero_grad()
                preds = self.model(x)
                if self.prediction_task == "binary_classification":
                    preds = preds
                else:
                    preds = preds.squeeze(1)
                loss = self.criterion(preds, y)
                # Here you compute the loss on the normalized targets, so gradient will be smaller.
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                loss_window.append(loss)

                pbar.set_postfix(
                    loss=f"{sum(loss_window)/len(loss_window):.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )
                self.logger.log_batch(
                    split="train",
                    epoch=epoch,
                    batch=batch_idx,
                    loss=loss.item(),
                    fold=self.fold_idx if self.fold_idx is not None else -1,
                )
                if self.logger.enabled:
                    train_preds_unnorm = preds.detach().cpu()
                    train_targets_unnorm = y.detach().cpu()
                    if self.prediction_task != "binary_classification":
                        train_preds_unnorm = (
                            train_preds_unnorm * self.target_std + self.target_mean
                        )
                        train_targets_unnorm = (
                            train_targets_unnorm * self.target_std + self.target_mean
                        )
                    train_preds.append(train_preds_unnorm)
                    train_targets.append(train_targets_unnorm)

            metrics = None
            if self.logger.enabled:
                preds = torch.cat(train_preds)
                preds = self.logger.finalize_preds(preds, self.prediction_task)
                metrics = compute_metrics(
                    y_true=torch.cat(train_targets).numpy(),
                    y_pred=preds.numpy(),
                    prediction_task=self.prediction_task,
                )

            self.logger.log_epoch(
                split="train",
                epoch=epoch,
                loss=train_loss / len(train_loader),
                metrics=metrics,
                fold=self.fold_idx if self.fold_idx is not None else -1,
            )
            val_loss = self._validate(val_loader, epoch)
            self.scheduler.step()
            self.log.info(
                f"[{self.run_id}] "
                f"Epoch {epoch+1}/{self.epochs} | "
                f"train_loss={train_loss / len(train_loader):.4f} | "
                f"val_loss={val_loss:.4f} | "
            )

        self.logger.flush(trainer=self)

    def _validate(self, val_loader, epoch):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x, y, *_ = batch
                x = x.to(self.device, non_blocking=True)
                if self.prediction_task == "binary_classification":
                    y = y.long().to(self.device, non_blocking=True)
                else:
                    y = y.float().to(self.device, non_blocking=True)
                    y = (y - self.target_mean) / self.target_std

                preds = self.model(x)
                if self.prediction_task == "binary_classification":
                    preds = preds
                else:
                    preds = preds.squeeze(1)
                loss = self.criterion(preds, y)
                val_loss += loss.item()

                if self.logger.enabled:
                    train_preds_unnorm = preds.cpu()
                    train_targets_unnorm = y.cpu()
                    if self.prediction_task != "binary_classification":
                        train_preds_unnorm = (
                            train_preds_unnorm * self.target_std + self.target_mean
                        )
                        train_targets_unnorm = (
                            train_targets_unnorm * self.target_std + self.target_mean
                        )
                    all_preds.append(train_preds_unnorm)
                    all_targets.append(train_targets_unnorm)

        val_loss /= len(val_loader)
        metrics = None
        if self.logger.enabled:
            preds = torch.cat(all_preds)
            preds = self.logger.finalize_preds(preds, self.prediction_task)
            metrics = compute_metrics(
                y_true=torch.cat(all_targets).numpy(),
                y_pred=preds.numpy(),
                prediction_task=self.prediction_task,
            )
        print(
            f"Validation accuracy: {metrics['accuracy']:.4f}"
            if metrics and "accuracy" in metrics
            else (
                f"Validation mae: {metrics['mae']:.4f}"
                if metrics and "mae" in metrics
                else ""
            )
        )
        self.logger.log_epoch(
            split="val",
            epoch=epoch,
            loss=val_loss,
            metrics=metrics,
            fold=self.fold_idx if self.fold_idx is not None else -1,
        )
        return val_loss

    def predict(self, dataloader):
        self.model.eval()
        outputs = []

        collate_fn = self.model.collate_fn

        # Configure cached dataset to use non-transformed features for prediction
        configure_cached_dataset_augmentation(dataloader.dataset, mode="fixed")

        predict_dataloader = DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            collate_fn=(
                dataloader.collate_fn
                if collate_fn is None
                else lambda batch: collate_fn(batch, transform=val_transforms)
            ),
            pin_memory=False,
        )

        with torch.no_grad():
            for batch in predict_dataloader:
                x, *_ = batch

                x = x.to(self.device)
                preds = self.model(x)
                if self.prediction_task == "binary_classification":
                    preds = preds.argmax(dim=1)
                else:
                    preds = preds.squeeze(1)
                    preds = preds * self.target_std + self.target_mean
                outputs.append(preds.cpu().detach())

        return torch.cat(outputs).numpy()

    def load(self, path: str):
        """Load model weights from a checkpoint file.

        Args:
            path (str): Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the checkpoint is incompatible with the current architecture.
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

        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))

        self.criterion = (
            nn.CrossEntropyLoss()
            if prediction_task == "binary_classification"
            else nn.MSELoss()
        )
        self.prediction_task = prediction_task

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y, *_ = batch
        preds = self(x)
        if self.prediction_task == "binary_classification":
            y = y.long()
        else:
            y = y.float()
            y = (y - self.target_mean) / self.target_std
            preds = preds.squeeze(1)

        loss = self.criterion(preds, y)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=False,
        )
        return {"loss": loss}

    def validation_step(self, batch, _):
        x, y, *_ = batch
        preds = self(x)
        if self.prediction_task == "binary_classification":
            y = y.long()
        else:
            y = y.float()
            y = (y - self.target_mean) / self.target_std
            preds = preds.squeeze(1)

        loss = self.criterion(preds, y)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=False,
        )
        return {"loss": loss}

    def predict_step(self, batch, _, __=None):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        preds = self(x)
        if self.prediction_task == "binary_classification":
            preds = preds.argmax(dim=1)
        else:
            preds = preds.squeeze(1)
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


def x_only_loader(dl: DataLoader):
    """Yield only input tensors from a ``(x, y, g)`` dataloader.

    Args:
        dl (DataLoader): Original dataloader yielding ``(x, y, g)`` tuples.

    Yields:
        tuple: Single-element tuple ``(x,)``.
    """
    for x, _, _ in dl:
        if isinstance(x, list):
            x = torch.stack(x)
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
        prediction_task,
        trainer_kwargs: dict,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        val_ratio: float = 0.2,
        seed: int = 42,
        **kwargs: Any,
    ):
        self.lightning_model = _LightningModuleAdapter(
            model=model,
            prediction_task=prediction_task,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        super().__init__(model)
        self.model = model
        self.run_id = kwargs.get("run_id", "default_run")
        print_cb = LightningPrintLogger(
            run_id=self.run_id,
            epochs=trainer_kwargs.get("max_epochs"),
        )
      
        trainer_kwargs = dict(trainer_kwargs)
        trainer_kwargs.setdefault("callbacks", []).append(print_cb)

        debug = kwargs.get("debug", False)
        debug_dir = f"exp_outputs/experiments/exp_{self.run_id}/debug/"
        if debug:
            self.run_id = kwargs["run_id"] if "run_id" in kwargs else "default_run"
            debug_cb = LightningDebugLogger(
                run_id=self.run_id,
                prediction_task=prediction_task,
                debug_dir=debug_dir,
                enabled=True,
            )
            trainer_kwargs = dict(trainer_kwargs)
            trainer_kwargs.setdefault("callbacks", []).append(debug_cb)

        self.trainer = pl.Trainer(**trainer_kwargs)
        self.val_ratio = val_ratio
        self.seed = seed

    def set_fold(self, fold_idx: int):
        super().set_fold(fold_idx)

        self.trainer.fold_idx = fold_idx

    def fit(self, dataloader):
        train_loader, val_loader = split_loader(
            dataloader, val_ratio=self.val_ratio, seed=self.seed
        )

        if self.lightning_model.prediction_task != "binary_classification":
            targets = []
            for batch in train_loader:
                _, y, *_ = batch
                targets.append(y)
            all_targets = torch.cat(targets).float()

            self.lightning_model.target_mean.fill_(all_targets.mean())
            self.lightning_model.target_std.fill_(all_targets.std() + 1e-8)
            print(
                f"Whitening targets: mean={self.lightning_model.target_mean.item():.4f}, std={self.lightning_model.target_std.item():.4f}"
            )

        self.trainer.fit(self.lightning_model, train_loader, val_loader)

    def predict(self, dataloader):
        configure_cached_dataset_augmentation(dataloader.dataset, mode="fixed")

        preds = self.trainer.predict(
            dataloaders=x_only_loader(dataloader), model=self.lightning_model
        )
        preds = torch.cat([p.cpu() for p in preds])

        if self.lightning_model.prediction_task != "binary_classification":
            preds = (
                preds * self.lightning_model.target_std.cpu()
                + self.lightning_model.target_mean.cpu()
            )

        return preds.numpy()
