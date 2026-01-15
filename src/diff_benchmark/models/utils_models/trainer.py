from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import pytorch_lightning as pl
import torch
from torchvision import transforms
from sklearn.base import BaseEstimator
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from diff_benchmark.utils.scores import compute_metrics
from diff_benchmark.utils.logger import TrainerLogRecord, TorchDebugLogger, LightningDebugLogger, tqdm_if_enabled, LightningPrintLogger
from collections import deque
from diff_benchmark.utils.logger import setup_logger


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


def split_loader(dataloader, collate_fn: Callable | None, val_ratio=0.2, seed=42):
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
    """
    dataset = dataloader.dataset
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator)
    # genders = np.asarray(dataset.dataset.gender[dataset.indices])
    # idx = np.arange(len(dataset))
    # train_ds, val_ds = train_test_split(
    #     idx,
    #     test_size=val_ratio,
    #     stratify=genders,
    #     random_state=42,
    # )

    train_loader = DataLoader(
        train_ds,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn if collate_fn is None else lambda batch: collate_fn(batch, transform=train_transforms),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn if collate_fn is None else lambda batch: collate_fn(batch, transform=val_transforms),
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
        prediction_task,
        epochs: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = "cuda",
        val_ratio: float = 0.2,
        **kwargs: Any,
    ):
        super().__init__(model)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.epochs = epochs
        self.val_ratio = val_ratio

        # self.optimizer = torch.optim.AdamW(
        #     self.model.parameters(),
        #     lr=learning_rate,
        #     weight_decay=weight_decay,
        # )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,   # default to 1e-5
            weight_decay=weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
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
            output_dir="./data/results/parquet/debug/",
            prediction_task=self.prediction_task,
        )

    def fit(self, dataloader):
        train_loader, val_loader = split_loader(dataloader, collate_fn=self.model.collate_fn, val_ratio=self.val_ratio)
        show_progress = not self.logger.enabled or self.logger.enabled 
        
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

            # for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch+1}/{self.epochs}]")):
            for batch_idx, batch in pbar:
                x, y, *_ = batch
                x = x.to(self.device, non_blocking=True)
                if self.prediction_task == "binary_classification":
                    y = y.long().to(self.device, non_blocking=True)
                else:
                    y = y.float().to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                preds = self.model(x)
                if self.prediction_task == "binary_classification":
                    preds = preds
                else:
                    preds = preds.squeeze(1)
                loss = self.criterion(preds, y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                loss_window.append(loss)
                # if pbar is not train_loader:
                #     pbar.set_postfix(
                #         loss=f"{sum(loss_window)/len(loss_window):.4f}",
                #         lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                #     )
                pbar.set_postfix(
                    loss=f"{sum(loss_window)/len(loss_window):.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )
                self.logger.log_batch(
                    split="train",
                    epoch=epoch,
                    batch=batch_idx,
                    loss=loss.item(),
                )
                if self.logger.enabled:
                    train_preds.append(preds.detach().cpu())
                    train_targets.append(y.detach().cpu())
                

                # breakpoint()
                # self.model.train()
    
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
            )
            val_loss = self._validate(val_loader, epoch)
            self.scheduler.step(val_loss)
            self.log.info(
                f"[{self.run_id}] "
                f"Epoch {epoch+1}/{self.epochs} | "
                f"train_loss={train_loss / len(train_loader):.4f} | "
                f"val_loss={val_loss:.4f}"
            )
        
        self.logger.flush()

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

                preds = self.model(x)
                if self.prediction_task == "binary_classification":
                    preds = preds
                else:
                    preds = preds.squeeze(1)
                loss = self.criterion(preds, y)
                val_loss += loss.item()
                
                if self.logger.enabled:
                    all_preds.append(preds.cpu())
                    all_targets.append(y.cpu())

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
        self.logger.log_epoch(
            split="val",
            epoch=epoch,
            loss=val_loss,
            metrics=metrics,
        )
        return val_loss

    def predict(self, dataloader):
        self.model.eval()
        outputs = []

        collate_fn = self.model.collate_fn

        predict_dataloader = DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn if collate_fn is None else lambda batch: collate_fn(batch, transform=val_transforms),
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
                outputs.append(preds.cpu().detach())

        return torch.cat(outputs).numpy()

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
        loss = self.criterion(preds, y)
        # if self.prediction_task == "classification":
        #     preds = preds.argmax(dim=1)
        # else:
        #     preds = preds.squeeze(1)
        # metrics
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=False,  # IMPORTANT: we print ourselves
        )
        return {"loss": loss}

    def validation_step(self, batch, _):
        x, y, *_ = batch
        preds = self(x)
        if self.prediction_task == "binary_classification":
            y = y.long()
        else:
            y = y.float()
        loss = self.criterion(preds, y)
        # if self.prediction_task == "classification":
        #     preds = preds.argmax(dim=1)
        # else:
        #     preds = preds.squeeze(1)
        # metrics
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
        # return torch.optim.AdamW(
        #     self.parameters(),
        #     lr=self.learning_rate,
        #     weight_decay=self.weight_decay,
        # )
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,   # default to 1e-5
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
        prediction_task, 
        trainer_kwargs: dict,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        val_ratio: float = 0.2,
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
        # trainer_kwargs.setdefault("callbacks", []).append(print_cb)
        trainer_kwargs = dict(trainer_kwargs)
        trainer_kwargs.setdefault("callbacks", []).append(print_cb)
        
        debug = kwargs.get("debug", False)
        debug_dir = "./data/results/parquet/debug/"
        if debug:
            self.run_id = kwargs["run_id"] if "run_id" in kwargs else "default_run"
            debug_cb = LightningDebugLogger(
                run_id=self.run_id,
                prediction_task=prediction_task,
                debug_dir=debug_dir,
                enabled=True,
            )
            # trainer_kwargs.setdefault("callbacks", []).append(debug_cb)
            trainer_kwargs = dict(trainer_kwargs)
            trainer_kwargs.setdefault("callbacks", []).append(debug_cb)
        # trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in pl.Trainer.__init__.__code__.co_varnames} 

        self.trainer = pl.Trainer(**trainer_kwargs)
        self.val_ratio = val_ratio

    def fit(self, dataloader):
        train_loader, val_loader = split_loader(dataloader, val_ratio=self.val_ratio)
        # self.trainer.fit(self.model, train_loader, val_loader)
        self.trainer.fit(self.lightning_model, train_loader, val_loader)

    def predict(self, dataloader):
        # preds = self.trainer.predict(self.model, dataloader)
        preds = self.trainer.predict(dataloaders=x_only_loader(dataloader), model=self.lightning_model)
        preds = torch.cat([p.cpu() for p in preds])
        return preds.numpy()
