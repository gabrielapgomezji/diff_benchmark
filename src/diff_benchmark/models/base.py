import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import (  # confusion_matrix,; roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from diff_benchmark.utils.logger import TrainLogger
from diff_benchmark.utils.scores import compute_metrics
from diff_benchmark.models.utils import create_trainer


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



class SklearnModel(ABC, BaseEstimator):
    """
    Base class for sklearn models.
    Implements fit and predict methods.
    """

    data_type = "array"
    
    def __init__(self):
        super().__init__()
        self.model = self._build_model()
    
    @abstractmethod
    def _build_model(self, **kwargs) -> BaseEstimator:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class NumpyAbstractModel(ABC):
    """
    Abstract base class for all models in the diff_benchmark framework.
    Defines the interface that all models must implement.
    """
    def __init__(self, model: SklearnModel):
        self.model = model
        self.output_dim = 1
    
    def _dataloader_to_numpy(self, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
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
        preds =  self.model.predict(features)
        if self.output_dim == 2:
            return preds.reshape(-1, 1)
        return preds


class TorchAbstractModel(ABC):
    """
    Abstract base class for all models in the diff_benchmark framework.
    Defines the interface that all models must implement.
    """

    @abstractmethod
    def fit(self, dataloader: DataLoader):
        """
        Fit the model to the training data.
        Args:
            dataloader (DataLoader): PyTorch DataLoader with training data.
        """

    @abstractmethod
    def predict(self, dataloader: DataLoader):
        """
        Predict using the fitted model.
        Args:
            dataloader (DataLoader): PyTorch DataLoader with data to predict.
        """


class TorchPipeline:
    """
    Abstract base class for Torch models that require training.
    Extends TorchAbstractModel to include training-specific methods.
    """

    def __init__(self, model: nn.Module, num_workers: int =10, device: torch.device =None, dtype: torch.dtype =None, **kwargs):

        self.num_workers = num_workers
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype if dtype is not None else torch.float32

        self.fold_idx = kwargs.get("fold_idx", -1)
        self.run_id = kwargs.get("run_id", "unnamed_run")
        self.epochs = kwargs.get("epochs", 100)
        self.average = kwargs.get("average", "binary")
        self._prediction_task = kwargs.get("prediction_task", None)

        self.model = model.to(self.device)

        self.learning_rate = kwargs.get("learning_rate", 1e-4)
        self.weight_decay = kwargs.get("weight_decay", 1e-2)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.prediction_task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif self.prediction_task == "regression":
            self.criterion = nn.MSELoss()

        self.max_lr = kwargs.get("max_lr", 1e-4)
        self.pct_start = kwargs.get("pct_start", 0.2)
        self.scheduler = None  # Defined later
        self.logger = None  # Defined later
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode="min", factor=0.5, patience=10
        # )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=0
        )
        self.history = {
            "train": {"epoch": [], "batch": [], "loss": [], "metrics": [], "lr": []},
            "val": {
                "epoch": [],
                "loss": [],
                "metrics": [],
                "batch_train_idx": [],
            },
        }

    @property
    def prediction_task(self):
        return self._prediction_task

    # @prediction_task.setter
    # def prediction_task(self, value):
    #     self._prediction_task = value

    def _save_logs(self, history: dict, save_path: str):
        """
        Save the provided history logs to a file in either JSON or CSV format.
        Args:
            history (dict): The history data to save. If saving as CSV, this should
                be a list of dictionaries where each dictionary represents a row.
            save_path (str): The file path where the logs will be saved. The file
                extension must be either '.json' or '.csv'.
        Raises:
            ValueError: If the save_path does not end with '.json' or '.csv'.
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

    def _train_val_loader_split(self, train_loader: DataLoader, val_ratio: float = 0.3) -> tuple[DataLoader, DataLoader]:
        """
        Splits a given training DataLoader into training and validation DataLoaders.
        This method takes a DataLoader containing the training dataset and splits it
        into two separate DataLoaders: one for training and one for validation. The
        split is performed based on the specified validation ratio, and the split
        is stratified by gender to ensure balanced representation in both subsets.
        Args:
            train_loader (DataLoader): The DataLoader containing the training dataset.
            val_ratio (float, optional): The ratio of the dataset to be used for validation.
                Defaults to 0.3.
        Returns:
            tuple[DataLoader, DataLoader]: A tuple containing the new training DataLoader
            and validation DataLoader.
        Notes:
            - The `train_loader` dataset is expected to have a `dataset.gender` attribute
              that provides gender information for stratified splitting.
            - The `self.model.collate_with_augmentation` function is used as the collate
              function for both training and validation DataLoaders.
            - The `train_transforms` and `val_transforms` are applied to the respective
              DataLoaders during data collation.
        """

        dataset = train_loader.dataset
        n = len(dataset)
        genders = np.asarray(dataset.dataset.gender[dataset.indices])

        indices = np.arange(n)
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, stratify=genders, random_state=42
        )

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        collate_train = self.model.collate_with_augmentation
        collate_val = self.model.collate_with_augmentation

        train_loader_new = DataLoader(
            train_subset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # collate_fn=lambda batch: collate_with_augmentation(
            #     batch, transform=train_transforms
            # ),
            collate_fn=lambda batch: collate_train(batch, transform=train_transforms),
        )
        val_loader_new = DataLoader(
            val_subset,
            batch_size=64,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=lambda batch: collate_with_augmentation(
            #     batch, transform=val_transforms
            # ),
            collate_fn=lambda batch: collate_val(batch, transform=val_transforms),
        )
        return train_loader_new, val_loader_new

    def fit(self, dataloader: DataLoader):
        """Fit the model to the training data.
        Args:
            dataloader (DataLoader): PyTorch DataLoader with training data.
        """
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

        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     max_lr=self.max_lr,
        #     epochs=self.epochs,
        #     steps_per_epoch=len(train_loader),
        #     anneal_strategy="cos",
        #     pct_start=self.pct_start,
        #     div_factor=3,  # 1.0e3, #10,
        #     final_div_factor=1.0e5,  # 1.0e4,
        # )

        print("Dataloaders created")
        for epoch in tqdm(range(self.epochs)):

            print(f"Epoch {epoch}")
            epoch_losses = []
            for batch_train_idx, (xb, yb, _) in enumerate(train_loader):

                # print("Batch loaded")
                # xb, yb = xb.to(self.device, non_blocking=True), yb.long().to(
                #     self.device, non_blocking=True
                # )
                # xb, yb = xb.to(self.device, non_blocking=True), yb.float().to(
                #     self.device, non_blocking=True
                # )
                xb = xb.to(self.device, non_blocking=True)

                if self.prediction_task == "classification":
                    yb = yb.long().to(self.device, non_blocking=True)
                else:
                    yb = yb.float().to(self.device, non_blocking=True)

                # print("Moved to device")
                self.optimizer.zero_grad()
                preds = self.model(xb)
                if self.prediction_task == "classification":
                    preds = preds  # .argmax(dim=1)
                else:
                    preds = preds.squeeze(1)

                loss = self.criterion(preds, yb)

                loss.backward()

                # print("Forward + Bakcward done")
                self.optimizer.step()

                epoch_losses.append(loss.item())
                y_true = yb.cpu().detach().numpy()
                # y_pred = preds.argmax(dim=1).cpu().detach().numpy()
                if self.prediction_task == "classification":
                    y_pred = preds.argmax(dim=1).cpu().detach().numpy()
                else:
                    y_pred = preds.cpu().detach().numpy()
                metrics = compute_metrics(y_true, y_pred, self.prediction_task)
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
                self.history["train"]["epoch"].append(epoch)
                self.history["train"]["batch"].append(batch_train_idx)
                self.history["train"]["loss"].append(loss.item())
                self.history["train"]["metrics"].append(metrics)

                # ############### VALIDATION ##############
                # print(f"Epoch {epoch+1}, Loss: {loss/len(train_loader):.4f}")
                if batch_train_idx % 10 == 0:
                    self.model.eval()
                    y_true = []
                    y_pred = []
                    val_loss = 0
                    with torch.no_grad():
                        for batch_val_idx, (xb, yb, _) in enumerate(val_loader):
                            print(f"Val: batch {batch_val_idx}")
                            # xb, yb = xb.to(
                            #     self.device, non_blocking=True
                            # ), yb.long().to(self.device, non_blocking=True)
                            xb = xb.to(self.device, non_blocking=True)

                            if self.prediction_task == "classification":
                                yb = yb.long().to(self.device, non_blocking=True)
                            else:
                                yb = yb.float().to(self.device, non_blocking=True)

                            preds = self.model(xb)
                            if self.prediction_task == "classification":
                                preds = preds  # .argmax(dim=1) # Remove the argmax for classification. Done in the cross entropy loss
                            else:
                                preds = preds.squeeze(1)
                            loss = self.criterion(preds, yb)
                            val_loss += loss.item()

                            y_true.append(yb.cpu().numpy())
                            # y_pred.append(preds.argmax(dim=1).cpu().numpy())
                            if self.prediction_task == "classification":
                                y_pred.append(
                                    preds.argmax(dim=1).cpu().detach().numpy()
                                )
                            else:
                                y_pred.append(preds.cpu().detach().numpy())

                    val_loss /= len(val_loader)

                    y_true = np.concatenate(y_true)
                    y_pred = np.concatenate(y_pred)
                    metrics = compute_metrics(y_true, y_pred, self.prediction_task)
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
                    self.history["val"]["epoch"].append(epoch)
                    self.history["val"]["batch_train_idx"].append(batch_train_idx)
                    self.history["val"]["loss"].append(val_loss)
                    self.history["val"]["metrics"].append(metrics)

                    # # self.logger.save_checkpoint(self.model, epoch, metrics["accuracy"])
                    # self.logger.update_smooth_checkpoint(
                    #     self.model, epoch, metrics["accuracy"]
                    # )
                    self.model.train()

                # self.scheduler.step()  # For one cycle scheduler
            self.scheduler.step()

            print(
                f"Epoch {epoch}: Training loss {np.mean(epoch_losses)} - Val Loss: {val_loss}"
            )

        plot_history_from_file(
            self.history, self.fold_idx, self.run_id, self.prediction_task
        )
        self.logger.save_checkpoint(self.model, self.epochs, 0, is_last=True)
        self.logger.save_logs()
        self._save_logs(
            self.history, f"./data/results/logs/{self.run_id}_training_log.json"
        )

    def predict(self, dataloader: DataLoader):
        """Prediction step using last model checkpoint.
        Args:
            dataloader (DataLoader): PyTorch DataLoader with data to predict.
        """
        checkpoint_path = Path(self.logger.best_path)
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
        else:
            print(
                f"Checkpoint {checkpoint_path} does not exist. Using current model weights."
            )

        self.model.eval()
        preds_all = []

        # mean = 0.5
        # std = 0.5
        mean = self.model.mean
        std = self.model.std

        with torch.no_grad():
            for xb, _, _ in dataloader:
                xb = (xb - mean) / std
                # xb = xb.unsqueeze(1) # REMOVE FOR THE 2DCNN
                xb = xb.to(self.device)
                logits = self.model(xb)
                # preds = torch.argmax(logits, dim=1)
                if self.prediction_task == "classification":
                    preds = logits.argmax(dim=1).cpu().detach()
                else:
                    preds = logits.squeeze(1).cpu().detach()
                preds_all.append(preds)  # .cpu())
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
        model: nn.Module,
        num_workers: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        average: str = "binary",
        scheduler_type: str = "plateau",
        optimizer_type: str = "adamw",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_workers = num_workers
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.average = average
        self.scheduler_type = scheduler_type
        self.optimizer_type = optimizer_type
        self.prediction_task = kwargs.get("prediction_task", None)
        
        
        # device str
        # run id
        # fold
        # epochs
        # freeze_backbone

        # Subclasses must define self.model and self.criterion
        self.model = model
        # self.criterion = None
        # self.criterion = torch.nn.CrossEntropyLoss()
        if self.prediction_task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif self.prediction_task == "regression":
            self.criterion = nn.MSELoss()


    def _train_val_loader_split(self, train_loader: DataLoader, val_ratio: float = 0.3) -> tuple[DataLoader, DataLoader]:
        """
        Splits a given DataLoader into training and validation DataLoaders based on a specified validation ratio.
        Args:
            train_loader (DataLoader): The DataLoader containing the dataset to be split.
            val_ratio (float, optional): The ratio of the dataset to be used for validation. Defaults to 0.3.
        Returns:
            tuple[DataLoader, DataLoader]: A tuple containing the new training DataLoader and validation DataLoader.
        Notes:
            - The split is stratified based on the 'gender' attribute of the dataset.
            - The new DataLoaders are created with specific batch sizes, shuffling, and optional transform-aware collation.
            - The random state for the split is fixed at 42 for reproducibility.
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
        collate_train = self.model.collate_with_augmentation
        collate_val = self.model.collate_with_augmentation

        train_loader_new = DataLoader(
            train_subset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # collate_fn=lambda batch: collate_with_augmentation(
            #     batch, transform=train_transforms
            # ),
            collate_fn=lambda batch: collate_train(batch, transform=train_transforms),
        )
        val_loader_new = DataLoader(
            val_subset,
            batch_size=64,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=lambda batch: collate_with_augmentation(
            #     batch, transform=val_transforms
            # ),
            collate_fn=lambda batch: collate_val(batch, transform=val_transforms),
        )
        # Transform-aware collation (optional)
        # train_loader_new = DataLoader(
        #     train_subset,
        #     batch_size=train_loader.batch_size,
        #     shuffle=True,
        #     num_workers=19,  # 0,#
        #     pin_memory=False,
        #     collate_fn=lambda batch: collate_with_augmentation(
        #         batch, transform=train_transforms
        #     ),
        # )
        # val_loader_new = DataLoader(
        #     val_subset,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=19,  # 0,#10,
        #     pin_memory=False,
        #     collate_fn=lambda batch: collate_with_augmentation(
        #         batch, transform=val_transforms
        #     ),
        # )
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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def fit(self, dataloader: DataLoader):
        """
        Lightning-based fit function to preserve compatibility with the old API.
        Splits the input dataloader into training and validation sets,
        sets up the trainer with early stopping and checkpointing,
        and runs Trainer.fit().
        Args:
            dataloader (DataLoader): The DataLoader containing the training data.
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
            dataloader (DataLoader): The DataLoader containing the data to predict on.
        Returns:
            np.ndarray: The predictions as a NumPy array.
        """
        dataset = dataloader.dataset
        
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=19,  # 0,#10,
            collate_fn=lambda batch: self.model.collate_with_augmentation(
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

    def compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        """Compute classification metrics.
        Args:
            y_true (torch.Tensor): True labels.
            y_pred (torch.Tensor): Predicted labels.
        Returns:
            dict: Dictionary with accuracy, precision, recall, and F1-score.
        """
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

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int, *args, **kwargs) -> torch.Tensor:
        """
        Performs a single training step for the model.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing the input tensor `x`, 
            the target tensor `y`, and an additional tensor (unused in this method).
            batch_idx (int): The index of the current batch.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed loss for the current batch.

        Notes:
            - The method computes the model's predictions (`logits`) using the input tensor `x`.
            - The target tensor `y` is converted to a long data type for compatibility with the loss function.
            - The loss is calculated using the specified criterion.
            - Predictions are obtained by taking the argmax of the logits along dimension 1.
            - Metrics are computed using the `compute_metrics` method and logged.
            - The training loss and metrics are logged using the `self.log` and `self.log_dict` methods.
        """
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

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int, *args, **kwargs) -> dict:
        """
        Performs a single validation step during the model evaluation phase.
        Args:
            batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing the input tensor `x`, 
                the target tensor `y`, and an additional tensor (unused in this method).
            batch_idx (int): The index of the current batch.
            *args: Additional positional arguments (not used in this method).
            **kwargs: Additional keyword arguments (not used in this method).
        Returns:
            dict: A dictionary containing the validation loss under the key "val_loss" and additional 
                computed metrics with keys prefixed by "val_".
        """
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

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int, *args, dataloader_idx: int = 0, **kwargs) -> torch.Tensor:
        """
        Perform a prediction step during inference.
        Args:
            batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A batch of data, typically containing input tensors 
                and possibly additional information. The first element of the batch is used as input to the model.
            batch_idx (int): The index of the current batch.
            *args: Additional positional arguments.
            dataloader_idx (int, optional): The index of the dataloader, useful when using multiple dataloaders. Defaults to 0.
            **kwargs: Additional keyword arguments.
        Returns:
            torch.Tensor: The predicted class indices for the input batch, obtained by applying `torch.argmax` 
                on the model's output logits along the class dimension.
        """
        
        # Unpack batch safely
        _, _ = batch_idx, dataloader_idx
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self) -> dict:
        """
        Configures the optimizer and learning rate scheduler for the model.
        Returns:
            dict: A dictionary containing the optimizer and learning rate scheduler configuration.
        The method supports the following optimizers:
            - AdamW: Adam optimizer with weight decay.
            - Adam: Standard Adam optimizer.
        The method supports the following learning rate schedulers:
            - "plateau": ReduceLROnPlateau scheduler, which reduces the learning rate when a monitored metric has stopped improving.
            - "step": StepLR scheduler, which decays the learning rate by a factor every fixed number of steps.
            - "exponential": ExponentialLR scheduler, which decays the learning rate exponentially.
            - "onecycle": OneCycleLR scheduler, which adjusts the learning rate cyclically over the course of training.
        Notes:
            - For the "plateau" scheduler, the learning rate is reduced based on the "val_loss" metric.
            - The "onecycle" scheduler requires the total number of training steps to be estimated using `self.trainer.estimated_stepping_batches`.
            - The "onecycle" scheduler operates on a per-step interval, while other schedulers typically operate on a per-epoch interval.
            - If an unsupported `scheduler_type` is provided, a default StepLR scheduler is used.
        """
        
        if self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
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
        if self.scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                anneal_strategy="cos",
                final_div_factor=1e4,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # OneCycleLR MUST be per-step
                },
            }

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


# def plot_history_from_file(path="history.json", save_path="training_history.pdf"):
def plot_history_from_file(history, fold_idx, run_id, prediction_task):
    """
    Plots training and validation curves (loss + accuracy) with epochs on the x-axis.
    Infers steps_per_epoch and validation interval automatically from history.
    """
    # os.makedirs(os.path.dirname('history.png'), exist_ok=True)

    # LOSS
    # --- Training steps/epochs ---
    steps_per_epoch = len(set(history["train"]["batch"]))  # ~19
    train_x = [
        e + b / steps_per_epoch
        for e, b in zip(history["train"]["epoch"], history["train"]["batch"])
    ]

    # --- Validation steps/epochs ---
    val_x = [
        e + i / steps_per_epoch
        for e, i in zip(history["val"]["epoch"], history["val"]["batch_train_idx"])
    ]

    epochs = sorted(set(history["train"]["epoch"]))

    # METRICS
    # --- Accuracy ---
    if prediction_task == "classification":
        train_acc = [m["accuracy"] for m in history["train"]["metrics"]]
        val_acc = [m["accuracy"] for m in history["val"]["metrics"]]

        train_prec = [m["precision"] for m in history["train"]["metrics"]]
        val_prec = [m["precision"] for m in history["val"]["metrics"]]
        train_rec = [m["recall"] for m in history["train"]["metrics"]]
        val_rec = [m["recall"] for m in history["val"]["metrics"]]

        train_f1 = [m["f1"] for m in history["train"]["metrics"]]
        val_f1 = [m["f1"] for m in history["val"]["metrics"]]

        train_epoch_acc = [
            np.mean(
                [acc for e, acc in zip(history["train"]["epoch"], train_acc) if e == ep]
            )
            for ep in epochs
        ]
        val_epoch_acc = [
            np.mean(
                [acc for e, acc in zip(history["val"]["epoch"], val_acc) if e == ep]
            )
            for ep in epochs
        ]

        train_epoch_prec = [
            np.mean(
                [
                    acc
                    for e, acc in zip(history["train"]["epoch"], train_prec)
                    if e == ep
                ]
            )
            for ep in epochs
        ]
        val_epoch_prec = [
            np.mean(
                [acc for e, acc in zip(history["val"]["epoch"], val_prec) if e == ep]
            )
            for ep in epochs
        ]
        train_epoch_rec = [
            np.mean(
                [acc for e, acc in zip(history["train"]["epoch"], train_rec) if e == ep]
            )
            for ep in epochs
        ]
        val_epoch_rec = [
            np.mean(
                [acc for e, acc in zip(history["val"]["epoch"], val_rec) if e == ep]
            )
            for ep in epochs
        ]

        train_epoch_f1 = [
            np.mean(
                [acc for e, acc in zip(history["train"]["epoch"], train_f1) if e == ep]
            )
            for ep in epochs
        ]
        val_epoch_f1 = [
            np.mean([acc for e, acc in zip(history["val"]["epoch"], val_f1) if e == ep])
            for ep in epochs
        ]
    elif prediction_task == "regression":
        train_mse = [m["mse"] for m in history["train"]["metrics"]]
        val_mse = [m["mse"] for m in history["val"]["metrics"]]

        train_epoch_mse = [
            np.mean(
                [mse for e, mse in zip(history["train"]["epoch"], train_mse) if e == ep]
            )
            for ep in epochs
        ]
        val_epoch_mse = [
            np.mean(
                [mse for e, mse in zip(history["val"]["epoch"], val_mse) if e == ep]
            )
            for ep in epochs
        ]
        train_r2 = [m["r2"] for m in history["train"]["metrics"]]
        val_r2 = [m["r2"] for m in history["val"]["metrics"]]
        train_epoch_r2 = [
            np.mean(
                [r2 for e, r2 in zip(history["train"]["epoch"], train_r2) if e == ep]
            )
            for ep in epochs
        ]
        val_epoch_r2 = [
            np.mean([r2 for e, r2 in zip(history["val"]["epoch"], val_r2) if e == ep])
            for ep in epochs
        ]
        train_explained_variance = [
            m["explained_variance"] for m in history["train"]["metrics"]
        ]
        val_explained_variance = [
            m["explained_variance"] for m in history["val"]["metrics"]
        ]
        train_epoch_explained_variance = [
            np.mean(
                [
                    ev
                    for e, ev in zip(
                        history["train"]["epoch"], train_explained_variance
                    )
                    if e == ep
                ]
            )
            for ep in epochs
        ]
        val_epoch_explained_variance = [
            np.mean(
                [
                    ev
                    for e, ev in zip(history["val"]["epoch"], val_explained_variance)
                    if e == ep
                ]
            )
            for ep in epochs
        ]
        train_mape = [m["mape"] for m in history["train"]["metrics"]]
        val_mape = [m["mape"] for m in history["val"]["metrics"]]
        train_epoch_mape = [
            np.mean(
                [
                    mape
                    for e, mape in zip(history["train"]["epoch"], train_mape)
                    if e == ep
                ]
            )
            for ep in epochs
        ]
        val_epoch_mape = [
            np.mean(
                [mape for e, mape in zip(history["val"]["epoch"], val_mape) if e == ep]
            )
            for ep in epochs
        ]

    # --- Create figure with 2 subplots ---
    _, axes = plt.subplots(2, 2, figsize=(20, 10))

    ax = axes[0, 0]
    ax.plot(
        train_x,
        history["train"]["loss"],
        "b-",
        alpha=0.7,
        linewidth=1.5,
        label="Training",
    )

    ax.plot(
        val_x,
        history["val"]["loss"],
        "r-",
        alpha=0.7,
        linewidth=2,
        label="Validation",
    )

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss\n({steps_per_epoch} steps/epoch, validation every 10 steps)")
    ax.legend()
    num_epochs = max(history["train"]["epoch"]) + 1
    ax.set_xlim(0, num_epochs)
    if prediction_task == "classification":
        ax = axes[0, 1]
        # ax.plot(
        #     train_x,
        #     train_acc,
        #     "b-",
        #     alpha=0.7,
        #     linewidth=1.5,
        #     label="Training",
        # )
        ax.plot(
            epochs,
            train_epoch_acc,
            "b-",
            markersize=5,
            markeredgewidth=1.5,
            label="Training",
        )

        # ax.plot(
        #     val_x,
        #     val_acc,
        #     "r-",
        #     alpha=0.7,
        #     linewidth=2,
        #     label="Validation",
        # )
        ax.plot(
            epochs,
            val_epoch_acc,
            "r-",
            markersize=5,
            markeredgewidth=1.5,
            label="Validation",
        )

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.set_title(
            f"Accuracy\n({steps_per_epoch} steps/epoch, validation every 10 steps)"
        )
        ax.legend()
        ax.set_xlim(0, num_epochs)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.grid(True)

        ax = axes[1, 0]
        ax.plot(
            epochs,
            train_epoch_prec,
            "b-",
            alpha=0.7,
            linewidth=1.5,
            label="Training precision",
        )
        ax.plot(
            epochs,
            train_epoch_rec,
            "b--",
            alpha=0.7,
            linewidth=1.5,
            label="Training recall",
        )

        ax.plot(
            epochs,
            val_epoch_prec,
            "r-",
            alpha=0.7,
            linewidth=2,
            label="Validation precision",
        )
        ax.plot(
            epochs,
            val_epoch_rec,
            "r--",
            alpha=0.7,
            linewidth=2,
            label="Validation recall",
        )

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Precision/Recall")
        ax.set_title(
            f"Precision/Recall\n({steps_per_epoch} steps/epoch, validation every 10 steps)"
        )
        ax.legend()
        ax.set_xlim(0, num_epochs)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.grid(True)

        ax = axes[1, 1]
        ax.plot(
            epochs,
            train_epoch_f1,
            "b-",
            alpha=0.7,
            linewidth=1.5,
            label="Training",
        )

        ax.plot(
            epochs,
            val_epoch_f1,
            "r-",
            alpha=0.7,
            linewidth=2,
            label="Validation",
        )

        ax.set_xlabel("Epochs")
        ax.set_ylabel("F1 score")
        ax.set_title(
            f"F1 score\n({steps_per_epoch} steps/epoch, validation every 10 steps)"
        )
        ax.legend()
        ax.set_xlim(0, num_epochs)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"history_{fold_idx}_{run_id}.png")
        plt.show()
    elif prediction_task == "regression":
        ax = axes[0, 1]
        ax.plot(
            epochs,
            train_epoch_mse,
            "b-",
            markersize=5,
            markeredgewidth=1.5,
            label="Training MSE",
        )

        ax.plot(
            epochs,
            val_epoch_mse,
            "b--",
            markersize=5,
            markeredgewidth=1.5,
            label="Validation MSE",
        )

        ax.plot(
            epochs,
            train_epoch_mape,
            "r-",
            alpha=0.7,
            linewidth=2,
            label="Training MAPE",
        )
        ax.plot(
            epochs,
            val_epoch_mape,
            "r--",
            alpha=0.7,
            linewidth=2,
            label="Validation MAPE",
        )

        ax.set_xlabel("Epochs")
        ax.set_ylabel("MSE and MAPE")
        ax.set_yscale("log")
        ax.set_title(
            f"MSE and MAPE\n({steps_per_epoch} steps/epoch, validation every 10 steps)"
        )
        ax.legend()
        ax.set_xlim(0, num_epochs)

        ax = axes[1, 0]
        ax.plot(
            epochs,
            train_epoch_r2,
            "b-",
            alpha=0.7,
            linewidth=1.5,
            label="Training",
        )
        ax.plot(
            epochs,
            val_epoch_r2,
            "r-",
            alpha=0.7,
            linewidth=2,
            label="Validation",
        )

        ax.set_xlabel("Epochs")
        ax.set_ylabel("R2 Score")
        ax.set_title(
            f"R2 Score\n({steps_per_epoch} steps/epoch, validation every 10 steps)"
        )
        ax.legend()
        ax.set_xlim(0, num_epochs)
        ax.set_ylim(-50, 5)

        ax = axes[1, 1]
        ax.plot(
            epochs,
            train_epoch_explained_variance,
            "b-",
            alpha=0.7,
            linewidth=1.5,
            label="Training",
        )
        ax.plot(
            epochs,
            val_epoch_explained_variance,
            "r-",
            alpha=0.7,
            linewidth=2,
            label="Validation",
        )
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Explained Variance")
        ax.set_title(
            f"Explained Variance\n({steps_per_epoch} steps/epoch, validation every 10 steps)"
        )
        ax.legend()
        ax.set_xlim(0, num_epochs)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"history_{fold_idx}_{run_id}.png")
        plt.show()
