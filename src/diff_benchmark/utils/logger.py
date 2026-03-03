import logging
import os
import sys
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm

LOG_FORMAT = "%(levelname)s - %(asctime)s - %(name)s - %(message)s"
DATE_FORMAT = "%H:%M:%S"


def tqdm_if_enabled(iterable, *, desc=None, total=None, enabled=True):
    """Wrap *iterable* in :func:`tqdm.tqdm`, suppressing output when not in a TTY.

    Args:
        iterable: The iterable to wrap.
        desc: Progress bar description label.
        total: Expected number of iterations.
        enabled: Set to ``False`` to always suppress the progress bar.

    Returns:
        A :class:`tqdm.tqdm` iterator.
    """
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=False,
        disable=not enabled or not sys.stdout.isatty(),
    )


@dataclass
class TrainerLogRecord:
    """A single epoch- or batch-level training record."""

    split: Literal["train", "val", "test"]
    epoch: int
    loss: float
    batch: Optional[int] = None
    metrics: Optional[dict[str, float]] = None
    fold: Optional[int] = None

    def to_dict(self) -> dict:
        """Flatten the record into a dict suitable for a Pandas DataFrame row.

        Returns:
            Dict with scalar values; metric entries are inlined at the top level.
        """
        base = {
            "split": self.split,
            "epoch": self.epoch,
            "loss": self.loss,
            "batch": self.batch,
            "fold": self.fold,
        }
        if self.metrics:
            base.update(self.metrics)
        return base


class TorchDebugLogger:
    """Records batch- and epoch-level training statistics to a Parquet file.

    Designed for use with plain PyTorch training loops (not Lightning).

    Args:
        enabled: When ``False``, all logging calls are no-ops.
        run_id: Unique run identifier used to name output files.
        output_dir: Directory where debug Parquet files are written.
        prediction_task: ``"binary_classification"`` or ``"regression"``.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        run_id: str,
        output_dir: str,
        prediction_task: str,
    ):
        self.enabled = enabled
        self.run_id = run_id
        self.output_dir = output_dir
        self.prediction_task = prediction_task
        self.records: list[dict] = []

        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Logging methods                                                      #
    # ------------------------------------------------------------------ #

    def log_batch(
        self,
        *,
        split: str,
        epoch: int,
        batch: int,
        loss: float,
        fold: Optional[int] = None,
    ) -> None:
        """Record a single batch-level loss.

        Args:
            split: ``"train"``, ``"val"``, or ``"test"``.
            epoch: Current epoch index.
            batch: Current batch index within the epoch.
            loss: Scalar loss value.
            fold: Cross-validation fold index, if applicable.
        """
        if not self.enabled:
            return
        self.records.append(
            TrainerLogRecord(split=split, epoch=epoch, batch=batch, fold=fold, loss=loss)
        )

    def log_epoch(
        self,
        *,
        split: str,
        epoch: int,
        loss: float,
        metrics: Optional[dict[str, float]] = None,
        fold: Optional[int] = None,
    ) -> None:
        """Record epoch-level loss and optional metrics.

        Args:
            split: ``"train"``, ``"val"``, or ``"test"``.
            epoch: Current epoch index.
            loss: Mean loss for the epoch.
            metrics: Optional dict of additional scalar metrics.
            fold: Cross-validation fold index, if applicable.
        """
        if not self.enabled:
            return
        self.records.append(
            TrainerLogRecord(split=split, epoch=epoch, fold=fold, loss=loss, metrics=metrics)
        )

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def finalize_preds(preds: torch.Tensor, task: str) -> torch.Tensor:
        """Convert raw model output to discrete predictions.

        Args:
            preds: Raw model output tensor.
            task: ``"binary_classification"`` (argmax) or any other (identity).

        Returns:
            Prediction tensor.
        """
        if task == "binary_classification":
            return preds.argmax(dim=1)
        return preds

    def flush(self, trainer: Optional[object] = None) -> None:
        """Write buffered records to a Parquet file and clear the buffer.

        Args:
            trainer: Optional trainer object; used to read ``fold_idx`` if present.
        """
        if not self.enabled or not self.records:
            return

        df = pd.DataFrame(r.to_dict() for r in self.records)
        fold_idx = getattr(trainer, "fold_idx", None) if trainer else None

        if fold_idx is not None:
            df["fold"] = fold_idx
            suffix = f"_fold{fold_idx}"
        else:
            suffix = ""

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"torch_debug_{self.run_id}{suffix}.parquet")
        df.to_parquet(path)


class LightningDebugLogger(pl.Callback):
    """PyTorch Lightning callback that logs epoch-level metrics to a Parquet file.

    Args:
        run_id: Unique run identifier.
        prediction_task: ``"binary_classification"`` or ``"regression"``.
        average: Averaging strategy forwarded to :func:`compute_metrics`.
        debug_dir: Root directory for debug output files.
        enabled: When ``False``, all callbacks are no-ops.
    """

    def __init__(
        self,
        *,
        run_id: str,
        prediction_task: str,
        average: str = "binary",
        debug_dir: str = "debug",
        enabled: bool = True,
    ):
        self.run_id = run_id
        self.enabled = enabled
        self.prediction_task = prediction_task
        self.average = average
        self.debug_dir = debug_dir

        self.records: list[dict] = []

        # Per-epoch buffers, cleared after each epoch callback.
        self._train_preds: list = []
        self._train_targets: list = []
        self._val_preds: list = []
        self._val_targets: list = []
        self._train_losses: list = []
        self._val_losses: list = []
        self._train_batches: list = []
        self._val_batches: list = []

    # ------------------------------------------------------------------ #
    # Train callbacks                                                      #
    # ------------------------------------------------------------------ #

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.enabled:
            return

        x, y, *_ = batch
        preds = pl_module(x)

        if self.prediction_task == "binary_classification":
            y = y.long()
            preds_cls = preds.argmax(dim=1)
        else:
            y = y.float()
            preds_cls = preds.squeeze(1)

        self._train_preds.append(preds_cls.detach().cpu())
        self._train_targets.append(y.detach().cpu())
        self._train_losses.append(outputs["loss"].detach().cpu().item())
        self._train_batches.append(batch_idx)

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.enabled:
            return

        self._flush_epoch(
            trainer=trainer,
            split="train",
            epoch=trainer.current_epoch,
            losses=self._train_losses,
            preds=self._train_preds,
            targets=self._train_targets,
            fold=getattr(trainer, "fold_idx", None),
        )

        self._train_preds.clear()
        self._train_targets.clear()
        self._train_losses.clear()
        self._train_batches.clear()

    # ------------------------------------------------------------------ #
    # Validation callbacks                                                 #
    # ------------------------------------------------------------------ #

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not self.enabled:
            return

        x, y, *_ = batch
        preds = pl_module(x)

        if self.prediction_task == "binary_classification":
            y = y.long()
            preds_cls = preds.argmax(dim=1)
        else:
            y = y.float()
            preds_cls = preds.squeeze(1)

        self._val_preds.append(preds_cls.detach().cpu())
        self._val_targets.append(y.detach().cpu())
        self._val_losses.append(outputs["loss"].detach().cpu().item())
        self._val_batches.append(batch_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.enabled:
            return

        self._flush_epoch(
            trainer=trainer,
            split="val",
            epoch=trainer.current_epoch,
            losses=self._val_losses,
            preds=self._val_preds,
            targets=self._val_targets,
        )

        self._val_preds.clear()
        self._val_targets.clear()
        self._val_losses.clear()
        self._val_batches.clear()

    # ------------------------------------------------------------------ #
    # Core flush logic                                                     #
    # ------------------------------------------------------------------ #

    def _flush_epoch(self, *, split, epoch, losses, preds, targets, fold=None):
        from diff_benchmark.utils.scores import compute_metrics

        preds_np = torch.cat(preds).numpy()
        targets_np = torch.cat(targets).numpy()

        metrics = compute_metrics(
            y_true=targets_np.tolist(),
            y_pred=preds_np.tolist(),
            prediction_task=self.prediction_task,
            average=self.average,
        )

        self.records.append(
            TrainerLogRecord(
                split=split,
                epoch=epoch,
                loss=float(np.mean(losses)),
                metrics=metrics,
                fold=fold,
            )
        )

    # ------------------------------------------------------------------ #
    # Save on fit end                                                      #
    # ------------------------------------------------------------------ #

    def on_fit_end(self, trainer, pl_module):
        if not self.enabled or not self.records:
            return

        fold_idx = getattr(trainer, "fold_idx", None)
        run_dir = os.path.join(self.debug_dir, self.run_id)
        os.makedirs(run_dir, exist_ok=True)

        df = pd.DataFrame(r.to_dict() for r in self.records)
        out_path = os.path.join(run_dir, f"lightning_debug_{self.run_id}.parquet")
        df.to_parquet(out_path)


class LightningPrintLogger(pl.Callback):
    """Lightning callback that prints a compact training summary after each epoch.

    Args:
        run_id: Unique run identifier shown in log messages.
        epochs: Total number of training epochs (used in the progress display).
    """

    def __init__(self, *, run_id: str, epochs: int):
        self.run_id = run_id
        self.epochs = epochs
        self.py_logger = setup_logger(__name__)

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")

        if train_loss is None or val_loss is None:
            return

        self.py_logger.info(
            f"[{self.run_id}] "
            f"Epoch {trainer.current_epoch + 1}/{self.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )


# ------------------------------------------------------------------ #
# Module-level logging helpers                                        #
# ------------------------------------------------------------------ #


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create (or retrieve) a named logger with a consistent format.

    Args:
        name: Logger name, typically ``__name__``.
        level: Logging level (e.g. ``logging.DEBUG``).
        log_file: Optional path to a file handler.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True  # Required for SLURM to forward messages to root.

    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def configure_logging(level: int = logging.DEBUG) -> None:
    """Configure the root logger.  Call once at application startup.

    Args:
        level: Logging level to apply to the root logger.
    """
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        root.addHandler(sh)
