import os
import pandas as pd
import torch
import pytorch_lightning as pl

from typing import Optional, Dict, List
import os
import pandas as pd
import torch

from dataclasses import dataclass
from typing import Dict, Optional, Literal

import logging
import sys
from typing import Optional

from tqdm import tqdm

def tqdm_if_enabled(iterable, *, desc=None, total=None, enabled=True):
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=False,
        disable=not enabled or not sys.stdout.isatty(),
    )

@dataclass
class TrainerLogRecord:
    split: Literal["train", "val", "test"]
    epoch: int
    loss: float
    # optional / event-dependent
    batch: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None
    fold: Optional[int] = None

    def to_dict(self) -> dict:
        """
        Flatten metrics so Pandas gets nice columns.
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

        self.records: List[Dict] = []

        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)

    # -------------------------
    # batch-level logging
    # -------------------------
    def log_batch(
        self,
        *,
        split: str,
        epoch: int,
        batch: int,
        loss: float,
        fold: Optional[int] = None,
    ):
        if not self.enabled:
            return

        self.records.append(
            TrainerLogRecord(
                split=split,
                epoch=epoch,
                batch=batch,
                fold=fold,
                loss=loss,
            )
        )

    # -------------------------
    # epoch-level logging
    # -------------------------
    def log_epoch(
        self,
        *,
        split: str,
        epoch: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        fold: Optional[int] = None,
    ):
        if not self.enabled:
            return

        self.records.append(
            TrainerLogRecord(
                split=split,
                epoch=epoch,
                fold=fold,
                loss=loss,
                metrics=metrics,
            )
        )

    # -------------------------
    # utils
    # -------------------------
    @staticmethod
    def finalize_preds(preds: torch.Tensor, task: str) -> torch.Tensor:
        if task == "binary_classification":
            return preds.argmax(dim=1)
        return preds

    def flush(self, trainer: Optional[object] = None):
        if not self.enabled or not self.records:
            return

        df = pd.DataFrame(r.to_dict() for r in self.records)
        fold_idx = getattr(trainer, "fold_idx", None) if trainer else None

        if fold_idx is not None:
            df["fold"] = fold_idx
            suffix = f"_fold{fold_idx}"
        else:
            suffix = ""
        
        run_dir = os.path.join(self.output_dir, self.run_id)
        os.makedirs(run_dir, exist_ok=True) 
    
        path = os.path.join(
            self.output_dir, f"{self.run_id}/torch_debug_{self.run_id}{suffix}.parquet"
        )
        df.to_parquet(path)


class LightningDebugLogger(pl.Callback):
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

        # epoch buffers
        self._train_preds = []
        self._train_targets = []
        self._val_preds = []
        self._val_targets = []
        self._train_losses = []
        self._val_losses = []
        self._train_batches = []
        self._val_batches = []

    # ---------- TRAIN ----------

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
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

    # ---------- VAL ----------

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

    # ---------- CORE ----------

    def _flush_epoch(self, *, split, epoch, losses, preds, targets, fold=None):
        import numpy as np
        from diff_benchmark.utils.scores import compute_metrics
        
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()

        metrics = compute_metrics(
            y_true=targets.tolist(),
            y_pred=preds.tolist(),
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

    # ---------- SAVE ----------

    def on_fit_end(self, trainer, pl_module):
        if not self.enabled or not self.records:
            return
        fold_idx = getattr(trainer, "fold_idx", None)
        suffix = f"_fold{fold_idx}" if fold_idx is not None else ""

        os.makedirs(os.path.join(self.debug_dir, self.run_id), exist_ok=True)
        df = pd.DataFrame(r.to_dict() for r in self.records)
        df.to_parquet(os.path.join(self.debug_dir, f"{self.run_id}/lightning_debug_{self.run_id}.parquet"))


class LightningPrintLogger(pl.Callback):
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

        epoch = trainer.current_epoch

        self.py_logger.info(
            f"[{self.run_id}] "
            f"Epoch {epoch + 1}/{self.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )


# ------------------
#   Print Logger
# ------------------

LOG_FORMAT = (
    "%(levelname)s - %(asctime)s - %(name)s - %(message)s"
)

DATE_FORMAT = "%H:%M:%S"


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True  # CRITICAL for SLURM

    if not logger.handlers:
        formatter = logging.Formatter(
            LOG_FORMAT,
            datefmt=DATE_FORMAT,
        )
        # stdout handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # optional file handler
        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def configure_logging(level=logging.DEBUG):
    """
    Configure global logging level.
    Call ONCE, from main.
    """
    # logging.getLogger().setLevel(level)
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        root.addHandler(sh)
    