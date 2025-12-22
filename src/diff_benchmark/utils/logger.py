import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsManager:
    """
    Compute and store metrics for classification tasks.

    Args:
        average (str): averaging method for multi-class/multi-label metrics
                        ('binary', 'macro', 'micro', 'weighted')
    """

    def __init__(self, average: str ="binary"):
        self.average = average
        self.reset()

    def reset(self):
        """Reset stored predictions."""
        self.y_true = []
        self.y_pred = []
        self.y_scores = []

    def update(self, y_true: list, y_pred: list, y_scores: list | None = None):
        """Update stored predictions with new batch results.
        Parameters:
            y_true (list): Ground truth labels.
            y_pred (list): Predicted labels.
            y_scores (list, optional): Prediction scores or probabilities.
        """
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)
        if y_scores is not None:
            self.y_scores.extend(y_scores)

    def compute_batch(self, y_true: list, y_pred: list, y_scores: list | None = None) -> dict:
        """Compute metrics for a single batch only.
        return metrics for a single batch only.
        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels.
            y_scores (array-like, optional): Prediction scores or probabilities.
        Returns:
            dict: Metrics results.
        """
        return self._compute_core(y_true, y_pred, y_scores)

    def compute(self) -> dict:
        """Compute metrics over ALL stored batches (epoch).
        Returns:
            dict: Metrics results.
        """
        return self._compute_core(self.y_true, self.y_pred, self.y_scores)

    def _compute_core(self, y_true: list, y_pred: list, y_scores: list | None = None) -> dict:
        """
        Compute a dictionary of metrics.

        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels.
            y_scores (array-like, optional): Prediction scores or probabilities.

        Returns:
            dict: Metrics results.
        """
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        y_scores = np.array(self.y_scores) if len(self.y_scores) > 0 else None

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average=self.average, zero_division="warn"
            ),
            "recall": recall_score(
                y_true, y_pred, average=self.average, zero_division="warn"
            ),
            "f1": f1_score(y_true, y_pred, average=self.average, zero_division="warn"),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        if y_scores is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_scores, multi_class="ovr")
            except ValueError:
                metrics["roc_auc"] = None
        return metrics


class TrainLogger:
    """
    Training logger and checkpoint saver.

    Args:
        fold_idx (int): Index of the current fold (for cross-validation).
        run_id (str): Identifier for the training run.
        save_dir (str): Base directory to save logs and models.
        monitor (str): Metric name to monitor for saving the best model.
        mode (str): "max" if higher is better, "min" if lower is better.
    """

    def __init__(
        self,
        fold_idx: int,
        run_id: str ="unnamed_run",
        save_dir: str ="./data/results/logger",
        monitor: str ="val_accuracy",
        mode: str ="max",
    ):
        self.run_id = run_id
        self.fold_idx = fold_idx
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode

        self.val_scores = []
        self.waiting_candidate = None
        self.patience_window = 3  # number of epochs to average
        self.tolerance = 0.005  # tolerance for next-epoch drop
        self.smoothed_best = float("-inf") if mode == "max" else float("inf")
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.history = {
            "train": {"epoch": [], "loss": [], "accuracy": []},
            "val": {"epoch": [], "loss": [], "accuracy": []},
            "predictions": {"epoch": [], "y_true": [], "y_pred": [], "scores": []},
        }

        # paths
        self.logs_path = self.save_dir / "logs"
        self.models_path = self.save_dir / "models"
        self.logs_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.best_path = (
            self.models_path / f"{self.run_id}_fold{self.fold_idx}_best.pth"
        )
        self.last_path = (
            self.models_path / f"{self.run_id}_fold{self.fold_idx}_last.pth"
        )

    # def log_batch(self, phase:str, epoch: int, loss: float, accuracy: float=None):
    #     self.history[phase]["epoch"].append(epoch)
    #     self.history[phase]["loss"].append(loss)
    #     if accuracy is not None:
    #         self.history[phase]["accuracy"].append(accuracy)

    def log_batch(self, phase: str, epoch: int, batch: int, loss: float, metrics: dict):
        """Log metrics for a batch.
        Args:
            phase (str): 'train' or 'val'.
            epoch (int): Current epoch.
            batch (int): Current batch.
            loss (float): Loss value.
            metrics (dict): Dictionary of additional metrics.
        """
        self.history["batch"].append(
            {"phase": phase, "epoch": epoch, "batch": batch, "loss": loss, **metrics}
        )

    def log_epoch(self, phase: str, epoch: int, metrics: dict):
        """Log metrics for an epoch.
        Args:
            phase (str): 'train' or 'val'.
            epoch (int): Current epoch.
            metrics (dict): Dictionary of metrics.
        """
        self.history["epoch"].append({"phase": phase, "epoch": epoch, **metrics})

    def log_predictions(self, epoch: int, y_true: list, y_pred: list, scores: list | None = None):
        """Log predictions for an epoch.
        Args:
            epoch (int): Current epoch.
            y_true (list): Ground truth labels.
            y_pred (list): Predicted labels.
            scores (list, optional): Prediction scores or probabilities.
        """
        self.history["predictions"]["epoch"].append(epoch)
        self.history["predictions"]["y_true"].append(y_true.to_list())
        self.history["predictions"]["y_pred"].append(y_pred.to_list())
        if scores is not None:
            self.history["predictions"]["scores"].append(scores.to_list())

    def log_metrics(
        self, phase: str, epoch: int, batch: int = None, metrics: dict = None
    ):
        """
        Log metrics at batch or epoch level.
        Args:
            phase (str): 'train' or 'val'.
            epoch (int): Current epoch.
            batch (int, optional): Current batch. If None, it's epoch-level logging.
            metrics (dict): Dictionary of metrics.
        """
        key = f"{phase}_metrics"
        if key not in self.history:
            self.history[key] = []

        entry = {"epoch": epoch, "metrics": metrics}
        if batch is not None:
            entry["batch"] = batch
        self.history[key].append(entry)
        print(f"[INFO] Metrics at epoch {epoch}: {metrics}")

    def _is_best(self, score: float) -> bool:
        """Check if the current score is the best so far.
        Args:
            score (float): Current value of the monitored metric.
        Returns:
            bool: True if current score is the best, False otherwise.
        """
        if self.mode == "max":
            return score > self.best_score
        if self.mode == "min":
            return score < self.best_score
        raise ValueError("mode should be 'max' or 'min'")

    def save_checkpoint(self, model: torch.nn.Module, epoch: int, current_score: float, is_last: bool = False):
        """
        Save model checkpoint if current score is the best.

        Args:
            model (torch.nn.Module): The model to save.
            epoch (int): Current epoch number.
            current_score (float): Current value of the monitored metric.
            is_last (bool): If True, save as the last checkpoint regardless of score.
        """
        if not is_last:
            if self._is_best(current_score):
                self.best_score = current_score
                torch.save(model.state_dict(), self.best_path)
                print(
                    f"[INFO] Saved best model at epoch {epoch} with {self.monitor}={current_score:.4f}"
                )
        else:
            torch.save(model.state_dict(), self.last_path)
            print(f"[INFO] Saved last model at epoch {epoch}")

    def save_logs(self):
        """Save history to JSON and CSV."""
        json_path = self.logs_path / f"{self.run_id}_log.json"
        # csv_path = self.logs_path / f"{self.run_id}_log.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)

        print(f"[INFO] Logs saved at {json_path}")

    def update_smooth_checkpoint(self, model: torch.nn.Module, epoch: int, val_score: float):
        """
        Check 3-step smoothed validation accuracy and save checkpoint
        only when improvement is stable.
        Args:
            model (torch.nn.Module): The model to save.
            epoch (int): Current epoch number.
            val_score (float): Current validation score.
        """
        self.val_scores.append(val_score)
        if len(self.val_scores) < self.patience_window:
            return  # not enough epochs yet

        # Compute 3-step moving average
        recent_avg = np.mean(self.val_scores[-self.patience_window :])

        # If this moving average is new best → candidate for saving
        is_improvement = (
            recent_avg > self.smoothed_best
            if self.mode == "max"
            else recent_avg < self.smoothed_best
        )

        if is_improvement:
            self.waiting_candidate = (epoch, recent_avg, model.state_dict())
            self.smoothed_best = recent_avg
            print(f"[Epoch {epoch}] Candidate smoothed avg: {recent_avg:.4f}")

        # Confirm stability one epoch later
        if (
            self.waiting_candidate is not None
            and len(self.val_scores) > self.patience_window
        ):
            last_val = self.val_scores[-1]
            candidate_epoch, candidate_score, _ = self.waiting_candidate

            # Stable if score hasn't dropped too much
            stable = (
                last_val >= candidate_score * (1 - self.tolerance)
                if self.mode == "max"
                else last_val <= candidate_score * (1 + self.tolerance)
            )
            if stable:
                print(
                    f"[INFO] Stable checkpoint saved (epoch {candidate_epoch}, smooth={candidate_score:.4f})"
                )
                self.save_checkpoint(
                    model, candidate_epoch, candidate_score, is_last=False
                )

                self.waiting_candidate = None
