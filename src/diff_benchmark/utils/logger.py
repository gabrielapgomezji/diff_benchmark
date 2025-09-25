import json
import csv
from pathlib import Path
import torch

class TrainLogger:
    def __init__(self, run_id="unnamed_run", save_dir="./data/results/logger", monitor="val_accuracy", mode="max"):
        """
        Training logger and checkpoint saver.

        Args:
            run_id (str): Identifier for the training run.
            save_dir (str): Base directory to save logs and models.
            monitor (str): Metric name to monitor for saving the best model.
            mode (str): "max" if higher is better, "min" if lower is better.
        """
        self.run_id = run_id
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode

        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.history = {
            "train": {"epoch": [], "loss": [], "accuracy": []},
            "val": {"epoch": [], "loss": [], "accuracy": []},
            "predictions": {"epoch": [], "y_true": [], "y_pred": [], "scores": []}
        }

        # paths
        self.logs_path = self.save_dir / "logs"
        self.models_path = self.save_dir / "models"
        self.logs_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.best_path = self.models_path / f"{self.run_id}_best.pth"
        self.last_path = self.models_path / f"{self.run_id}_last.pth"
    
    def log_batch(self, phase:str, epoch: int, loss: float, accuracy: float=None):
        self.history[phase]["epoch"].append(epoch)
        self.history[phase]["loss"].append(loss)
        if accuracy is not None:
            self.history[phase]["accuracy"].append(accuracy)
            
    def log_predictions(self, epoch, y_true, y_pred, scores=None):
        self.history["predictions"]["epoch"].append(epoch)
        self.history["predictions"]["y_true"].append(y_true.to_list())
        self.history["predictions"]["y_pred"].append(y_pred.to_list())
        if scores is not None:
            self.history["predictions"]["scores"].append(scores.to_list())

    def _is_best(self, score):
        if self.mode == "max":
            return score > self.best_score
        elif self.mode == "min":
            return score< self.best_score
        raise ValueError("mode should be 'max' or 'min'")
        
    def save_checkpoint(self, model, epoch, current_score, is_last=False):
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
                print(f"[INFO] Saved best model at epoch {epoch} with {self.monitor}={current_score:.4f}")
        else:
            torch.save(model.state_dict(), self.last_path)
            print(f"[INFO] Saved last model at epoch {epoch}")
    
    def save_logs(self):
        """Save history to JSON and CSV."""
        json_path = self.logs_path / f"{self.run_id}_log.json"
        # csv_path = self.logs_path / f"{self.run_id}_log.csv"
        
        with open(json_path, 'w', encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)
        
        
        print(f"[INFO] Logs saved at {json_path}")