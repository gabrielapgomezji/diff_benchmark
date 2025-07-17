import json
from pathlib import Path

import numpy as np


def save_fold_results(
    model_name: str,
    fold_results: list,
    output_dir: Path,
):
    """
    Save per-fold results to a JSON file.

    Args:
        model_name (str): Model name identifier.
        fold_results (list): List of dicts with keys ['fold', 'score', 'predictions', 'targets'].
        output_dir (Path): Where to save the results.
    """
    model_name = model_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_name}_fold_results.json"
    with open(out_path, "w") as f:
        json.dump(fold_results, f, indent=2)
    print(f"Saved per-fold results to: {out_path}")


def save_summary_results(
    model_name,
    train_scores,
    test_scores,
    train_preds,
    test_preds,
    train_targets,
    test_targets,
    output_dir: Path,
):
    """
    Save summary statistics (mean/std) for train and test sets.

    Args:
        model_name (str): Model name identifier.
        all_scores (list): List of float scores (1 per fold).
        all_preds (list): List of prediction arrays/lists.
        all_targets (list): List of true target arrays/lists.
        output_dir (Path): Where to save the summary.
    """
    model_name = model_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_name}_summary.json"
    train_preds = np.array(train_preds)
    test_preds = np.array(test_preds)
    train_targets = np.array(train_targets)
    test_targets = np.array(test_targets)

    summary = {
        "model": model_name,
        "train_score_mean": float(np.mean(train_scores)),
        "train_score_std": float(np.std(train_scores)),
        "test_score_mean": float(np.mean(test_scores)),
        "test_score_std": float(np.std(test_scores)),
        "train_predictions_mean": np.mean(train_preds, axis=0).tolist(),
        "train_targets_mean": np.mean(train_targets, axis=0).tolist(),
        "test_predictions_mean": np.mean(test_preds, axis=0).tolist(),
        "test_targets_mean": np.mean(test_targets, axis=0).tolist(),
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary results to {out_path}")
