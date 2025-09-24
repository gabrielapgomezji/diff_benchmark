import json
import os
from pathlib import Path

import numpy as np
from filelock import FileLock


def is_cached(
    run_id: str, output_dir: Path, results_filename="all_results.json"
) -> bool:
    """
    Checks if the results for a given run ID are cached in the specified output directory.
    Args:
        run_id (str): The unique identifier for the run whose results are being checked.
        output_dir (Path): The directory where the results file is stored.
        results_filename (str, optional): The name of the results file. Defaults to "all_results.json".
    Returns:
        bool: True if the results for the specified run ID are found in the results file, False otherwise.
    """
    out_path = output_dir / results_filename
    if not out_path.exists():
        return False
    with open(out_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    return any(
        result.get("pipeline", {}).get("run_id") == run_id for result in all_results
    )


def save_model_results(
    summary: dict, output_dir: Path, results_filename="all_results.json"
):
    """
    Saves the model results to a specified output directory in JSON format.
    Parameters:
        summary (dict): A dictionary containing the model results and metadata.
        output_dir (Path): The directory where the results will be saved.
        results_filename (str, optional): The name of the results file. Defaults to "all_results.json".
    The function checks for a training log associated with the run ID in the summary.
    If the log exists, it loads the training history from the log file and adds it to the summary.
    The log file is then deleted to clean up.
    The function creates the output directory if it does not exist and acquires a file lock
    to ensure that results are saved safely. It appends the new results to the existing results
    if the results file already exists, or creates a new file if it does not.
    Finally, it prints a message indicating that the results have been saved.
    """
    run_id = summary.get("pipeline", {}).get("run_id", None)
    if run_id:
        log_file = Path("./data/results") / f"{run_id}_training_log.json"
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            summary["history"] = history
            os.remove(log_file)  # cleanup
        else:
            summary["history"] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / results_filename

    lock = FileLock(f"{out_path}.lock")
    with lock:
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        else:
            all_results = []

        all_results.append(summary)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        print(
            f"Saved results for {summary['model_name']} (run_id={summary['pipeline']['run_id']})"
        )


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
    with open(out_path, "w", encoding="utf-8") as f:
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

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary results to {out_path}")
