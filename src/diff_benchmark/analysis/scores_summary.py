import os
import json
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error

def summarize_folds_to_csv(fold_results_path: Path, output_csv_path: Path):
    # Load fold results
    with open(fold_results_path, "r") as f:
        fold_results = json.load(f)

    os.makedirs(output_csv_path.parent, exist_ok=True)
    
    n_folds = len(fold_results)
    # Check number of features from first fold predictions (train)
    # n_features = len(fold_results[0]["train"]["predictions"][0])
    example_pred = fold_results[0]["train"]["predictions"][0]
    if isinstance(example_pred, (float, int)):
        n_features = 1
    else:
        n_features = len(example_pred)

    # Prepare arrays to store per-fold scores per feature
    train_scores = np.zeros((n_folds, n_features))
    test_scores = np.zeros((n_folds, n_features))

    # Loop over folds and features to compute MSE per fold per feature
    for fold_idx, fold in enumerate(fold_results):
        train_preds = np.array(fold["train"]["predictions"])
        train_targets = np.array(fold["train"]["targets"])
        test_preds = np.array(fold["test"]["predictions"])
        test_targets = np.array(fold["test"]["targets"])

        for feat_idx in range(n_features):
            if n_features == 1:
                # Univariate case, use arrays as-is
                train_scores[fold_idx, feat_idx] = mean_squared_error(train_targets, train_preds)
                test_scores[fold_idx, feat_idx] = mean_squared_error(test_targets, test_preds)
            else:
                # Multivariate case, index the feature column
                train_scores[fold_idx, feat_idx] = mean_squared_error(
                    train_targets[:, feat_idx], train_preds[:, feat_idx]
                )
                test_scores[fold_idx, feat_idx] = mean_squared_error(
                    test_targets[:, feat_idx], test_preds[:, feat_idx]
                )

    # Compute mean and std across folds per feature
    train_mean = np.mean(train_scores, axis=0)
    train_std = np.std(train_scores, axis=0)
    test_mean = np.mean(test_scores, axis=0)
    test_std = np.std(test_scores, axis=0)

    # Write to CSV
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Train MSE Mean", "Train MSE Std", "Test MSE Mean", "Test MSE Std"])
        for i in range(n_features):
            writer.writerow([i + 1, train_mean[i], train_std[i], test_mean[i], test_std[i]])

    print(f"Saved summary CSV at {output_csv_path}")
    
    per_fold_csv_path = output_csv_path.parent / (output_csv_path.stem + "_per_fold.csv")

    with open(per_fold_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Fold", "Train MSE (avg)", "Test MSE (avg)"])
        for fold_idx in range(n_folds):
            train_avg = train_scores[fold_idx].mean()
            test_avg = test_scores[fold_idx].mean()
            writer.writerow([output_csv_path.stem, fold_idx, train_avg, test_avg])

    print(f"Saved per-fold summary CSV at {per_fold_csv_path}")

    print(f"Saved summary CSV at {per_fold_csv_path}")
    