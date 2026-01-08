import numpy as np

def make_fold_summary(fold_idx, train_score, test_score, y_train, train_pred, y_test, test_pred, primary_metric):
    """
    Create a standardized summary dict for a fold.
    """
    return {
        f"fold_{fold_idx+1}": {
            "train": {
                "score": train_score[primary_metric],
                "predictions": train_pred.tolist(),
                "targets": y_train.tolist(),
            },
            "test": {
                "score": test_score[primary_metric],
                "predictions": test_pred.tolist(),
                "targets": y_test.tolist(),
            },
        }
    }

def update_summary(summary, fold_idx, train_score, test_score, y_train, train_pred, y_test, test_pred, primary_metric):
    """
    Update the summary dict with a new fold summary.
    """
    fold_summary = make_fold_summary(fold_idx, train_score, test_score, y_train, train_pred, y_test, test_pred, primary_metric)
    summary["results"]["folds"].update(fold_summary)
    return summary

def compute_summary_stats(scores_list, primary_metric):
    values = [fold[primary_metric] for fold in scores_list]
    return float(np.mean(values)), float(np.std(values))