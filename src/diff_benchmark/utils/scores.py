import numpy as np
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.utils.class_weight import compute_sample_weight

__all__ = ["compute_metrics"]


def compute_metrics(
    y_true: list,
    y_pred: list,
    prediction_task: str,
    average: str = "binary",
    zero_division: str = "warn",
) -> dict:
    """Compute standard classification and regression metrics.
    Parameters:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        prediction_task (str): Type of prediction task - "classification" or "regression".
        average (str): Averaging method for multi-class classification.
        zero_division (str): Handling of zero division cases.
    """
    sample_weight = compute_sample_weight("balanced", y_true)
    if prediction_task == "binary_classification":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "accuracy_weighted": accuracy_score(
                y_true, y_pred, sample_weight=sample_weight
            ),
            "precision": precision_score(
                y_true, y_pred, average=average, zero_division=zero_division
            ),
            "recall": recall_score(
                y_true, y_pred, average=average, zero_division=zero_division
            ),
            "f1": f1_score(
                y_true, y_pred, average=average, zero_division=zero_division
            ),
        }

    if prediction_task == "regression":
        return {
            "rmse": root_mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "explained_variance": explained_variance_score(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "mae_weighted": mean_absolute_error(
                y_true, y_pred, sample_weight=sample_weight
            ),
            "rmse_weighted": root_mean_squared_error(
                y_true, y_pred, sample_weight=sample_weight
            ),
            "pearson_correlation": np.corrcoef(y_true, y_pred)[0, 1],
        }

    raise ValueError(
        "Invalid prediction_task. Choose either 'binary_classification' or 'regression'."
    )
