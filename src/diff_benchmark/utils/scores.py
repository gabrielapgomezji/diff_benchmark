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
    """Compute standard evaluation metrics for a prediction task.

    Args:
        y_true: Ground-truth labels or values.
        y_pred: Model predictions.
        prediction_task: Either ``"binary_classification"`` or ``"regression"``.
        average: Averaging strategy for multi-class classification metrics.
        zero_division: How sklearn handles zero-division in precision/recall/F1.

    Returns:
        Dictionary mapping metric names to scalar float values.

    Raises:
        ValueError: If *prediction_task* is not one of the supported values.
    """
    sample_weight = compute_sample_weight("balanced", y_true)

    if prediction_task == "binary_classification":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "accuracy_weighted": accuracy_score(y_true, y_pred, sample_weight=sample_weight),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=zero_division),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=zero_division),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=zero_division),
        }

    if prediction_task == "regression":
        return {
            "rmse": root_mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "explained_variance": explained_variance_score(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "mae_weighted": mean_absolute_error(y_true, y_pred, sample_weight=sample_weight),
            "rmse_weighted": root_mean_squared_error(y_true, y_pred, sample_weight=sample_weight),
            "pearson_correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
        }

    raise ValueError(
        f"Invalid prediction_task '{prediction_task}'. "
        "Choose either 'binary_classification' or 'regression'."
    )
