from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

__all__ = ["accuracy_score"]


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
    if prediction_task == "binary_classification":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average=average, zero_division=zero_division
            ),
            "recall": recall_score(
                y_true, y_pred, average=average, zero_division=zero_division
            ),
            "f1": f1_score(
                y_true, y_pred, average=average, zero_division=zero_division
            ),
            # "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    if prediction_task == "regression":
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "explained_variance": explained_variance_score(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
        }

    raise ValueError(
        "Invalid prediction_task. Choose either 'classification' or 'regression'."
    )
