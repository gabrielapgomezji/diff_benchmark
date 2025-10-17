from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

__all__ = ["accuracy_score"]

def compute_metrics(y_true, y_pred, average="binary", zero_division="warn"):
    """Compute standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=zero_division),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=zero_division),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=zero_division),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }



# def accuracy_score(y_true, y_pred):
#     """
#     Calculate the accuracy score between true and predicted values.
#     Parameters:
#         y_true (array-like): The ground truth (correct) target values.
#         y_pred (array-like): The estimated target values.
#     Returns:
#         float: The accuracy score between the true and predicted values.
#     """
#     accuracy = accuracy_score(y_true, y_pred)
#     return accuracy
