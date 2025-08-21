import numpy as np
from sklearn.metrics import accuracy_score

__all__ = ["accuracy_score"]


def mse_score(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.
    Parameters:
        y_true (array-like): The ground truth (correct) target values.
        y_pred (array-like): The estimated target values.
    Returns:
        float: The mean squared error between the true and predicted values.
    """

    mse = np.mean((y_true - y_pred) ** 2)
    return mse


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
