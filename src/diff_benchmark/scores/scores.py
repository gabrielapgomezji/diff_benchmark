import numpy as np


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
