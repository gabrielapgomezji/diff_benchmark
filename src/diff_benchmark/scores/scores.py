import numpy as np


def mse_score(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        return mse