import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diff_benchmark.models.utils_models.trainer import SklearnModel
from sklearn.base import BaseEstimator

from torch.utils.data import DataLoader


class PCALinearModel(SklearnModel):
    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", None)
        self.output_dim = 1 #if self.prediction_task == "regression" else 2
        if self.prediction_task == "classification":
            head = LogisticRegression(max_iter=1000)
            scoring = "accuracy"

            param_grid = {
                "pca__n_components": [10],  # [10, 50, 100],
                "linear__C": [0.01, 0.1, 1],  # [0.01, 0.1, 1, 10, 100],
                "linear__solver": ["lbfgs"],
                "linear__penalty": ["l2"],
            }

        else:  # if self.prediction_task == "regression":  # regression
            head = Ridge()  # or LinearRegression()
            scoring = "neg_mean_squared_error"

            param_grid = {
                "pca__n_components": [10],  # [10, 50, 100],
                "linear__alpha": [
                    0.01,
                    0.1,
                    1,
                ],  # [0.01, 0.1, 1, 10],  # Ridge regularization
            }

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("linear", head),
            ]
        )

        # Grid search object
        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

class LinearModel(SklearnModel):
    """
    LinearModel is a model that uses Logistic Regression for dimensionality reduction and classification.
    Attributes:
        n_components (int): The number of principal components to keep.
        pca (PCA): PCA instance for dimensionality reduction.
        model (LogisticRegression): Logistic regression model for classification.
    Methods:
        _dataloader_to_numpy(dataloader):
            Converts the data from the dataloader into numpy arrays for features and labels.
        fit(dataloader):
            Fits the PCA and logistic regression model on the provided dataloader.
        predict(dataloader):
            Transforms the input data using PCA and predicts the class labels using the logistic regression model.
    """

    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", None)
        self.output_dim = 1 #2 if self.prediction_task == "regression" else 1

        if self.prediction_task == "classification":
            head = LogisticRegression(max_iter=1000)
            scoring = "accuracy"
            param_grid = {
                "linear__C": [0.01, 0.1, 1],
                "linear__solver": ["lbfgs"],
                "linear__penalty": ["l2"],
            }
        else:
            head = Ridge()
            scoring = "neg_mean_squared_error"
            param_grid = {
                "linear__alpha": [0.01, 0.1, 1],
            }

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("linear", head),
            ]
        )

        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
