import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diff_benchmark.models.utils_models.trainer import SklearnModel


class PCALinearModel(SklearnModel):
    """
    A scikit-learn based model combining PCA dimensionality reduction with linear classification or regression.
    This class extends SklearnModel to create a pipeline that preprocesses data with standardization
    and PCA, then applies either logistic regression for classification tasks or Ridge regression
    for regression tasks. Hyperparameters are optimized using GridSearchCV with 5-fold cross-validation.
    Attributes:
        prediction_task (str): Type of prediction task - either "classification" or "regression".
        output_dim (int): Number of output dimensions (currently fixed to 1).
    Methods:
        _build_model(**kwargs) -> BaseEstimator:
            Constructs and returns a GridSearchCV object wrapping a Pipeline.
            For classification tasks:
                - Uses LogisticRegression as the final estimator
                - Optimizes PCA n_components, LogisticRegression C, solver, and penalty parameters
                - Uses accuracy as the scoring metric
            For regression tasks:
                - Uses Ridge regression as the final estimator
                - Optimizes PCA n_components and Ridge alpha (regularization) parameters
                - Uses negative mean squared error as the scoring metric
            The pipeline consists of three stages:
                1. StandardScaler: Standardizes features
                2. PCA: Reduces dimensionality to specified components
                3. Linear model: Applies LogisticRegression or Ridge
            Args:
                **kwargs: Keyword arguments including 'prediction_task' specifying the task type.
            Returns:
                GridSearchCV: Configured grid search object for model optimization with 5-fold CV
                             and parallel processing using all available cores.
    """

    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", None)
        self.output_dim = 1
        if self.prediction_task == "binary_classification":
            head = LogisticRegression(max_iter=1000)
            scoring = "balanced_accuracy"

            param_grid = {
                "pca__n_components": [
                    10,
                    20,
                    30,
                    50,
                    60,
                    75,
                    100,
                    400,
                ],  # [10, 50, 100],
                "linear__C": np.logspace(-10, 10, 21),  # [0.01, 0.1, 1, 10, 100],
                "linear__solver": ["lbfgs"],
                "linear__penalty": ["l2"],
            }

        else:  # if self.prediction_task == "regression":  # regression
            head = Ridge()  # or LinearRegression()
            scoring = "neg_mean_absolute_error"

            param_grid = {
                "pca__n_components": [
                    10,
                    20,
                    30,
                    50,
                    60,
                    75,
                    100,
                    400,
                ],  # [10, 50, 100],
                "linear__alpha": np.logspace(
                    -10, 10, 21
                ),  # [0.01, 0.1, 1, 10],  # Ridge regularization
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
    LinearModel applies GridSearchCV over a StandardScaler → Linear head pipeline.

    For binary classification, the head is LogisticRegression (L2, lbfgs) and the
    regularisation strength ``C`` is searched on a log-scale grid.
    For regression, the head is Ridge and the regularisation ``alpha`` is searched
    on the same log-scale grid.  No PCA stage is used.
    """

    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", None)
        self.output_dim = 1

        if self.prediction_task == "binary_classification":
            head = LogisticRegression(max_iter=1000)
            scoring = "balanced_accuracy"
            param_grid = {
                "linear__C": np.logspace(-10, 10, 21),  # [0.01, 0.1, 1], #
                "linear__solver": ["lbfgs"],
                # "linear__penalty": ["l2"],
            }
        else:
            head = Ridge()
            scoring = "neg_mean_absolute_error"
            param_grid = {
                "linear__alpha": np.logspace(-10, 10, 21),  # [0.01, 0.1, 1],
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


class LassoModel(SklearnModel):
    """
    LassoModel applies GridSearchCV over a StandardScaler → L1-penalised linear head pipeline.

    For binary classification, the head is LogisticRegression with ``penalty="l1"``
    (saga solver) and the regularisation strength ``C`` is searched on a log-scale grid.
    For regression, the head is sklearn's ``Lasso`` and ``alpha`` is searched on the
    same grid.  No PCA stage is used.
    """

    def _build_model(self, **kwargs) -> BaseEstimator:
        self.prediction_task = kwargs.get("prediction_task", None)
        self.output_dim = 1

        if self.prediction_task == "binary_classification":
            head = LogisticRegression(
                penalty="l1", solver="saga", max_iter=5000  # or "liblinear"
            )
            scoring = "balanced_accuracy"
            param_grid = {
                "linear__C": np.logspace(-10, 10, 21),  # [0.01, 0.1, 1], #
                # "linear__solver": ["lbfgs"],
            }
        else:
            head = Lasso(max_iter=10000)
            scoring = "neg_mean_absolute_error"
            param_grid = {
                "linear__alpha": np.logspace(-10, 10, 21),
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
