import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diff_benchmark.models.base import NumpyAbstractModel


class PCALogisticRegressionModel(NumpyAbstractModel):
    """
    PCALogisticRegressionModel is a model that combines Principal Component Analysis (PCA)
    with Logistic Regression for dimensionality reduction and classification.
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

    data_type = "array"
    prediction_task = None

    def __init__(self):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("logreg", LogisticRegression(max_iter=1000)),
            ]
        )

        # Grid of hyperparameters
        param_grid = {
            "pca__n_components": [10, 50, 100],  # number of PCA components to try
            "logreg__C": [0.01, 0.1, 1, 10, 100],  # inverse regularization strength
            "logreg__penalty": ["l2"],  # penalty (l1 requires saga solver)
            "logreg__solver": ["lbfgs"],  # stable solver for l2
        }

        # Grid search object
        self.model = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="accuracy",  # for binary classification
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

    def _dataloader_to_numpy(self, dataloader):
        features_list = []
        targets_list = []
        for features_batch, targets_batch, _ in dataloader:
            features_list.append(features_batch.numpy())
            targets_list.append(targets_batch.numpy())
        features = np.concatenate(features_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        return features, targets

    def fit(self, dataloader):
        """Fit PCA and then logistic regression on reduced features."""
        features, targets = self._dataloader_to_numpy(dataloader)
        features_reshaped = features.reshape(features.shape[0], -1)
        self.model.fit(features_reshaped, targets.flatten())
        # print("Best params found:", self.model.best_params_)

    def predict(self, dataloader):
        """Transform input with PCA and predict with logistic regression."""
        features, _ = self._dataloader_to_numpy(dataloader)
        features_reshaped = features.reshape(features.shape[0], -1)
        return self.model.predict(features_reshaped)

class PCALinearModel(NumpyAbstractModel):
    
    data_type = "array"
    prediction_task = None

    def __init__(self):
        if self.prediction_task == "classification":
            head = LogisticRegression(max_iter=1000)
            scoring = "accuracy"

            param_grid = {
                "pca__n_components": [10, 50, 100],
                "linear__C": [0.01, 0.1, 1, 10, 100],
                "linear__solver": ["lbfgs"],
                "linear__penalty": ["l2"],
            }

        elif self.prediction_task == "regression":  # regression
            head = Ridge()  # or LinearRegression()
            scoring = "neg_mean_squared_error"

            param_grid = {
                "pca__n_components": [10, 50, 100],
                "linear__alpha": [0.01, 0.1, 1, 10],  # Ridge regularization
            }
            
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("linear", head),
            ]
        )

        # Grid search object
        self.model = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

    def _dataloader_to_numpy(self, dataloader):
        features_list = []
        targets_list = []
        for features_batch, targets_batch, _ in dataloader:
            features_list.append(features_batch.numpy())
            targets_list.append(targets_batch.numpy())
        features = np.concatenate(features_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        return features, targets

    def fit(self, dataloader):
        """Fit PCA and then logistic regression on reduced features."""
        assert self.prediction_task in ["classification", "regression"], \
            f"prediction_task must be set before calling fit(). Got {self.prediction_task}"
        features, targets = self._dataloader_to_numpy(dataloader)
        features_reshaped = features.reshape(features.shape[0], -1)
        self.model.fit(features_reshaped, targets.flatten())
        # print("Best params found:", self.model.best_params_)

    def predict(self, dataloader):
        """Transform input with PCA and predict with logistic regression."""
        features, _ = self._dataloader_to_numpy(dataloader)
        features_reshaped = features.reshape(features.shape[0], -1)
        return self.model.predict(features_reshaped)

class LogisticRegressionModel(NumpyAbstractModel):
    """
    LogisticRegressionModel is a model that uses Logistic Regression for dimensionality reduction and classification.
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
    data_type = "array"
    prediction_task = None
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=100)

    def _dataloader_to_numpy(self, dataloader):
        features_list = []
        targets_list = []
        for features_batch, targets_batch, _ in dataloader:
            features_list.append(features_batch.numpy())
            targets_list.append(targets_batch.numpy())
        features = np.concatenate(features_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        return features, targets

    def fit(self, dataloader):
        """Fit logistic regression on reduced features."""
        features, targets = self._dataloader_to_numpy(dataloader)
        features_reshaped = features.reshape(features.shape[0], -1)
        self.model.fit(features_reshaped, targets.flatten())

    def predict(self, dataloader):
        """predict with logistic regression."""
        features, _ = self._dataloader_to_numpy(dataloader)
        features_reshaped = features.reshape(features.shape[0], -1)
        return self.model.predict(features_reshaped).reshape(-1, 1)
