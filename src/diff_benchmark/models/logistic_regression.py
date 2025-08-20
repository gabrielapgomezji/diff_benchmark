import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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

    def __init__(self):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("logreg", LogisticRegression(max_iter=1000))
        ])

        # Grid of hyperparameters
        param_grid = {
            "pca__n_components": [50, 100, 400],   # number of PCA components to try
            "logreg__C": [0.01, 0.1, 1, 10, 100],   # inverse regularization strength
            "logreg__penalty": ["l2"],              # penalty (l1 requires saga solver)
            "logreg__solver": ["lbfgs"],            # stable solver for l2
        }

        # Grid search object
        self.model = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="accuracy",   # for binary classification
            cv=5,
            n_jobs=-1,
            verbose=1
        )

    def _dataloader_to_numpy(self, dataloader):
        X_list = []
        Y_list = []
        for x_batch, y_batch, _ in dataloader:
            X_list.append(x_batch.numpy())
            Y_list.append(y_batch.numpy())
        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)
        return X, Y

    def fit(self, dataloader):
        """Fit PCA and then logistic regression on reduced features."""
        X, y = self._dataloader_to_numpy(dataloader)
        X_reshaped = X.reshape(X.shape[0], -1)
        self.model.fit(X_reshaped, y.flatten())
        # print("Best params found:", self.model.best_params_)

    def predict(self, dataloader):
        """Transform input with PCA and predict with logistic regression."""
        X, _ = self._dataloader_to_numpy(dataloader)
        X_reshaped = X.reshape(X.shape[0], -1)
        return self.model.predict(X_reshaped)


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

    def __init__(self):
        self.model = LogisticRegression(max_iter=100)

    def _dataloader_to_numpy(self, dataloader):
        X_list = []
        Y_list = []
        for x_batch, y_batch, _ in dataloader:
            X_list.append(x_batch.numpy())
            Y_list.append(y_batch.numpy())
        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)
        return X, Y

    def fit(self, dataloader):
        """Fit logistic regression on reduced features."""
        X, y = self._dataloader_to_numpy(dataloader)
        X_reshaped = X.reshape(X.shape[0], -1)
        self.model.fit(X_reshaped, y.flatten())

    def predict(self, dataloader):
        """predict with logistic regression."""
        X, _ = self._dataloader_to_numpy(dataloader)
        X_reshaped = X.reshape(X.shape[0], -1)
        return self.model.predict(X_reshaped).reshape(-1, 1)
