import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

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

    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
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
        """Fit PCA and then logistic regression on reduced features."""
        X, y = self._dataloader_to_numpy(dataloader)
        X_reshaped = X.reshape(X.shape[0], -1)
        X_reduced = self.pca.fit_transform(X_reshaped)
        self.model.fit(X_reduced, y.flatten())

    def predict(self, dataloader):
        """Transform input with PCA and predict with logistic regression."""
        X, _ = self._dataloader_to_numpy(dataloader)
        X_reshaped = X.reshape(X.shape[0], -1)
        X_reduced = self.pca.transform(X_reshaped)
        return self.model.predict(X_reduced).reshape(-1, 1)
