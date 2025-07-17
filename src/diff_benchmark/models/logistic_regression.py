import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from diff_benchmark.models.base import NumpyAbstractModel


class PCALogisticRegressionModel(NumpyAbstractModel):
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
