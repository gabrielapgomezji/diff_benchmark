import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# import torch


class CanonicalCorrelationRegressor:
    def __init__(self, n_components=10, ridge_alpha=1.0):
        self.n_components = n_components
        self.ridge_alpha = ridge_alpha

        self.cca = CCA(n_components=n_components)
        self.scaler_y = StandardScaler()
        self.regressor = Ridge(alpha=self.ridge_alpha)

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
        # Convert DataLoaders to numpy arrays
        X, y = self._dataloader_to_numpy(dataloader)

        # Standardize target view y (X will be standardized by CCA)
        # X_std = self.scaler_X.fit_transform(X)
        y_std = self.scaler_y.fit_transform(y)

        # Fit CCA on both views
        self.cca.fit(X, y_std)

        # Project both views to latent space
        Z1, Z2 = self.cca.transform(X, y_std)

        # Fit regression in latent space
        self.regressor.fit(Z1, Z2)

    def predict(self, dataloader):
        # Convert input dataloader to numpy
        X, _ = self._dataloader_to_numpy(dataloader)

        # Project X to latent space
        Z1 = self.cca.transform(X)

        # Predict in latent space
        Z2_pred = self.regressor.predict(Z1)

        # Reconstruct original y
        _, y_pred_std = self.cca.inverse_transform(Z1, Z2_pred)
        y_pred = self.scaler_y.inverse_transform(y_pred_std)

        return y_pred

    def score(self, dataloader):
        _, y_true = self._dataloader_to_numpy(dataloader)
        y_pred = self.predict(dataloader)
        mse = np.mean((y_true - y_pred) ** 2)
        return mse  # return negative MSE
