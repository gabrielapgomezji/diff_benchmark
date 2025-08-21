import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# import torch


class CanonicalCorrelationRegressor:
    """
    CanonicalCorrelationRegressor is a regression model that utilizes Canonical Correlation Analysis (CCA)
    to learn the relationship between two sets of variables. It projects the data into a latent space
    where a regression model is then fitted.
    Attributes:
        n_components (int): The number of components to use for CCA.
        ridge_alpha (float): The regularization strength for the Ridge regression.
        cca (CCA): An instance of the CCA model.
        scaler_targets (StandardScaler): A scaler for standardizing the target variable.
        regressor (Ridge): An instance of the Ridge regression model.
    Methods:
        _dataloader_to_numpy(dataloader):
            Converts a PyTorch DataLoader into numpy arrays for features and targets.
        fit(dataloader):
            Fits the CanonicalCorrelationRegressor model to the data provided by the DataLoader.
        predict(dataloader):
            Predicts the target variable for the given DataLoader using the fitted model.
    """

    def __init__(self, n_components=10, ridge_alpha=1.0):
        self.n_components = n_components
        self.ridge_alpha = ridge_alpha

        self.cca = CCA(n_components=n_components)
        self.scaler_targets = StandardScaler()
        self.regressor = Ridge(alpha=self.ridge_alpha)

    def _dataloader_to_numpy(self, dataloader):
        """
        Converts a PyTorch DataLoader containing batches of data into NumPy arrays.
        Args:
            dataloader (torch.utils.data.DataLoader): The DataLoader object that yields batches of data.
        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing two NumPy arrays:
                - features: Concatenated array of input features from all batches.
                - targets: Concatenated array of target labels from all batches.
        """

        features_list = []
        targets_list = []
        for features_batch, targets_batch, _ in dataloader:
            features_list.append(features_batch.numpy())
            targets_list.append(targets_batch.numpy())
        features = np.concatenate(features_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        return features, targets

    def fit(self, dataloader):
        """
        Fit the CCA model to the provided dataloader.
        This method converts the data from the dataloader into numpy arrays,
        standardizes the target view, fits the CCA model on both views, and
        projects the data into a latent space. Finally, it fits a regression
        model in the latent space.
        Parameters:
            dataloader (DataLoader): A PyTorch DataLoader containing the input data.
        Returns:
            None
        """

        # Convert DataLoaders to numpy arrays
        features, targets = self._dataloader_to_numpy(dataloader)

        # Standardize target view y (X will be standardized by CCA)
        # X_std = self.scaler_X.fit_transform(X)
        targets_std = self.scaler_targets.fit_transform(targets)

        # Fit CCA on both views
        self.cca.fit(features, targets_std)

        # Project both views to latent space
        embed_features, embed_targets = self.cca.transform(features, targets_std)

        # Fit regression in latent space
        self.regressor.fit(embed_features, embed_targets)

    def predict(self, dataloader):
        """
        Predicts the output based on the input dataloader.
        This method takes a dataloader as input, converts it to a numpy array,
        projects the data into a latent space using Canonical Correlation Analysis (CCA),
        and then uses a regressor to predict the output in that latent space.
        Finally, it reconstructs the original output using the inverse transformation
        of the CCA and scales it back to the original range.
        Parameters:
            dataloader (DataLoader): A dataloader containing the input data.
        Returns:
            numpy.ndarray: The predicted output after reconstruction and scaling.
        """

        # Convert input dataloader to numpy
        features, _ = self._dataloader_to_numpy(dataloader)

        # Project X to latent space
        embed_features = self.cca.transform(features)

        # Predict in latent space
        embed_targets_pred = self.regressor.predict(embed_features)

        # Reconstruct original y
        _, targets_pred_std = self.cca.inverse_transform(
            embed_features, embed_targets_pred
        )
        targets_pred = self.scaler_targets.inverse_transform(targets_pred_std)

        return targets_pred

    # SHOULD NOT BE INSIDE THE MODEL
    # def score(self, dataloader):
    #     _, y_true = self._dataloader_to_numpy(dataloader)
    #     y_pred = self.predict(dataloader)
    #     mse = np.mean((y_true - y_pred) ** 2)
    #     return mse  # return negative MSE
