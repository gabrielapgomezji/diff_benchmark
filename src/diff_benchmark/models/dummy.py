from collections import Counter

import numpy as np

from diff_benchmark.models.base import NumpyAbstractModel
from torch.utils.data import DataLoader


class DummyRegressor(NumpyAbstractModel):
    """
    DummyRegressor is a simple regression model that predicts the mean of the target values
    from the training data. It inherits from NumpyAbstractModel.
    Methods:
        - __init__: Initializes the DummyRegressor instance and sets the prediction attribute to None.
        - _dataloader_to_numpy: Converts a dataloader containing batches of data into numpy arrays.
            Args:
                dataloader: A data loader that yields batches of (input, target, _) tuples.
            Returns:
                Tuple of numpy arrays (X, Y) where X is the input data and Y is the target data.
        - fit: Fits the model to the provided dataloader by calculating the mean of the target values.
            Args:
                dataloader: A data loader containing batches of data for training.
        - predict: Predicts target values for the given dataloader by returning the mean prediction.
            Args:
                dataloader: A data loader containing batches of input data for prediction.
            Returns:
                A numpy array filled with the mean prediction value for each input sample.
    """

    data_type = "array"
    prediction_task = None

    def __init__(self):
        self.prediction_ = None

    def _dataloader_to_numpy(self, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts a dataloader containing batches of data into NumPy arrays.
        Args:
            dataloader (iterable): An iterable that yields batches of data in the form
                                   (x_batch, y_batch, _), where x_batch and y_batch
                                   are the input and target tensors, respectively.
        Returns:
            tuple: A tuple containing two NumPy arrays:
                - features (np.ndarray): Concatenated array of input data.
                - targets (np.ndarray): Concatenated array of target data.
        """

        features_list = []
        targets_list = []
        for features_batch, targets_batch, _ in dataloader:
            features_list.append(features_batch.numpy())
            targets_list.append(targets_batch.numpy())
        features = np.concatenate(features_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        return features, targets

    def fit(self, dataloader: DataLoader):
        """
        Fit the model using the provided dataloader.
        This method converts the data from the dataloader into numpy arrays and computes
        the mean of the target values. The computed mean is stored as the model's prediction.
        Parameters:
            dataloader (DataLoader): A DataLoader object containing the input data and target values.
        Returns:
            None
        """

        # Convert DataLoaders to numpy arrays
        _, targets = self._dataloader_to_numpy(dataloader)
        self.prediction_ = np.mean(targets)

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """
        Predicts the output for the given dataloader.
        Args:
            dataloader (DataLoader): A PyTorch DataLoader object containing the input data.
        Returns:
            numpy.ndarray: An array filled with the predicted value for each input sample.
        """

        # Convert input dataloader to numpy
        features, _ = self._dataloader_to_numpy(dataloader)

        return np.full((len(features),), self.prediction_)


class DummyClassifier(NumpyAbstractModel):
    """
    DummyClassifier is a simple classifier that predicts the most common class
    from the training data. It inherits from NumpyAbstractModel.
    Methods
    -------
    - __init__(): Initializes the DummyClassifier and sets the class_ attribute to None.
    - _dataloader_to_numpy(dataloader): Converts the input dataloader into numpy arrays for features and labels.
    - fit(dataloader): Fits the model to the data by determining the most common class in the labels.
    - predict(dataloader): Predicts the class for the input data by returning the most common class for all samples.
    """

    data_type = "array"
    prediction_task = None

    def __init__(self):
        self.class_ = None

    def _dataloader_to_numpy(self, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts a dataloader containing batches of data into NumPy arrays.
        Args:
            dataloader (iterable): An iterable that yields batches of data in the form
                                   (x_batch, y_batch, _), where x_batch and y_batch
                                   are the input and target tensors, respectively.
        Returns:
            tuple: A tuple containing two NumPy arrays:
                - X (np.ndarray): Concatenated array of input data.
                - Y (np.ndarray): Concatenated array of target data.
        """
        features_list = []
        targets_list = []
        for features_batch, targets_batch, _ in dataloader:
            features_list.append(features_batch.numpy())
            targets_list.append(targets_batch.numpy())
        features = np.concatenate(features_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        return features, targets

    def fit(self, dataloader: DataLoader):
        """
        Fit the model using the provided dataloader.
        This method converts the data from the dataloader into numpy arrays and computes
        the mean of the target values. The computed mean is stored as the model's prediction.
        Parameters:
            dataloader (DataLoader): A DataLoader object containing the input data and target values.
        Returns:
            None
        """
        # Convert DataLoaders to numpy arrays
        _, targets = self._dataloader_to_numpy(dataloader)
        self.class_ = Counter(targets.flatten()).most_common(1)[0][0]

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """
        Predicts the output for the given dataloader.
        Args:
            dataloader (DataLoader): A PyTorch DataLoader object containing the input data.
        Returns:
            numpy.ndarray: An array filled with the predicted value for each input sample.
        """
        # Convert input dataloader to numpy
        features, _ = self._dataloader_to_numpy(dataloader)

        return np.full((len(features),), self.class_)
