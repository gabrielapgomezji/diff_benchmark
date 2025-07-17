from abc import ABC, abstractmethod


class NumpyAbstractModel(ABC):
    def __init__(self):
        pass
        """
        Abstract base class for all models in the diff_benchmark framework.
        Defines the interface that all models must implement.
        """

    @abstractmethod
    def _dataloader_to_numpy(self, dataloader):
        """
        Convert a DataLoader to numpy arrays.
        This method should be implemented by all subclasses to handle the conversion.
        """
        pass

    @abstractmethod
    def fit(self, dataloader):
        """
        Fit the model to the training data.
        """
        pass

    @abstractmethod
    def predict(self, dataloader):
        """
        Predict using the fitted model.
        """
        pass


class TorchAbstractModel(ABC):
    """
    Abstract base class for all models in the diff_benchmark framework.
    Defines the interface that all models must implement.
    """

    def __init__(self):
        pass

    @abstractmethod
    def _dataloader_to_numpy(self, dataloader):
        """
        Convert a DataLoader to numpy arrays.
        This method should be implemented by all subclasses to handle the conversion.
        """
        pass

    @abstractmethod
    def fit(self, dataloader):
        """
        Fit the model to the training data.
        """
        pass

    @abstractmethod
    def predict(self, dataloader):
        """
        Predict using the fitted model.
        """
        pass
