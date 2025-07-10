import numpy as np
from collections import Counter


class DummyRegressor:
    def __init__(self):
        self.prediction_ = None

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
        self.prediction_ = np.mean(y)

    def predict(self, dataloader):
        # Convert input dataloader to numpy
        X, _ = self._dataloader_to_numpy(dataloader)

        return np.full((len(X),), self.prediction_)

class DummyClassifier:
    def __init__(self):
        self.class_ = None
    
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
        _, y = self._dataloader_to_numpy(dataloader)
        self.class_ = Counter(y.flatten()).most_common(1)[0][0]
        
    def predict(self, dataloader):
        # Convert input dataloader to numpy
        X, _ = self._dataloader_to_numpy(dataloader)

        return np.full((len(X),), self.class_)