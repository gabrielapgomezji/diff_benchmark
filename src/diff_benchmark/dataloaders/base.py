from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DatasetSpecs:
    num_samples: int
    num_features: int
    num_targets: int
    gender_distribution: dict

class AbstractPreprocessedData(ABC):

    def __init__(self, X, y, genders, n_splits=5):
        self.X = X
        self.y = y
        self.genders = genders
        self.n_splits = n_splits
    @abstractmethod
    def get_fold_indices(self):
        pass
    
    @abstractmethod
    def get_dataloader_fold(self):
        pass
    
    @abstractmethod
    def get_arrays_from_indices(self):
        pass
    
    @abstractmethod
    def get_folds_as_dataloaders(self):
        pass

    @abstractmethod
    def get_folds_as_arrays(self):
        pass

    @abstractmethod
    def get_specs(self) -> DatasetSpecs:
        pass
