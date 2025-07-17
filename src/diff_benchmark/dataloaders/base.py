# from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DatasetSpecs:
    """
    DatasetSpecs is a class that holds specifications for a dataset.
    Attributes:
        num_samples (int): The total number of samples in the dataset.
        num_features (int): The number of features for each sample.
        num_targets (int): The number of target variables for each sample.
        gender_distribution (dict): A dictionary representing the distribution of genders in the dataset.
    """

    num_samples: int
    num_features: int
    num_targets: int
    gender_distribution: dict


# class AbstractPreprocessedData(ABC):

#     def __init__(self, X, y, genders, n_splits=5):
#         self.X = X
#         self.y = y
#         self.genders = genders
#         self.n_splits = n_splits

#     @abstractmethod
#     def get_fold_indices(self):
#         pass

#     @abstractmethod
#     def get_dataloader_fold(self):
#         pass

#     @abstractmethod
#     def get_arrays_from_indices(self):
#         pass

#     @abstractmethod
#     def get_folds_as_dataloaders(self):
#         pass

#     @abstractmethod
#     def get_folds_as_arrays(self):
#         pass

#     @abstractmethod
#     def get_specs(self) -> DatasetSpecs:
#         pass
