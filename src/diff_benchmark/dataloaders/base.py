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
