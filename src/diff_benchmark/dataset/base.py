from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class DatasetBuilder(ABC):
    """
    DatasetBuilder is an abstract base class for building datasets.
    Attributes:
        base_path (Path): The base path for the dataset.
        h5_filename (str): The name of the HDF5 file.
        output_dataset_filename (str): The name of the output dataset file.
    Methods:
        verify_files(subject_dir: Path) -> bool:
            Verifies the existence and validity of files in the specified subject directory.
        filter_features(features: np.ndarray) -> bool:
            Filters features based on specific criteria.
        extract_information(subject_dir: Path):
            Extracts relevant information from the specified subject directory.
        save_dataset(X, y, genders):
            Saves the dataset with the provided features, labels, and gender information.
        create_dataset():
            Creates the dataset by orchestrating the necessary steps.
    """

    def __init__(self, base_path: Path, h5_filename: str, output_dataset_filename: str):
        self.base_path = Path(base_path)
        self.h5_filename = h5_filename
        self.output_dataset_filename = output_dataset_filename

    @abstractmethod
    def verify_files(self, subject_dir: Path) -> bool:
        """Verify the existence and validity of files in the specified subject directory."""

    @abstractmethod
    def filter_features(self, features: np.ndarray) -> bool:
        """Filter features based on some criteria or pass."""

    @abstractmethod
    def extract_information(self, subject_dir: Path):
        """Extract relevant information from the specified subject directory."""

    @abstractmethod
    def save_dataset(self, features, targets, genders):
        """Save dataset function."""

    @abstractmethod
    def create_dataset(self):
        """Create the dataset by orchestrating the necessary steps."""


class DataLoadingStrategy(ABC):
    """
    Abstract base class for data loading strategies.
    This class defines the interface for loading data from HDF5 files, validating the loaded data,
    and converting the data into feature vectors. Subclasses must implement the following methods:
    - load_data(h5_path: Path): Load embeddings, power, metadata, etc., from the given HDF5 file.
    - is_valid(data: dict) -> bool: Check if the loaded data is valid.
    - to_features(data: dict) -> np.ndarray: Convert the data dict into a feature vector.
    """

    @abstractmethod
    def load_data(self, h5_path: Path):
        """Load embeddings, power, metadata, etc., from the given HDF5 file."""

    @abstractmethod
    def is_valid(self, data: dict) -> bool:
        """Check if the loaded data is valid."""

    @abstractmethod
    def to_features(self, data: dict) -> np.ndarray:
        """Convert the data dict into a feature vector."""
