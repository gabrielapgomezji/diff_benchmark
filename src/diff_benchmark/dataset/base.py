from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

class DatasetBuilder(ABC):
    def __init__(self, base_path: Path, h5_filename: str, output_dataset_filename: str):
        self.base_path = Path(base_path)
        self.h5_filename = h5_filename
        self.output_dataset_filename = output_dataset_filename

    @abstractmethod
    def verify_files(self, subject_dir: Path) -> bool:
        pass
    
    @abstractmethod
    def filter_features(self, features: np.ndarray) -> bool:
        """Filter features based on some criteria or pass."""
        pass

    @abstractmethod
    def extract_information(self, subject_dir: Path):
        pass

    @abstractmethod
    def save_dataset(self, X, y, genders):
        pass

    @abstractmethod
    def create_dataset(self):
        pass

class DataLoadingStrategy(ABC):
    @abstractmethod
    def load_data(self, h5_path: Path):
        """Load embeddings, power, metadata, etc., from the given HDF5 file."""
        pass

    @abstractmethod
    def is_valid(self, data: dict) -> bool:
        """Check if the loaded data is valid."""
        pass

    @abstractmethod
    def to_features(self, data: dict) -> np.ndarray:
        """Convert the data dict into a feature vector."""
        pass
