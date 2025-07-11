from abc import ABC, abstractmethod
from pathlib import Path

class BrainDataPreprocessor(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def preprocess_subject(self, raw_data_path: Path, subject_id: str, save_path: Path):
        """Preprocess brain data for a single subject."""
        pass

    @abstractmethod
    def preprocess_dataset(self):
        """Preprocess brain data for the entire dataset."""
        pass

    @abstractmethod
    def save_subject_info(self, subject_id: str):
        """Save metadata/info for a single subject."""
        pass

    @abstractmethod
    def save_dataset_info(self):
        """Save metadata/info for the full dataset."""
        pass
