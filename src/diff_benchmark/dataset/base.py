from abc import ABC, abstractmethod
from pathlib import Path

class AbstractDatasetBuilder(ABC):
    def __init__(self, base_path: Path, h5_filename: str, output_dataset_filename: str):
        self.base_path = Path(base_path)
        self.h5_filename = h5_filename
        self.output_dataset_filename = output_dataset_filename

    @abstractmethod
    def verify_files(self, subject_dir: Path) -> bool:
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
