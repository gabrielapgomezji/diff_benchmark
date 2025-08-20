from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from joblib import Parallel, delayed


@dataclass
class ProcessingResult:
    """
    A class to store and manage the results of processing subjects in a dataset.
    Attributes:
        valid_subjects (list[str]): A list of subject IDs that are considered valid.
        invalid_subjects (list[str]): A list of subject IDs that are considered invalid.
    Methods:
        add_valid(subject_id: str): Adds a subject ID to the list of valid subjects.
        add_invalid(subject_id: str): Adds a subject ID to the list of invalid subjects.
    """

    valid_subjects: list[str] = field(default_factory=list)
    invalid_subjects: list[str] = field(default_factory=list)

    def add_valid(self, subject_id: str):
        """
        Adds a valid subject ID to the list of valid subjects.
        Args:
            subject_id (str): The ID of the subject to be added as valid.
        """

        self.valid_subjects.append(subject_id)

    def add_invalid(self, subject_id: str):
        """
        Adds a subject ID to the list of invalid subjects.
        Parameters:
            subject_id (str): The ID of the subject to be marked as invalid.
        """

        self.invalid_subjects.append(subject_id)


class BrainDataPreprocessor(ABC):
    """
    BrainDataPreprocessor is an abstract base class for preprocessing brain data.
    Attributes:
        config (dict): Configuration settings for the preprocessing.
        processing_result (ProcessingResult): Object to store the results of the processing.
    Methods:
        preprocess_subject(raw_data_path: Path, subject_id: str, save_path: Path):
            Abstract method to preprocess brain data for a single subject.
        preprocess_dataset():
            Preprocess brain data for the entire dataset, handling multiple subjects in parallel.
        save_subject_info(subject_id: str):
            Abstract method to save metadata/info for a single subject.
        save_dataset_info():
            Abstract method to save metadata/info for the full dataset.
    """

    def __init__(self, config: dict):
        self.config = config
        self.processing_result = ProcessingResult()

    @abstractmethod
    def preprocess_subject(self, raw_data_path: Path, subject_id: str, save_path: Path):
        """Preprocess brain data for a single subject."""
        pass

    # @abstractmethod
    def preprocess_dataset(self):
        """Preprocess brain data for the entire dataset."""
        base_path = Path(self.config["results_path"])
        output_base = Path(self.config["results_path"])
        subjects = [p.name for p in base_path.iterdir() if p.is_dir()]
        n_jobs = self.config.get("n_jobs", -1)

        valid_subjects = []

        for sub_id in subjects:
            if not sub_id.isdigit():
                print(f"Skipping {sub_id} — not a numeric subject ID")
            else:
                raw_path = base_path / sub_id / "raw_surface_data2.h5"
                if not raw_path.exists():
                    print(f"Skipping {sub_id} — raw data file not found at {raw_path}")
                    # self.processing_result.add_invalid(sub_id)
                    valid_subjects.append(sub_id)
                else:
                    # valid_subjects.append(sub_id)
                    self.processing_result.add_invalid(sub_id)

        def process_single_subject(sub_id):
            raw_path = base_path / sub_id / "raw_surface_data.h5"
            save_path = output_base / sub_id / "processed"

            try:
                self.preprocess_subject(raw_path, sub_id, save_path)
                self.processing_result.add_valid(sub_id)
            except Exception as _:
                self.processing_result.add_invalid(sub_id)

        Parallel(n_jobs=n_jobs)(
            delayed(process_single_subject)(sub_id) for sub_id in valid_subjects
        )
        pass

    @abstractmethod
    def save_subject_info(self, subject_id: str):
        """Save metadata/info for a single subject."""
        pass

    @abstractmethod
    def save_dataset_info(self):
        """Save metadata/info for the full dataset."""
        pass
