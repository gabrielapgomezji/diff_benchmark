from abc import ABC, abstractmethod
from pathlib import Path
from joblib import Parallel, delayed


from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ProcessingResult:
    valid_subjects: list[str] = field(default_factory=list)
    invalid_subjects: list[str] = field(default_factory=list)

    def add_valid(self, subject_id: str):
        self.valid_subjects.append(subject_id)

    def add_invalid(self, subject_id: str):
        self.invalid_subjects.append(subject_id)


class BrainDataPreprocessor(ABC):
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
        output_base = Path(self.config["results_path_2"])
        subjects = [p.name for p in base_path.iterdir() if p.is_dir()]
        n_jobs = self.config.get("n_jobs", -1)
        
        valid_subjects = []

        for sub_id in subjects:
            raw_path = base_path / sub_id / "raw_surface_data.h5"
            if not raw_path.exists():
                print(f"Skipping {sub_id} — raw data file not found at {raw_path}")
                self.processing_result.add_invalid(sub_id)
            else:
                valid_subjects.append(sub_id)

        def process_single_subject(sub_id):
            raw_path = base_path / sub_id / "raw_surface_data.h5"
            save_path = output_base / sub_id / "processed"
            
            try:
                self.preprocess_subject(raw_path, sub_id, save_path)
                self.processing_result.add_valid(sub_id)
            except Exception as e:
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
