from abc import ABC, abstractmethod
from pathlib import Path

from joblib import Parallel, delayed


class RawDataProcessor(ABC):
    """
    RawDataProcessor is an abstract base class for processing raw data related to subjects in a dataset.
    Attributes:
        config (dict): Configuration dictionary containing settings for processing.
    Methods:
        save_subject_info(sub: str):
            Save metadata or summary info about a subject.
        save_dataset_info():
            Save metadata about the dataset as a whole.
        project_dwi_to_cortex(sub: str):
            Project the DWI data to the cortical surface and save it.
        run(sub: str):
            Entry point for full raw data processing for a subject.
        run_parallel():
            Process multiple subjects in parallel based on the configuration.
    """

    def __init__(self, config: dict):
        self.config = config

    # @abstractmethod
    # def check_required_files(self, sub: SubjectInfo) -> bool:
    #     """Check if all required files for processing exist."""
    #     pass

    @abstractmethod
    def save_subject_info(self, sub: str):
        """Save metadata or summary info about a subject."""
        pass

    @abstractmethod
    def save_dataset_info(self):
        """Save metadata about the dataset as a whole."""
        pass

    @abstractmethod
    def project_dwi_to_cortex(self, sub: str):
        """Project the DWI data to the cortical surface and save it."""
        pass

    def run(self, sub: str):
        """Entry point for full raw data processing for a subject."""
        # if self.check_required_files(sub):
        self.project_dwi_to_cortex(sub)
        # self.save_subject_info(sub)

    def run_parallel(self):
        base_folder = Path(self.config["base_path"])
        subjects = [p.name for p in base_folder.iterdir() if p.is_dir()]

        Parallel(n_jobs=self.config["n_jobs"])(
            delayed(self.run)(sub) for sub in subjects
        )
