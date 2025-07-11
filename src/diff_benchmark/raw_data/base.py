from abc import ABC, abstractmethod

class RawDataProcessor(ABC):
    def __init__(self, config: dict):
        self.config = config

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
        self.save_subject_info(sub)
        self.project_dwi_to_cortex(sub)