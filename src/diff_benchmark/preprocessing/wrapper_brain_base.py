# base_pipeline.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


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


class DataPreparationBrain(ABC):
    """
    DataPreparationBrain is an abstract base class for preparing and analyzing brain data.
    Attributes:
        config (dict): Configuration settings for data preparation and analysis.
        results (dict): A dictionary to store results of the analysis.
    Methods:
        verify_subject_files(subject_id: str, metric: str) -> bool:
            Abstract method to verify the existence of required files for a subject.
        compute_microstructure(subject_id: str):
            Abstract method to compute microstructure data for a subject.
        run_analysis():
            Abstract method to execute the analysis on the prepared data.
        extract_features():
            Abstract method to extract features from the analyzed data.
        export_to_csv(output_path: Path):
            Exports the results to a CSV file at the specified output path.
        run_pipeline():
            Orchestrates the data preparation and analysis pipeline, ensuring all required files exist
            before running the analysis and exporting results to CSV.
    """

    def __init__(self, config: dict):
        self.config = config
        self.results = {}

    @abstractmethod
    def verify_raw_files(self, subject_id: str) -> bool:
        """
        Verifies the existence of raw files for a given subject ID.
        Args:
            subject_id (str): The unique identifier for the subject whose raw files are to be verified.
        Returns:
            bool: True if the raw files exist, False otherwise.
        """

    @abstractmethod
    def verify_subject_files(self, subject_id: str, metric: str) -> bool:
        """
        Verifies the existence and validity of subject files for a given subject ID and metric.
        Args:
            subject_id (str): The unique identifier for the subject whose files are to be verified.
            metric (str): The metric type that is being checked for the subject.
        Returns:
            bool: True if the subject files are valid and exist, False otherwise.
        """

    @abstractmethod
    def compute_microstructure(self, subject_id: str):
        """
        Compute the microstructure for a given subject.
        Parameters:
            subject_id (str): The unique identifier for the subject whose microstructure is to be computed.
        """

    @abstractmethod
    def run_analysis(self):
        """
        Executes the analysis process.
        This method is intended to be overridden in subclasses to implement
        specific analysis logic. Currently, it is a placeholder and does not
        perform any operations.
        """

    @abstractmethod
    def extract_features(self):
        """
        Extract features from the input data.
        This method is intended to be overridden in subclasses to implement
        specific feature extraction logic. It currently does not perform
        any operations and serves as a placeholder.
        """

    def export_to_csv(self) -> pd.DataFrame:
        """
        Exports the results to a CSV file.
        Parameters:
            output_path (Path): The file path where the CSV will be saved.
        Returns:
            DataFrame: A pandas DataFrame containing the exported results.
        """
        if not self.results:
            raise ValueError("No results to save.")
        df = pd.DataFrame.from_dict(self.results, orient="index")
        df.index.name = "subject_id"
        return df

    def run_pipeline(self, recompute: bool = False) -> pd.DataFrame:
        """
        Main orchestration: ensures all required files exist before running analysis.
        Args:
            recompute (bool): Whether to recompute microstructure even if files exist.
        Returns:
            DataFrame: A pandas DataFrame containing the exported results.
        """
        subject_list = sorted(
            [
                p.name
                for p in Path(self.config["data_paths"]["hcp_base"]).iterdir()
                if p.is_dir() and p.name.isdigit()
            ]
        )

        def process_subject(subject_id):
            """Processes a single subject by checking for required files"""
            # if not self.verify_subject_files(
            if self.verify_raw_files(subject_id):
                if (
                    self.verify_subject_files(
                        subject_id, self.config["metric_to_compute"]
                    )
                    and recompute
                ):
                    logger.info(f"[{subject_id}] Recomputing microstructure.")
                    self.compute_microstructure(subject_id)
                else:
                    logger.info(f"[{subject_id}] Computing microstructure.")
                    self.compute_microstructure(subject_id)
            # else:
            #     print(f"[{subject_id}] All required files found.")

        Parallel(n_jobs=50)(
            delayed(process_subject)(subject_id)
            for subject_id in tqdm(subject_list, desc="Processing subjects")
        )

        # Once all files are ready, run the analysis
        logger.info("All required files are ready. Now you can run analysis!")
        # self.run_analysis()
        # df = self.export_to_csv()
        # return df

    def run_microstructure_pipeline(self) -> pd.DataFrame:
        """
        Main orchestration: ensures all required files exist before running analysis.
        Returns:
            DataFrame: A pandas DataFrame containing the exported results.
        """
        logger.info(
            "All data should be preprocessed already. Getting microstructure files..."
        )
        self.run_analysis()
        df = self.export_to_csv()
        return df
