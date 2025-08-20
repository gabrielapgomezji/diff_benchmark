# base_pipeline.py
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

class DataPreparationBrain(ABC):

    def __init__(self, config: dict):
        self.config = config
        self.results = {}
    
    @abstractmethod
    def verify_subject_files(self, subject_id: str, metric: str) -> bool:
        pass

        
    @abstractmethod
    def compute_microstructure(self, subject_id: str):
        pass

    @abstractmethod
    def run_analysis(self):
        pass

    @abstractmethod
    def extract_features(self):
        pass

    def export_to_csv(self, output_path: Path):
        if not self.results:
            raise ValueError("No results to save.")
        df = pd.DataFrame.from_dict(self.results, orient="index")
        df.index.name = "subject_id"
        df.to_csv(output_path)
        return df

    def run_pipeline(self):
        """
        Main orchestration: ensures all required files exist before running analysis.
        """
        subject_list = sorted([
            p.name for p in Path(self.config["base_path"]).iterdir()
            if p.is_dir() and p.name.isdigit()
        ])
        
        def process_subject(subject_id):
            if not self.verify_subject_files(subject_id, self.config["metric_to_compute"]):
                print(f"[{subject_id}] Missing files — computing microstructure.")
                self.compute_microstructure(subject_id)
            else:
                print(f"[{subject_id}] All required files found.")
        Parallel(n_jobs=20)(
            delayed(process_subject)(subject_id)
            for subject_id in tqdm(subject_list, desc="Processing subjects")
        )

        # Once all files are ready, run the analysis
        self.run_analysis()
        df = self.export_to_csv(
            Path(self.config["results_path"]) / "results.csv"
        )
        return df