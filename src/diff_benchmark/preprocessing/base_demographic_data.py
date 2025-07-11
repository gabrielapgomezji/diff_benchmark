from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class DemographicsPreprocessor(ABC):
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path, index_col=0)

    @abstractmethod
    def filter(self, target_columns: list[str]) -> None:
        pass

    @abstractmethod
    def categorical_to_numeric(self) -> None:
        pass

    @abstractmethod
    def clean_df(self) -> None:
        pass

    def preprocess(self, target_columns: list[str]) -> pd.DataFrame:
        self.filter(target_columns)
        self.categorical_to_numeric()
        self.clean_df()
        return self.df
