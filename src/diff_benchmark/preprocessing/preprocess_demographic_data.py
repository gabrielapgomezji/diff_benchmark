import pandas as pd
from diff_benchmark.preprocessing.base_demographic_data import DemographicsPreprocessor

class DefaultDemographicsPreprocessor(DemographicsPreprocessor):
    def filter(self, target_columns: list[str]) -> None:
        # Always include "Subject" and "Gender" if available
        columns = ["Subject"] + target_columns
        if "Gender" not in columns and "Gender" in self.df.columns:
            columns.append("Gender")
        self.df = self.df.loc[:, [col for col in columns if col in self.df.columns]]

    def categorical_to_numeric(self) -> None:
        if "Gender" in self.df.columns and self.df["Gender"].dtype == object:
            self.df["Gender"] = self.df["Gender"].map({"M": 1, "F": 0})

    def clean_df(self) -> None:
        self.df = self.df.dropna()
