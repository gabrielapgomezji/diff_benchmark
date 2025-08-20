# import pandas as pd

from diff_benchmark.preprocessing.base_demographic_data import DemographicsPreprocessor


class DefaultDemographicsPreprocessor(DemographicsPreprocessor):
    """
    DefaultDemographicsPreprocessor is a class that extends the DemographicsPreprocessor
    to preprocess demographic data specifically for a dataset.
    Methods:
        filter(target_columns: list[str]) -> None:
            Filters the DataFrame to include only the specified target columns,
            ensuring that "Subject" and "Gender" are included if available.
        categorical_to_numeric() -> None:
            Converts the "Gender" column from categorical values ("M", "F") to numeric
            values (1 for "M" and 0 for "F") if the column exists and is of object type.
        clean_df() -> None:
            Cleans the DataFrame by removing any rows with missing values.
    """

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
        
    # def gender_stratification(self) -> None:
    #     # This method is not implemented in the base class, but can be overridden if needed.
    #     if "Gender" in self.df.columns:
    #         columns.append("Gender")
