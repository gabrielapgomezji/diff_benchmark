from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DemographicsPreprocessor(ABC):
    """
    DemographicsPreprocessor is an abstract base class for preprocessing demographic data.
    Attributes:
        csv_path (Path): The path to the CSV file containing demographic data.
        df (DataFrame): The DataFrame loaded from the CSV file.
    Methods:
        filter(target_columns: list[str]) -> None:
            Abstract method to filter the DataFrame based on the specified target columns.
        categorical_to_numeric() -> None:
            Abstract method to convert categorical variables in the DataFrame to numeric format.
        clean_df() -> None:
            Abstract method to clean the DataFrame by handling missing values and other inconsistencies.
        preprocess(target_columns: list[str]) -> pd.DataFrame:
            Preprocesses the demographic data by applying filtering, conversion, and cleaning steps.
            Returns the processed DataFrame.
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path, index_col=0)

    @abstractmethod
    def filter(self, target_columns: list[str]) -> None:
        """
        Filters the demographic data based on the specified target columns.
        Args:
            target_columns (list[str]): A list of column names to filter the demographic data.
        Returns:
            None
        """

    @abstractmethod
    def categorical_to_numeric(self) -> None:
        """
        Converts categorical variables in the dataset to numeric format.
        This method processes the categorical data within the class's dataset,
        transforming each categorical feature into a corresponding numeric representation.
        This is typically done to prepare the data for machine learning algorithms that
        require numerical input.
        Returns:
            None: This method modifies the dataset in place and does not return any value.
        """

    @abstractmethod
    def clean_df(self) -> None:
        """
        Cleans the demographic data DataFrame.
        This method is responsible for preprocessing the demographic data by
        removing any inconsistencies, handling missing values, and ensuring
        that the data is in a suitable format for further analysis.
        It modifies the DataFrame in place and does not return any value.
        """

    def preprocess(self, target_columns: list[str]) -> pd.DataFrame:
        """
        Preprocess the demographic data by applying a series of transformations.
        This method filters the DataFrame based on the specified target columns,
        converts categorical variables to numeric representations, and cleans the DataFrame.
        Args:
            target_columns (list[str]): A list of column names to filter the DataFrame.
        Returns:
            pd.DataFrame: The processed DataFrame after applying the transformations.
        """

        self.filter(target_columns)
        self.categorical_to_numeric()
        self.clean_df()
        # self.df is a dataframe with the cognitive columns defined in config + Subject + Gender
        return self.df
