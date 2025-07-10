from pathlib import Path

import numpy as np
import pandas as pd


def preprocess_csv(csv_path: str, target_columns: np.ndarray):
    """
    Preprocesses a CSV file by loading it into a DataFrame and filtering rows based on the specified target columns.
    Args:
        csv_path (str or Path): The file path to the CSV file to be processed.
        target_columns (np.ndarray): An array of column names or indices to filter the rows in the DataFrame.
    Returns:
        pd.DataFrame: A DataFrame containing only the rows corresponding to the specified target columns.
    """

    csv_path = Path(csv_path)
    df_targets = pd.read_csv(csv_path, index_col=0)
    df_targets = df_targets.loc[:, ["Subject"] + target_columns]

    if "Gender" in df_targets.columns:
        df_targets["Gender"] = df_targets["Gender"].map({"M": 1, "F": 0})

    df_targets = df_targets.dropna(subset=target_columns.tolist())
    
    return df_targets
