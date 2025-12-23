from pathlib import Path
import pandas as pd
from typing import Iterable

COLUMN_ALIASES = {
    "Subject": {
        "subject",
        "participant_id",
        "participant",
        "sub_id",
        "sub",
    },
    "Age": {
        "age",
        "age_in_yrs",
        "age_in_years",
        "age_years",
        "ageyrs",
    },
    "Gender": {
        "gender",
        "sex",
        "gender_text",
    },
}


class DefaultDemographicsPreprocessor:
    """
    Unified demographics preprocessor.

    Supports:
    - Unicentre datasets (single CSV/TSV)
    - Multicentre datasets (directory with per-site subdirectories)

    Output:
    - One row per subject
    - Always returns a single DataFrame
    - Adds a 'Site' column automatically for multisite datasets
    """

    def __init__(
        self,
        path: str | Path | list[str | Path],
        site_column: str = "Site",
    ):
        self.is_multisite = isinstance(path, list)
        self.paths = self._normalize_paths(path)
        self.site_column = site_column

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def preprocess(self, target_columns: list[str]) -> pd.DataFrame:
        """
        Entry point used by the benchmark.
        Args:
            target_columns: List of target demographic columns to retain.
        Returns:
            Preprocessed demographics DataFrame.
        """
        df = self._load_all()
        df = self._filter(df, target_columns)
        df = self._normalize_subject_ids(df)
        df = self._categorical_to_numeric(df)
        df = df.dropna()
        return df

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------
    def _normalize_paths(
        self, path: str | Path | list[str | Path]
    ) -> list[Path]:
        """
        Normalize input paths into a list of Path objects.
        This method accepts a single path or a list of paths, where each path can 
        be a string or a Path object, and returns a list of Path objects.
        Args:
            path (str | Path | list[str | Path]): A single path as a string or Path 
                object, or a list of paths where each element is either a string 
                or a Path object.
        Returns:
            list[Path]: A list of Path objects.
        Raises:
            TypeError: If the input is not a string, Path object, or a list of 
                strings/Path objects.
        """
        if isinstance(path, (str, Path)):
            return [Path(path)]

        if isinstance(path, Iterable):
            return [Path(p) for p in path]

        raise TypeError(
            "path must be a str, Path, or list of str/Path"
        )

    def _normalize_subject_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Subject" in df.columns:
            df["Subject"] = (
                df["Subject"]
                .astype(str)
                .str.strip()
                .str.replace(r"^sub-", "", regex=True)
            )
        return df
    # ------------------------------------------------------------------
    # Loading logic
    # ------------------------------------------------------------------
    def _load_all(self) -> pd.DataFrame:
        """
        Load and concatenate data from multiple file paths into a single DataFrame.
        This method iterates over a list of file paths, loads the data from each path,
        and appends it to a list of DataFrames. If a site identifier is available for
        a given path, it is added as a new column to the DataFrame. Finally, all
        DataFrames are concatenated into a single DataFrame.
        Returns:
            pd.DataFrame: A concatenated DataFrame containing data from all specified paths.
        """
        
        dfs = []

        for p in self.paths:
            df, site = self._load_single_path(p)
            if site is not None:
                df[self.site_column] = site
            dfs.append(df)

        return pd.concat(dfs, axis=0, ignore_index=True)

    def _load_single_path(self, path: Path) -> tuple[pd.DataFrame, str | None]:
        """
        Load a single file or directory containing demographic data.
        This method handles both file and directory inputs. If the input is a file, 
        it loads the file directly. If the input is a directory, it expects exactly 
        one demographics file (with a `.tsv` or `.csv` extension) within the directory 
        and loads it. The method also determines the site name based on the input path 
        if the `is_multisite` attribute is set to True.
        Args:
            path (Path): The path to a file or directory containing demographic data.
        Returns:
            tuple[pd.DataFrame, str | None]: A tuple containing:
                - A pandas DataFrame with the loaded demographic data.
                - The site name as a string if `is_multisite` is True, otherwise None.
        Raises:
            ValueError: If the input is a directory and does not contain exactly one 
                        demographics file.
            FileNotFoundError: If the input path does not exist.
        """
        
        if path.is_file():
            site = path.parent.name if self.is_multisite else None
            return self._load_file(path), site

        if path.is_dir():
            demo_files = list(path.glob("*.tsv")) + list(path.glob("*.csv"))

            if len(demo_files) != 1:
                raise ValueError(
                    f"Expected exactly one demographics file in {path}, "
                    f"found {len(demo_files)}"
                )

            df = self._load_file(demo_files[0])
            site = path.name if self.is_multisite else None
            return df, site

        raise FileNotFoundError(path)

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a file into a pandas DataFrame.
        This method reads a file from the specified path and loads its content
        into a pandas DataFrame. The file format is determined by its extension:
        tab-separated values (TSV) for ".tsv" files and comma-separated values (CSV)
        for other file types.
        Args:
            file_path (Path): The path to the file to be loaded.
        Returns:
            pd.DataFrame: A pandas DataFrame containing the data from the file.
        """
        
        sep = "\t" if file_path.suffix == ".tsv" else ","
        return pd.read_csv(file_path, sep=sep) #, index_col=0)

    # ------------------------------------------------------------------
    # Preprocessing logic
    # ------------------------------------------------------------------
    def _filter(self, df: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
        """
        Filters and preprocesses a DataFrame based on specified target columns and predefined column aliases.
        Args:
            df (pd.DataFrame): The input DataFrame to be filtered and preprocessed.
            target_columns (list[str]): A list of target column names to retain in the DataFrame.
        Returns:
            pd.DataFrame: A filtered and preprocessed DataFrame containing the specified target columns,
                          along with additional columns such as "Subject", "Gender" (if present), and the site column
                          (if applicable).
        Notes:
            - If the DataFrame has an index name that matches any alias for "Subject", the index is reset.
            - Column names in the DataFrame are renamed to their canonical names based on the `COLUMN_ALIASES` mapping.
            - The resulting DataFrame will include the "Subject" column, the specified `target_columns`, and optionally
              "Gender" and the site column if they exist in the input DataFrame.
        """
        
        if (
            df.index.name
            and df.index.name.lower() in COLUMN_ALIASES["Subject"]
        ):
            df = df.reset_index()

        df = df.rename(
            columns={
                c: canonical
                for canonical, aliases in COLUMN_ALIASES.items()
                for c in df.columns
                if c.lower() in aliases
            }
        )

        columns = ["Subject"] + target_columns

        if "Gender" not in columns and "Gender" in df.columns:
            columns.append("Gender")

        if self.site_column in df.columns:
            columns.append(self.site_column)

        df = df.loc[:, [c for c in columns if c in df.columns]]
        return df

    def _categorical_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts categorical columns in the DataFrame to numeric values.
        This method specifically checks for a "Gender" column in the DataFrame.
        If the column exists and its data type is object, it maps gender values
        to numeric representations:
            - "M" or "MALE" -> 1
            - "F" or "FEMALE" -> 0
        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data to be processed.
        Returns:
            pd.DataFrame: The DataFrame with the "Gender" column converted to numeric values,
                          if applicable. Other columns remain unchanged.
        """
        
        if "Gender" in df.columns and df["Gender"].dtype == object:
            df["Gender"] = (
                df["Gender"]
                .astype(str)
                .str.upper()
                .map({"M": 1, "F": 0, "MALE": 1, "FEMALE": 0})
            )
        return df
