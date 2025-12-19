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
        self.df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def preprocess(self, target_columns: list[str]) -> pd.DataFrame:
        """
        Entry point used by the benchmark.
        """
        self.df = self._load_all()
        self._filter(target_columns)
        self._categorical_to_numeric()
        self._clean_df()
        return self.df

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------
    def _normalize_paths(
        self, path: str | Path | list[str | Path]
    ) -> list[Path]:
        if isinstance(path, (str, Path)):
            return [Path(path)]

        if isinstance(path, Iterable):
            return [Path(p) for p in path]

        raise TypeError(
            "path must be a str, Path, or list of str/Path"
        )

    # ------------------------------------------------------------------
    # Loading logic
    # ------------------------------------------------------------------
    def _load_all(self) -> pd.DataFrame:
        dfs = []

        for p in self.paths:
            df, site = self._load_single_path(p)
            if site is not None:
                df[self.site_column] = site
            dfs.append(df)

        return pd.concat(dfs, axis=0, ignore_index=True)

    def _load_single_path(self, path: Path) -> tuple[pd.DataFrame, str | None]:
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
        sep = "\t" if file_path.suffix == ".tsv" else ","
        return pd.read_csv(file_path, sep=sep, index_col=0)

    # ------------------------------------------------------------------
    # Preprocessing logic
    # ------------------------------------------------------------------
    def _filter(self, target_columns: list[str]) -> None:
        if (
            self.df.index.name
            and self.df.index.name.lower() in COLUMN_ALIASES["Subject"]
        ):
            self.df = self.df.reset_index()

        self.df = self.df.rename(
            columns={
                c: canonical
                for canonical, aliases in COLUMN_ALIASES.items()
                for c in self.df.columns
                if c.lower() in aliases
            }
        )

        columns = ["Subject"] + target_columns

        if "Gender" not in columns and "Gender" in self.df.columns:
            columns.append("Gender")

        if self.site_column in self.df.columns:
            columns.append(self.site_column)

        self.df = self.df.loc[:, [c for c in columns if c in self.df.columns]]

    def _categorical_to_numeric(self) -> None:
        if "Gender" in self.df.columns and self.df["Gender"].dtype == object:
            self.df["Gender"] = (
                self.df["Gender"]
                .astype(str)
                .str.upper()
                .map({"M": 1, "F": 0, "MALE": 1, "FEMALE": 0})
            )

    def _clean_df(self) -> None:
        self.df = self.df.dropna()