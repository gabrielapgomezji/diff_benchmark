"""Demographics preprocessing pipeline and BIDS layout helpers.

Contains:
- :data:`COLUMN_ALIASES` – canonical column name → set of known aliases.
- :class:`DemographicsPreparationPipeline` – loads, normalises, and filters
  demographic CSV/TSV files for both uni- and multi-centre datasets.
- :class:`CachedBIDSFile` – lightweight wrapper around a DataFrame row that
  mimics the ``pybids`` ``BIDSFile`` interface.
- :class:`CachedBIDSLayout` – ``BIDSLayout``-compatible interface backed by a
  DataFrame for fast loading without re-indexing.
"""
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Column alias registry
# ---------------------------------------------------------------------------

COLUMN_ALIASES: dict[str, set[str]] = {
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
        "age_at_scan ",  # trailing space intentional — matches raw HCP header
    },
    "Gender": {
        "gender",
        "sex",
        "gender_text",
    },
}


# ---------------------------------------------------------------------------
# Demographics pipeline
# ---------------------------------------------------------------------------


class DemographicsPreparationPipeline:
    """Unified demographics preprocessor for uni- and multi-centre datasets.

    Supports:
    - Single CSV/TSV file (unicentre).
    - List of paths — one per site (multicentre); a ``Site`` column is added
      automatically.

    The output is always a single ``pd.DataFrame`` with one row per subject.
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

    def preprocess(
        self, target_columns: list[str], binarize: bool = True
    ) -> pd.DataFrame:
        """Load, normalise, filter, and optionally binarize demographics.

        Args:
            target_columns: Demographic columns to retain (e.g. ``["Age"]``).
            binarize: When ``True``, columns with exactly 2 unique values are
                mapped to ``{0, 1}``.

        Returns:
            Preprocessed demographics ``DataFrame``.
        """
        df = self._load_all()
        df = self._drop_hcp_age_column(df)
        df = self._filter(df, target_columns)
        df = self._normalize_subject_ids(df)
        df = self._categorical_to_numeric(df)
        if binarize:
            df = self._binarize_columns(df)
        df = df.dropna()
        return df

    def get_full_demographics(
        self, available_subjects: list[str] | None = None
    ) -> pd.DataFrame:
        """Load the full demographics table without column filtering.

        Args:
            available_subjects: Optional list of subject IDs to keep.  When
                provided, rows whose ``Subject`` is not in the list are dropped.

        Returns:
            Full demographics ``DataFrame``, optionally filtered by subjects.
        """
        df = self._load_all()
        df = self._drop_hcp_age_column(df)
        df = self._normalize_subject_ids(df)

        # Rename known columns to canonical names (keep all columns)
        df = df.rename(
            columns={
                c: canonical
                for canonical, aliases in COLUMN_ALIASES.items()
                for c in df.columns
                if c.lower() in aliases
            }
        )

        df = self._categorical_to_numeric(df)

        if available_subjects is not None:
            available_str = [str(s) for s in available_subjects]
            df = df[df["Subject"].astype(str).isin(available_str)]

        return df

    # ------------------------------------------------------------------
    # Private helpers — path handling
    # ------------------------------------------------------------------

    def _normalize_paths(self, path: str | Path | list[str | Path]) -> list[Path]:
        """Coerce any supported path input into a ``list[Path]``.

        Args:
            path: A single path (str or :class:`Path`) or an iterable of them.

        Returns:
            List of :class:`Path` objects.

        Raises:
            TypeError: When ``path`` is not a str, Path, or iterable thereof.
        """
        if isinstance(path, (str, Path)):
            return [Path(path)]
        if isinstance(path, Iterable):
            return [Path(p) for p in path]
        raise TypeError("path must be a str, Path, or list of str/Path")

    def _normalize_subject_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip the ``sub-`` prefix from the ``Subject`` column if present."""
        if "Subject" in df.columns:
            df["Subject"] = (
                df["Subject"]
                .astype(str)
                .str.strip()
                .str.replace(r"^sub-", "", regex=True)
            )
        return df

    # ------------------------------------------------------------------
    # Private helpers — loading
    # ------------------------------------------------------------------

    def _load_all(self) -> pd.DataFrame:
        """Load all paths and concatenate into a single ``DataFrame``.

        A ``Site`` column is added for multi-site datasets.
        """
        dfs = []
        for p in self.paths:
            df, site = self._load_single_path(p)
            if site is not None:
                df[self.site_column] = site
            dfs.append(df)
        return pd.concat(dfs, axis=0, ignore_index=True)

    def _load_single_path(self, path: Path) -> tuple[pd.DataFrame, str | None]:
        """Load a single demographics file or directory.

        For directories, exactly one ``.tsv`` or ``.csv`` file must exist.

        Args:
            path: File or directory path.

        Returns:
            ``(DataFrame, site_name)`` where ``site_name`` is ``None`` for
            single-site datasets.

        Raises:
            ValueError: If a directory contains more or fewer than one
                demographics file.
            FileNotFoundError: If ``path`` does not exist.
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
            site = path.name if self.is_multisite else None
            return self._load_file(demo_files[0]), site

        raise FileNotFoundError(path)

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """Read a ``.tsv`` or ``.csv`` file into a ``DataFrame``."""
        sep = "\t" if file_path.suffix == ".tsv" else ","
        return pd.read_csv(file_path, sep=sep)

    # ------------------------------------------------------------------
    # Private helpers — preprocessing
    # ------------------------------------------------------------------

    def _drop_hcp_age_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the raw ``age`` column when HCP data is detected.

        HCP stores age as a bucketed string (e.g. ``"26-30"``); dropping it
        prevents it from shadowing the clean age column from a merged table.
        """
        if any("hcp" in str(p).lower() for p in self.paths):
            age_cols = [c for c in df.columns if c.lower() == "age"]
            if age_cols:
                logger.info(
                    "HCP detected in demographics paths; dropping columns: %s",
                    age_cols,
                )
                df = df.drop(columns=age_cols)
        return df

    def _filter(self, df: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
        """Rename columns to canonical names and keep only the required subset.

        Always keeps ``Subject``, optionally keeps ``Gender`` and ``Site``.

        Args:
            df: Raw demographics ``DataFrame``.
            target_columns: Columns requested by the caller.

        Returns:
            Filtered ``DataFrame``.
        """
        if df.index.name and df.index.name.lower() in COLUMN_ALIASES["Subject"]:
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

        return df.loc[:, [c for c in columns if c in df.columns]]

    def _categorical_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map ``Gender`` string values to numeric (``M``/``MALE`` → 1, ``F``/``FEMALE`` → 0)."""
        if "Gender" in df.columns and df["Gender"].dtype == object:
            df["Gender"] = (
                df["Gender"]
                .astype(str)
                .str.upper()
                .map({"M": 1, "F": 0, "MALE": 1, "FEMALE": 0})
            )
        return df

    def _binarize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map any column with exactly 2 unique values to ``{0, 1}``.

        ``Subject`` and the site column are never binarized. Columns already
        encoded as ``{0, 1}`` are skipped to avoid redundant work.
        """
        for col in df.columns:
            if col in ("Subject", self.site_column):
                continue

            unique_vals = df[col].dropna().unique()
            if len(unique_vals) != 2:
                continue

            try:
                v0, v1 = sorted(unique_vals)
            except TypeError:
                v0, v1 = sorted(unique_vals, key=str)

            if {v0, v1} == {0, 1}:
                continue  # already correctly encoded

            mapping = {v0: 0, v1: 1}
            logger.info(f"Binarizing column '{col}': {mapping}")
            df[col] = df[col].map(mapping)

        return df


# ---------------------------------------------------------------------------
# Cached BIDS layout helpers
# ---------------------------------------------------------------------------


class CachedBIDSFile:
    """Lightweight wrapper around a DataFrame row mimicking ``pybids`` BIDSFile.

    Used by :class:`CachedBIDSLayout` so callers can use the same
    ``.path``, ``.filename``, and ``.get_entities()`` interface as with a real
    ``BIDSLayout``.
    """

    def __init__(self, row):
        self._row = row
        self.path = str(row["path"])
        self.filename = Path(self.path).name

    def get_entities(self) -> dict:
        """Return entity key-value pairs (excludes ``path`` and NaN columns)."""
        return {
            k: v
            for k, v in self._row.items()
            if k != "path" and pd.notna(v) and not str(k).startswith("Unnamed")
        }

    def __repr__(self) -> str:
        return f"<CachedBIDSFile filename='{self.filename}'>"


class CachedBIDSLayout:
    """``BIDSLayout``-compatible interface backed by a DataFrame.

    Provides a faster alternative to ``pybids.BIDSLayout`` for large datasets
    by reading from a pre-built index ``DataFrame`` instead of re-crawling the
    filesystem on every instantiation.

    The ``get()`` method supports the same keyword-based filtering as
    ``BIDSLayout.get()``, including ``return_type`` and extension normalisation.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        if "subject" in self.df.columns:
            self.df["subject"] = self.df["subject"].astype(str)

    def get_subjects(self) -> list[str]:
        """Return a sorted list of unique subject IDs."""
        if "subject" not in self.df.columns:
            return []
        return sorted(self.df["subject"].dropna().unique().tolist())

    def get(self, return_type: str = "object", **kwargs) -> list:
        """Filter the layout index and return matching files.

        Args:
            return_type: ``"file"`` / ``"files"`` → list of path strings;
                anything else → list of :class:`CachedBIDSFile`.
            **kwargs: BIDS entity filters. ``None`` values match missing /
                NaN entries. ``extension`` is normalised (leading ``.``
                is optional).

        Returns:
            List of paths (strings) or :class:`CachedBIDSFile` objects.
        """
        mask = np.ones(len(self.df), dtype=bool)

        for k, v in kwargs.items():
            if k in ("return_type", "scope", "regex_search"):
                continue

            if k == "extension":
                mask &= self._match_extension(v)
                continue

            if k in self.df.columns:
                if v is None:
                    mask &= self.df[k].isna()
                elif isinstance(v, list):
                    mask &= self.df[k].isin(v)
                else:
                    mask &= self.df[k] == v

        filtered = self.df[mask]
        if "path" in filtered.columns:
            filtered = filtered.sort_values("path")

        if return_type in ("file", "files", "filename", "filenames"):
            return filtered["path"].tolist()

        return [CachedBIDSFile(row) for _, row in filtered.iterrows()]

    def _match_extension(self, v) -> "np.ndarray":
        """Return a boolean mask for extension matching (dot-prefix agnostic)."""
        if "extension" not in self.df.columns:
            return np.ones(len(self.df), dtype=bool)

        col_vals = self.df["extension"].astype(str)

        if isinstance(v, list):
            v_no_dot = [x.lstrip(".") for x in v]
            v_with_dot = ["." + x for x in v_no_dot]
            return col_vals.isin(v_no_dot) | col_vals.isin(v_with_dot)

        v_no_dot = str(v).lstrip(".")
        v_with_dot = "." + v_no_dot
        return (col_vals == v_no_dot) | (col_vals == v_with_dot)

    def to_df(self) -> pd.DataFrame:
        """Return the underlying index ``DataFrame``."""
        return self.df

    def get_file(self, filename: str) -> "CachedBIDSFile | None":
        """Return the first entry whose path ends with ``filename``, or ``None``.

        This mirrors the ``BIDSLayout.get_file()`` interface. It is primarily
        used to locate ``participants.tsv`` in ``prepare_data.py``.
        """
        if "path" in self.df.columns:
            matches = self.df[self.df["path"].astype(str).str.endswith(filename)]
            if not matches.empty:
                return CachedBIDSFile(matches.iloc[0])
        return None
