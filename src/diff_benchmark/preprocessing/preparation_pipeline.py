# base_pipeline.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import bids
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.filebasedimages import ImageFileError
from nilearn import image as nimage

from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.preprocessing.utils_brain_feature_extraction import (
    compute_save_and_project_metric,
    create_masks,
    extract_selected_labels,
    resample_schaefer_onto_fs_lr,
)
from diff_benchmark.utils.job_manager import run_jobs
from diff_benchmark.utils.logger import setup_logger

from typing import Iterable

logger = setup_logger(__name__)

@dataclass(frozen=True)
class DiffusionInputs:
    """
    Data container for diffusion MRI preprocessing pipeline inputs.
    Attributes:
        dwi_data (Path): Path to diffusion weighted imaging (DWI) data file.
        bvals (Path): Path to b-values file containing gradient strengths.
        bvecs (Path): Path to b-vectors file containing gradient directions.
        aparc_aseg (Path): Path to FreeSurfer aparc+aseg segmentation file.
        nodif_mask (Path | None): Optional path to no-diffusion (b0) mask file. Defaults to None.
        surfaces (Dict[str, Path]): Dictionary mapping surface identifiers to their file paths.
            Common keys include 'white', 'pial', etc. Defaults to an empty dictionary.
    """

    dwi_data: Path
    bvals: Path
    bvecs: Path
    aparc_aseg: Path
    nodif_mask: Path | None = None
    surfaces: Dict[str, Path] = field(default_factory=dict)

    def iter_paths(self) -> Dict[str, Path]:
        """
        Iterate over all relevant file paths for the preprocessing pipeline.
        Returns:
            Dict[str, Path]: A dictionary containing paths to:
                - dwi_data: Diffusion weighted imaging data file
                - bvals: B-values file
                - bvecs: B-vectors file
                - aparc_aseg: Aparc+aseg segmentation file
                - nodif_mask: (Optional) No diffusion mask file, only included if provided
                - surface:<key>: Surface files, prefixed with "surface:" for each entry in self.surfaces
        """

        base = {
            "dwi_data": self.dwi_data,
            "bvals": self.bvals,
            "bvecs": self.bvecs,
            "aparc_aseg": self.aparc_aseg,
        }
        if self.nodif_mask is not None:
            base["nodif_mask"] = self.nodif_mask

        return {**base, **{f"surface:{k}": v for k, v in self.surfaces.items()}}


def process_subject_wrapper(subject_id, pipeline_cls, dataset_config, recompute):
    """
    Wrapper to call the instance method safely.
    This function is top-level, so joblib can pickle it.
    """
    # pipeline._process_subject(subject_id, recompute)
    pipeline = pipeline_cls(dataset_config)
    pipeline._process_subject(subject_id, recompute)


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


class BrainDataPreparationPipeline(ABC):
    """
    BrainDataPreparationPipeline is an abstract base class for preparing and analyzing brain data.
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

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        self.data_reading = dataset_config.data_reading
        self.base_dir = Path(dataset_config.base_dir)
        self.in_derivatives = self.base_dir / "derivatives"
        self.metric = dataset_config.metric_to_compute
        self.scale = dataset_config.scale
        self.schaefer_resampled = resample_schaefer_onto_fs_lr(self.scale)
        self.big_delta = dataset_config.big_delta
        self.small_delta = dataset_config.small_delta
        self.big_delta_per_bvalue = dataset_config.big_delta_per_bvalue
        self.results = {}

        self.dwi_desc = dataset_config.dwi_desc
        self.bvec_extensions = dataset_config.bvec_extensions
        self.bval_extensions = dataset_config.bval_extensions
        self.nodif_mask_extension = dataset_config.nodif_mask_extension
        self.aparcaseg_extension = dataset_config.aparcaseg_extension

        if "bids" in self.data_reading:
            # File extension
            # --- Detect uni vs multicenter automatically ---
            if (self.base_dir / "sub-").exists() or any(
                p.name.startswith("sub-") for p in self.base_dir.iterdir()
            ):
                center_dirs = [self.base_dir]
            else:
                center_dirs = [p for p in self.base_dir.iterdir() if p.is_dir()]

            self.layouts = [
                bids.BIDSLayout(
                    str(center),
                    derivatives=center / "derivatives",
                    validate=False,
                )
                for center in center_dirs
            ]

    def get_subjects(self) -> list[str]:
        """
        Retrieve a sorted list of unique subject identifiers from the layouts.
        This method aggregates subject identifiers from all layouts associated
        with the instance, removes duplicates, and returns them in sorted order.
        Returns:
            list[str]: A sorted list of unique subject identifiers.
        """

        return sorted(
            {subject for layout in self.layouts for subject in layout.get_subjects()}
        )

    def get_layout_for_subject(self, subject_id: str) -> bids.BIDSLayout:
        """
        Find the layout containing this subject.
        Args:
            subject_id (str): The subject identifier to find the layout for.
        Returns:
            bids.BIDSLayout: The layout containing the subject.
        Raises:
            ValueError: If the subject is not found in any center.
        """
        for layout in self.layouts:
            if subject_id in layout.get_subjects():
                return layout
        raise ValueError(f"Subject {subject_id} not found in any center")

    def _get_required_raw_files(self, subject_id: str) -> DiffusionInputs | None:
        """
        Retrieves the required raw files for a given subject ID based on the data reading method.
        Args:
            subject_id (str): The unique identifier for the subject whose raw files are to be retrieved.
        Returns:
            Dict[str, Union[Path, Dict[str, Path]]]: A dictionary containing paths to the required raw files.
        Raises:
            ValueError: If the data reading method is not recognized.
        """
        try:
            if self.data_reading == "hcp":
                subject_dir = self.base_dir / subject_id
                diffusion_dir = subject_dir / "T1w" / "Diffusion"

                surfaces = {
                    f"{h}.{s}": subject_dir
                    / "T1w"
                    / "fsaverage_LR32k"
                    / f"{subject_id}.{h}.{s}.32k_fs_LR.surf.gii"
                    for s in ("white", "pial")
                    for h in ("L", "R")
                }

                return DiffusionInputs(
                    dwi_data=diffusion_dir / self.dwi_desc,
                    bvals=diffusion_dir / self.bval_extensions,
                    bvecs=diffusion_dir / self.bvec_extensions,
                    nodif_mask=diffusion_dir / self.nodif_mask_extension,
                    aparc_aseg=subject_dir / self.aparcaseg_extension,
                    surfaces=surfaces,
                )
            if "bids" in self.data_reading:
                layout = self.get_layout_for_subject(subject_id)

                aparcaseg = layout.get(
                    subject=subject_id,
                    desc="aparcaseg",
                    suffix="dseg",
                    return_type="file",
                )[0]

                dwi_bids = layout.get(
                    subject=subject_id,
                    suffix="dwi",
                    extension=".nii.gz",
                    desc=self.dwi_desc,
                )[0]

                entities = dwi_bids.get_entities()

                bvals = layout.get(
                    subject=entities["subject"],
                    extension=self.bval_extensions,
                    return_type="file",
                )[0]

                bvecs = layout.get(
                    subject=entities["subject"],
                    extension=self.bvec_extensions,
                    return_type="file",
                )[0]

                surfaces = {
                    f"{h}.{s}": Path(
                        layout.get(
                            subject=subject_id,
                            suffix=s,
                            hemi=h,
                            space=None,
                            extension=".surf.gii",
                            return_type="files",  # , density='32k'
                        )[0]
                    )
                    for s in ("white", "pial", "inflated")
                    for h in ("L", "R")
                }

                return DiffusionInputs(
                    dwi_data=Path(dwi_bids.path),
                    bvals=Path(bvals),
                    bvecs=Path(bvecs),
                    aparc_aseg=Path(aparcaseg),
                    nodif_mask=None,
                    surfaces=surfaces,
                )
        except (IndexError, KeyError):
            return None

    def verify_raw_files(self, subject_id: str) -> bool:
        """
        Verifies the existence of raw files for a given subject ID.
        Args:
            subject_id (str): The unique identifier for the subject whose raw files are to be verified.
        Returns:
            bool: True if all required raw files exist and are non-empty, False otherwise.
        """
        required_files = self._get_required_raw_files(subject_id)
        if required_files is None:
            logger.warning(f"[SKIP] {subject_id}: required raw files not found")
            return False
        missing_or_empty = []
        for name, path in required_files.iter_paths().items():
            if not path.exists():
                missing_or_empty.append(f"{name} (missing)")
            elif path.is_file() and path.stat().st_size == 0:
                missing_or_empty.append(f"{name} (empty)")

        if missing_or_empty:
            logger.warning(f"[WARNING] Missing or empty files for subject {subject_id}: " + ", ".join(missing_or_empty))
            return False

        return True

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

    def compute_microstructure(self, subject_id: str):
        """
        Compute the microstructure for a given subject.
        Parameters:
            subject_id (str): The unique identifier for the subject whose microstructure is to be computed.
        """
        try:
            derivatives_dir = (
                self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
            )
            derivatives_dir.mkdir(parents=True, exist_ok=True)

            files = self._get_required_raw_files(subject_id)
            assert files is not None

            aparc_aseg = files.aparc_aseg
            dwi_nib = nib.load(files.dwi_data)
            bvals = np.loadtxt(files.bvals)
            labels = extract_selected_labels(aparc_aseg)

            surfaces = files.surfaces

            if self.data_reading == "hcp":
                nodif_mask = files.nodif_mask
                bvecs = np.loadtxt(files.bvecs).T
                nodif_mask = files["nodif mask"]
                aparc_resampled = nimage.resample_to_img(
                    aparc_aseg,
                    nodif_mask,
                    interpolation="nearest",
                    force_resample=True,
                    copy_header=True,
                )

                selected_labels = None

            elif "bids" in self.data_reading:
                bvecs = np.loadtxt(files.bvecs)
                aparc_resampled = nimage.resample_img(
                    aparc_aseg,
                    target_affine=dwi_nib.affine,
                    target_shape=dwi_nib.shape[:3],
                    interpolation="nearest",
                    force_resample=True,
                    copy_header=True,
                )

                selected_labels = [
                    k
                    for k in labels
                    if (
                        ("ctx" in k)
                        or ("thalamus" in k)
                        or ("caudate" in k)
                        or ("putamen" in k)
                        or ("pallidum" in k)
                    )
                ]

            ctx_mask, vent_mask = create_masks(aparc_resampled, labels, selected_labels)

            compute_save_and_project_metric(
                metric=self.metric,
                dwi_nib=dwi_nib,
                ctx_mask=ctx_mask,
                vent_mask=vent_mask,
                bvals=bvals,
                bvecs=bvecs,
                big_delta=self.big_delta,
                small_delta=self.small_delta,
                big_delta_per_bvalue=self.big_delta_per_bvalue,
                surfaces=surfaces,
                derivatives_dir=derivatives_dir,
                subject_id=subject_id,
            )
        except (
            FileNotFoundError,
            OSError,
            ImageFileError,
            KeyError,
            ValueError,
            IndexError,
        ) as e:
            print(f"[{subject_id}] Expected error during microstructure: {e}")
            logger.error(f"[{subject_id}] Expected error during microstructure: {e}")

    @abstractmethod
    def run_analysis(self):
        """
        Executes the analysis process.
        This method is intended to be overridden in subclasses to implement
        specific analysis logic. Currently, it is a placeholder and does not
        perform any operations.
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
        print("Exporting data to DataFrame...")
        df = pd.DataFrame.from_dict(self.results, orient="index")
        df.index.name = "subject_id"
        return df

    def _process_subject(self, subject_id: str, recompute: bool):
        """
        Processes a single subject by checking for required files
        and computing microstructure if necessary.
        Args:
            subject_id (str): The unique identifier for the subject to be processed.
            recompute (bool): Whether to recompute microstructure even if files exist.
        """
        if not self.verify_raw_files(subject_id):
            logger.warning(f"[{subject_id}] Missing raw files, skipping")
            return  # Skip this subject

        if self.verify_subject_files(subject_id, self.metric) and recompute:
            logger.info(f"[{subject_id}] Recomputing microstructure.")
            self.compute_microstructure(subject_id)
        elif not self.verify_subject_files(subject_id, self.metric):
            logger.info(f"[{subject_id}] Computing microstructure.")
            self.compute_microstructure(subject_id)
        else:
            logger.info(f"[{subject_id}] All files already present.")

    def run_pipeline(self, recompute: bool = False) -> pd.DataFrame:
        """
        Main orchestration: ensures all required files exist before running analysis.
        Args:
            recompute (bool): Whether to recompute microstructure even if files exist.
        Returns:
            pd.DataFrame: DataFrame containing the results after running the analysis.
        """

        def parse_subject_ids(dataset):
            base = Path(dataset.base_dir)

            glob_patterns = {
                "multicenter-bids": "*/sub-*",
                "bids": "sub-*",
                "hcp": "*",
            }

            try:
                pattern = glob_patterns[dataset.data_reading]
            except KeyError:
                raise ValueError(f"Unknown data_reading: {dataset.data_reading}")

            subjects = []

            for p in base.glob(pattern):
                name = p.name
                sid = name if dataset.data_reading == "hcp" else name[4:]

                subjects.append(sid)

            return sorted(subjects)

        subject_list = parse_subject_ids(self.dataset_config)

        run_jobs(
            run_fn=process_subject_wrapper,
            fn_kwargs_list=[
                {
                    "subject_id": subject_id,
                    "pipeline_cls": type(self),
                    "dataset_config": self.dataset_config,
                    "recompute": recompute,
                }
                for subject_id in subject_list
            ],
            parallel_type="slurm",
            slurm_cfg={
                "cpus_per_task": 1,
                "timeout_min": 900,
                "mem_gb": 50,
            },
            n_jobs=50,
        )

        # Once all files are ready, run the analysis
        print("All required files are ready. Now you can run analysis!")
        logger.info("All required files are ready. Now you can run analysis!")
        # self.run_analysis()
        # df = self.export_to_csv()
        # return df

    def load_features(self) -> pd.DataFrame:
        """
        Main orchestration: ensures all required files exist before running analysis.
        Returns:
            pd.DataFrame: DataFrame containing the results after running the analysis.
        """
        logger.info("All data should be preprocessed already. Getting microstructure files...")
        self.run_analysis()
        print("Computing df...")
        df = self.export_to_csv()
        return df


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

class DemographicsPreparationPipeline:
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
    def _normalize_paths(self, path: str | Path | list[str | Path]) -> list[Path]:
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

        raise TypeError("path must be a str, Path, or list of str/Path")

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
        return pd.read_csv(file_path, sep=sep)  # , index_col=0)

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
