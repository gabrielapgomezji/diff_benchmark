from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.filebasedimages import ImageFileError
from nilearn import image as nimage
from tqdm import tqdm

from diff_benchmark.preprocessing.wrapper_brain_base import DataPreparationBrain
from diff_benchmark.preprocessing.wrapper_utils_brain_data import (
    average_per_parcel,
    compute_md,
    compute_rtop,
    create_masks,
    extract_region_data,
    extract_selected_labels,
    project_to_surface,
    resample_schaefer_onto_fs_lr,
)


class DefaultHcpPipeline(DataPreparationBrain):
    """
    DefaultHcpPipeline is a class that extends the DataPreparationBrain class to handle
    the preprocessing of brain data for the Human Connectome Project (HCP) pipeline.
    Attributes:
        hcp_dir (Path): The directory containing HCP data.
        results_root (Path): The root directory for storing results.
        metric (str): The metric to compute (e.g., 'rtop', 'md').
        schaefer_resampled: Resampled Schaefer atlas onto fs_LR.
        big_delta (float): The big delta value for diffusion metrics.
        small_delta (float): The small delta value for diffusion metrics.
    Methods:
        verify_subject_files(subject_id: str, metric: str) -> bool:
            Checks if both hemispheres' .scalar.gii files exist for the given subject and metric.
        compute_microstructure(subject_id: str):
            Computes microstructure metrics for the given subject and saves the results.
        run_analysis():
            Runs the analysis on the scalar files and computes average data per parcel.
        extract_features():
            Placeholder method for extracting features (to be implemented).
    """

    def __init__(self, config):
        super().__init__(config)
        self.hcp_dir = Path(config["data_paths"]["hcp_base"])
        self.results_root = Path(config["data_paths"]["hcp_results"]) / "default"
        self.metric = config["metric_to_compute"]
        self.scale = config.get("scale", 1000)
        self.schaefer_resampled = resample_schaefer_onto_fs_lr(scale=1000)
        self.big_delta = config["big_delta"]
        self.small_delta = config["small_delta"]

    def verify_raw_files(self, subject_id: str) -> bool:
        """
        Verifies the existence and non-emptiness of required raw files for a given subject.
        This method checks if all the necessary files for diffusion processing are present
        and non-empty in the specified subject's directory. The required files include:
        - DWI data: `data.nii.gz`
        - bvals: `bvals`
        - bvecs: `bvecs`
        - nodif mask: `nodif_brain_mask.nii.gz`
        - aparc+aseg: `aparc+aseg.nii.gz`
        If any of the files are missing or empty, a warning message is printed, and the method
        returns `False`. Otherwise, it returns `True`.
        Args:
            subject_id (str): The identifier of the subject whose files are to be verified.
        Returns:
            bool: `True` if all required files are present and non-empty, `False` otherwise.
        """
        
        subject_dir = self.hcp_dir / subject_id
        diffusion_dir = subject_dir / "T1w" / "Diffusion"

        required_files = {
            "DWI data": diffusion_dir / "data.nii.gz",
            "bvals": diffusion_dir / "bvals",
            "bvecs": diffusion_dir / "bvecs",
            "nodif mask": diffusion_dir / "nodif_brain_mask.nii.gz",
            "aparc+aseg": subject_dir / "T1w" / "aparc+aseg.nii.gz",
        }

        missing_or_empty = []
        for name, path in required_files.items():
            if not path.exists():
                missing_or_empty.append(f"{name} (missing)")
            elif path.is_file() and path.stat().st_size == 0:
                missing_or_empty.append(f"{name} (empty)")

        if missing_or_empty:
            print(
                f"[WARNING] Missing or empty files for subject {subject_id}: "
                + ", ".join(missing_or_empty)
            )
            return False

        return True

    def verify_subject_files(self, subject_id: str, metric: str) -> bool:
        """
        Check if both hemispheres' .scalar.gii files exist for the given subject and metric.
        Args:
            subject_id (str): The identifier of the subject.
            metric (str): The metric to check (e.g., 'rtop', 'md').
        Returns:
            bool: True if both left and right hemisphere files exist, False otherwise.
        """
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        left_file = (
            derivatives_dir / f"sub-{subject_id}_hemi-L_param-{metric}.scalar.gii"
        )
        right_file = (
            derivatives_dir / f"sub-{subject_id}_hemi-R_param-{metric}.scalar.gii"
        )

        return left_file.exists() and right_file.exists()

    def compute_microstructure(self, subject_id: str):
        """Compute microstructure metrics for the given subject and save the results.
        Args:
            subject_id (str): The identifier of the subject.
        """
        try:
            derivatives_dir = (
                self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
            )
            derivatives_dir.mkdir(parents=True, exist_ok=True)

            subject_dir = self.hcp_dir / subject_id

            diffusion_dir = subject_dir / "T1w" / "Diffusion"
            dwi_nib = nib.load(diffusion_dir / "data.nii.gz")
            bvals, bvecs = diffusion_dir / "bvals", diffusion_dir / "bvecs"
            bvals = np.loadtxt(bvals)
            bvecs = np.loadtxt(bvecs).T
            nodif_mask = diffusion_dir / "nodif_brain_mask.nii.gz"

            aparc_aseg = subject_dir / "T1w" / "aparc+aseg.nii.gz"

            labels = extract_selected_labels(aparc_aseg)
            aparc_resampled = nimage.resample_to_img(
                aparc_aseg,
                nodif_mask,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )

            ctx_mask, vent_mask = create_masks(aparc_resampled, labels)

            surfaces = {
                f"{h}.{s}": subject_dir
                / "T1w"
                / "fsaverage_LR32k"
                / f"{subject_id}.{h}.{s}.32k_fs_LR.surf.gii"
                for s in ("white", "pial")
                for h in ("L", "R")
            }

            if self.metric == "rtop":
                rtop_img = compute_rtop(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                )
                nib.save(
                    rtop_img,
                    derivatives_dir / f"sub-{subject_id}_param-rtop_dwimap.nii.gz",
                )
                project_to_surface(
                    rtop_img,
                    ctx_mask,
                    surfaces,
                    derivatives_dir,
                    subject_id,
                    self.metric,
                )
            elif self.metric == "md":
                md_img = compute_md(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                )
                nib.save(
                    md_img, derivatives_dir / f"sub-{subject_id}_param-md_dwimap.nii.gz"
                )
                project_to_surface(
                    md_img, ctx_mask, surfaces, derivatives_dir, subject_id, self.metric
                )

        except (FileNotFoundError, OSError, ImageFileError, KeyError, ValueError) as e:
            print(f"[{subject_id}] Expected error during microstructure: {e}")

    # def run_analysis(self):  # Doing a test
    def run_analysis_good(self):
        """Run the analysis on scalar files and compute average data per parcel."""
        scalar_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}.scalar.gii"
            )
        )
        for left_file in tqdm(scalar_files, desc="Running analysis"):
            try:
                subject_id = left_file.stem.split("_")[0].replace("sub-", "")
                right_file = left_file.with_name(
                    left_file.name.replace("hemi-L", "hemi-R")
                )

                left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(
                    0, 7
                )
                right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(
                    0, 7
                )

                avg_data = average_per_parcel(
                    left_data, right_data, self.schaefer_resampled
                )
                self.results[subject_id] = avg_data
            except (FileNotFoundError, OSError, ValueError, IndexError) as e:
                print(f"[{subject_id}] Expected error during analysis: {e}")

    # def run_analysis_region(self):
    def run_analysis(self):
        """Run analysis extracting region data."""
        scalar_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}.scalar.gii"
            )
        )
        for left_file in tqdm(scalar_files, desc="Running analysis"):
            try:
                subject_id = left_file.stem.split("_")[0].replace("sub-", "")
                right_file = left_file.with_name(
                    left_file.name.replace("hemi-L", "hemi-R")
                )

                left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(
                    0, 7
                )
                right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(
                    0, 7
                )

                target = self.config["data_preparation"]["region"]
                avg_data = extract_region_data(
                    left_data,
                    right_data,
                    self.schaefer_resampled,
                    target_substring=target,
                    average=False,
                )
                self.results[subject_id] = avg_data
            except (FileNotFoundError, OSError, ValueError, IndexError) as e:
                print(f"[{subject_id}] Expected error during analysis: {e}")

    def extract_features(self):
        pass


class ImageHcpPipeline(DataPreparationBrain):
    """
    ImageHcpPipeline is a class that extends the DataPreparationBrain class to handle
    the preprocessing of brain data for the Human Connectome Project (HCP) pipeline.
    Attributes:
        hcp_dir (Path): The directory containing HCP data.
        results_root (Path): The root directory for storing results.
        metric (str): The metric to compute (e.g., 'rtop', 'md').
        schaefer_resampled: Resampled Schaefer atlas onto fs_LR.
        big_delta (float): The big delta value for diffusion metrics.
        small_delta (float): The small delta value for diffusion metrics.
    Methods:
        verify_subject_files(subject_id: str, metric: str) -> bool:
            Checks if both hemispheres' .scalar.gii files exist for the given subject and metric.
        compute_microstructure(subject_id: str):
            Computes microstructure metrics for the given subject and saves the results.
        run_analysis():
            Runs the analysis on the scalar files and computes average data per parcel.
        extract_features():
            Placeholder method for extracting features (to be implemented).
    """

    def __init__(self, config):
        super().__init__(config)
        self.hcp_dir = Path(config["data_paths"]["hcp_base"])
        self.results_root = Path(config["data_paths"]["hcp_results"]) / "default"
        self.metric = config["metric_to_compute"]
        self.scale = config.get("scale", 1000)
        self.schaefer_resampled = resample_schaefer_onto_fs_lr(scale=1000)
        self.big_delta = config["big_delta"]
        self.small_delta = config["small_delta"]

    def verify_raw_files(self, subject_id: str) -> bool:
        """
        Verifies the existence and non-emptiness of required raw files for a given subject.
        This method checks if all the necessary files for diffusion data processing
        are present and non-empty in the specified subject's directory. If any file
        is missing or empty, a warning message is printed, and the method returns False.
        Otherwise, it confirms that all required files are valid and returns True.
        Args:
            subject_id (str): The identifier of the subject whose files are to be verified.
        Returns:
            bool: True if all required files are present and non-empty, False otherwise.
        Required Files:
            - DWI data: data.nii.gz
            - bvals: bvals
            - bvecs: bvecs
            - nodif mask: nodif_brain_mask.nii.gz
            - aparc+aseg: aparc+aseg.nii.gz
        Notes:
            - The method assumes that the subject's data is organized in a specific directory
              structure under `self.hcp_dir`.
            - Missing or empty files are logged with a warning message.
        """
        
        subject_dir = self.hcp_dir / subject_id
        diffusion_dir = subject_dir / "T1w" / "Diffusion"

        required_files = {
            "DWI data": diffusion_dir / "data.nii.gz",
            "bvals": diffusion_dir / "bvals",
            "bvecs": diffusion_dir / "bvecs",
            "nodif mask": diffusion_dir / "nodif_brain_mask.nii.gz",
            "aparc+aseg": subject_dir / "T1w" / "aparc+aseg.nii.gz",
        }

        missing_or_empty = []
        for name, path in required_files.items():
            if not path.exists():
                missing_or_empty.append(f"{name} (missing)")
            elif path.is_file() and path.stat().st_size == 0:
                missing_or_empty.append(f"{name} (empty)")

        if missing_or_empty:
            print(
                f"[WARNING] Missing or empty files for subject {subject_id}: "
                + ", ".join(missing_or_empty)
            )
            return False

        print(
            f"[INFO] All required files found and non-empty for subject {subject_id}."
        )
        return True

    def verify_subject_files(self, subject_id: str, metric: str) -> bool:
        """
        Check if whole brain .nii.gii files exist for the given subject and metric.
        Args:
            subject_id (str): The identifier of the subject.
            metric (str): The metric to check (e.g., 'rtop', 'md').
        Returns:
            bool: True if the file exists, False otherwise.
        """
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        file = derivatives_dir / f"sub-{subject_id}_param-{metric}_dwimap.nii.gz"

        return file.exists()

    def compute_microstructure(self, subject_id: str):
        """Compute microstructure metrics for the given subject and save the results.
        Args:
            subject_id (str): The identifier of the subject.
        """
        try:
            derivatives_dir = (
                self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
            )
            derivatives_dir.mkdir(parents=True, exist_ok=True)

            subject_dir = self.hcp_dir / subject_id

            diffusion_dir = subject_dir / "T1w" / "Diffusion"
            dwi_nib = nib.load(diffusion_dir / "data.nii.gz")
            bvals, bvecs = diffusion_dir / "bvals", diffusion_dir / "bvecs"
            bvals = np.loadtxt(bvals)
            bvecs = np.loadtxt(bvecs).T
            nodif_mask = diffusion_dir / "nodif_brain_mask.nii.gz"

            aparc_aseg = subject_dir / "T1w" / "aparc+aseg.nii.gz"

            labels = extract_selected_labels(aparc_aseg)
            aparc_resampled = nimage.resample_to_img(
                aparc_aseg,
                nodif_mask,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )

            ctx_mask, vent_mask = create_masks(aparc_resampled, labels)

            if self.metric == "rtop":
                rtop_img = compute_rtop(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                )
                nib.save(
                    rtop_img,
                    derivatives_dir / f"sub-{subject_id}_param-rtop_dwimap.nii.gz",
                )
            elif self.metric == "md":
                md_img = compute_md(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                )
                nib.save(
                    md_img, derivatives_dir / f"sub-{subject_id}_param-md_dwimap.nii.gz"
                )

        except (FileNotFoundError, OSError, ImageFileError, KeyError, ValueError) as e:
            print(f"[{subject_id}] Expected error during microstructure: {e}")

    def run_analysis(self):
        """Run analysis extracting region data."""
        img_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_param-{self.metric}_dwimap.nii.gz"
            )
        )
        for file in tqdm(img_files, desc="Running analysis"):
            try:
                subject_id = file.stem.split("_")[0].replace("sub-", "")
                self.results[subject_id] = file
            except (FileNotFoundError, OSError, ValueError, IndexError) as e:
                print(f"[{subject_id}] Expected error during analysis: {e}")

    def run_analysis_region(self):
        """Run analysis extracting region data."""
        img_files = sorted(
            self.results_root.glob(
                f"derivatives/sub-*/dwi/*_param-{self.metric}_dwimap.nii.gz"
            )
        )

        for file in tqdm(img_files, desc="Running analysis"):
            try:
                subject_id = file.stem.split("_")[0].replace("sub-", "")
                self.results[subject_id] = file
            except (FileNotFoundError, OSError, ValueError, IndexError) as e:
                print(f"[{subject_id}] Expected error during analysis: {e}")

    def extract_features(self):
        pass
