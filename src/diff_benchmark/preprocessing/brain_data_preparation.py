from pathlib import Path
from typing import Dict, Union

import bids
import nibabel as nib
import numpy as np
from nibabel.filebasedimages import ImageFileError
from nilearn import image as nimage
from tqdm import tqdm

from diff_benchmark.preprocessing.preparation_pipeline_unified import BrainDataPreparationPipeline
from diff_benchmark.preprocessing.wrapper_utils_brain_data import (
    average_per_parcel,
    extract_region_data,
    extract_selected_labels,
    project_to_surface,
    resample_schaefer_onto_fs_lr,
)


class DefaultPipeline(BrainDataPreparationPipeline):
    """
    DefaultPipeline is a class that extends the BrainDataPreparationPipeline class to handle
    the preprocessing of brain data for the CamCAN pipeline.
    Attributes:
        base_dir (Path): The directory containing the directory data.
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

    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.results_root = Path(dataset_config.results_dir) / "default"

    def verify_subject_files(self, subject_id: str, metric: str) -> bool:
        """
        Check if both hemispheres' .scalar.gii files exist for the given subject and metric.
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

    def run_analysis(self):
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


class ImagePipeline(BrainDataPreparationPipeline):
    """
    ImagePipeline is a class that extends the BrainDataPreparationPipeline class to handle
    the preprocessing of brain data for the Human Connectome Project (HCP) pipeline.
    Attributes:
        base_dir (Path): The directory containing the directory data.
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
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # self.in_derivatives = self.base_dir / "derivatives"
        self.results_root = Path(dataset_config.results_dir) / "default"

    def verify_subject_files(self, subject_id: str, metric: str) -> bool:
        """
        Check if whole brain .nii.gii files exist for the given subject and metric.
        """
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        file = derivatives_dir / f"sub-{subject_id}_param-{metric}_dwimap.nii.gz"

        return file.exists()

    def run_analysis(self):
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

