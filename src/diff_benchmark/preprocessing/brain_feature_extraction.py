from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.preprocessing.preparation_pipeline import (
    BrainDataPreparationPipeline,
)
from diff_benchmark.preprocessing.utils_brain_feature_extraction import extract_region_data
from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


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

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)
        self.results_root = Path(dataset_config.results_dir) / "default"

    def verify_subject_files(self, subject_id: str, metric: str) -> bool:
        """
        Check if both hemispheres' .scalar.gii files exist for the given subject and metric.
        Args:
            subject_id (str): The subject identifier.
            metric (str): The metric to check (e.g., 'rtop', 'md').
        Returns:
            bool: True if both files exist, False otherwise.
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
    
    def verify_resampling(self, subject_id: str) -> bool:
        """
        Check if data has been properly resampled to template space.
        
        For BIDS datasets, checks if the vertex count matches the expected template space size.
        For HCP datasets, always returns True (data is already in template space).
        
        Args:
            subject_id (str): The subject identifier.
        
        Returns:
            bool: True if data is properly resampled (or doesn't need resampling), False otherwise.
        """
        # HCP data is already in template space, no resampling needed
        if "bids" not in self.data_reading:
            return True
        
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        left_file = (
            derivatives_dir / f"sub-{subject_id}_hemi-L_param-{self.metric}.scalar.gii"
        )
        right_file = (
            derivatives_dir / f"sub-{subject_id}_hemi-R_param-{self.metric}.scalar.gii"
        )
        
        if not (left_file.exists() and right_file.exists()):
            return False
        
        try:
            # Expected vertex count for fsLR 32k template space
            EXPECTED_VERTICES = 32492
            
            left_data = nib.load(left_file).darrays[0].data
            right_data = nib.load(right_file).darrays[0].data
            
            left_vertices = left_data.shape[0]
            right_vertices = right_data.shape[0]
            
            # Check if data has the expected number of vertices for template space
            is_resampled = (left_vertices == EXPECTED_VERTICES and 
                          right_vertices == EXPECTED_VERTICES)
            
            if not is_resampled:
                logger.debug(f"[{subject_id}] Data in native space: L={left_vertices}, R={right_vertices} vertices")
            
            return is_resampled
            
        except Exception as e:
            logger.warning(f"[{subject_id}] Error checking resampling status: {e}")
            return False

    def resample_data(self, subject_id: str):
        """
        Resample existing scalar.gii files from native space to template space.
        This is useful for data that was preprocessed before the automatic resampling feature.
        
        Args:
            subject_id (str): The subject identifier.
        """
        if "bids" not in self.data_reading:
            logger.info(f"[{subject_id}] Not a BIDS dataset, no resampling needed")
            return
        
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        left_file = (
            derivatives_dir / f"sub-{subject_id}_hemi-L_param-{self.metric}.scalar.gii"
        )
        right_file = (
            derivatives_dir / f"sub-{subject_id}_hemi-R_param-{self.metric}.scalar.gii"
        )
        
        if not (left_file.exists() and right_file.exists()):
            logger.warning(f"[{subject_id}] Scalar files not found, skipping resampling")
            return
        
        try:
            logger.info(f"[{subject_id}] Resampling existing data to template space")
            
            # Load native space data
            left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(0, 7)
            right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(0, 7)
            
            # Resample to template space
            from diff_benchmark.preprocessing.utils_brain_feature_extraction import resample_subject_to_template
            
            left_resampled, right_resampled = resample_subject_to_template(
                subject_id=subject_id,
                left_data=left_data,
                right_data=right_data,
                layouts=self.layouts,
                target_space=self.surface_space,
            )
            
            # Save resampled data with the same filenames (overwriting)
            left_gii = nib.gifti.GiftiImage()
            left_gii.add_gifti_data_array(
                nib.gifti.GiftiDataArray(data=left_resampled.astype(np.float32))
            )
            nib.save(left_gii, left_file)
            
            right_gii = nib.gifti.GiftiImage()
            right_gii.add_gifti_data_array(
                nib.gifti.GiftiDataArray(data=right_resampled.astype(np.float32))
            )
            nib.save(right_gii, right_file)
            
            logger.info(f"[{subject_id}] Resampled data saved successfully")
            
        except Exception as e:
            logger.error(f"[{subject_id}] Error during resampling: {e}")

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

                # Load data (already in template space if BIDS dataset was properly preprocessed)
                left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(
                    0, 7
                )
                right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(
                    0, 7
                )

                # No resampling needed here - data should already be in template space
                # If you see warnings about mismatched sizes, it means preprocessing 
                # was done before this optimization was implemented

                target = self.dataset_config.region
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
                logger.warning(f"[{subject_id}] Expected error during analysis: {e}")


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

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)
        # self.in_derivatives = self.base_dir / "derivatives"
        self.results_root = Path(dataset_config.results_dir) / "default"

    def verify_subject_files(self, subject_id: str, metric: str) -> bool:
        """
        Check if whole brain .nii.gii files exist for the given subject and metric.
        Args:
            subject_id (str): The subject identifier.
            metric (str): The metric to check (e.g., 'rtop', 'md').
        Returns:
            bool: True if the file exists, False otherwise.
        """
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        file = derivatives_dir / f"sub-{subject_id}_param-{metric}_dwimap.nii.gz"

        return file.exists()

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
                logger.warning(f"[{subject_id}] Expected error during analysis: {e}")
