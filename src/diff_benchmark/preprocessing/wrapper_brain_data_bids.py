import json
from pathlib import Path

import h5py
import networkx as nx
import nibabel as nib
import nilearn as ni
import numpy as np
import pandas as pd
import torch
from dipy.core.gradients import gradient_table
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.reconst.mapmri import MapmriModel
from joblib import Parallel, delayed
from nibabel.filebasedimages import ImageFileError
from nilearn import image as nimage
from tqdm import tqdm

from diff_benchmark.preprocessing.lcot.sliced_lcot import EmbeddingCircleWeights
from diff_benchmark.preprocessing.wrapper_brain_base_general import DataPreparationBrain
from diff_benchmark.preprocessing.wrapper_utils_brain_data import (
    average_per_parcel,
    compute_data,
    compute_md,
    compute_rtop,
    create_masks,
    extract_region_data,
    extract_selected_labels,
    load_vertexwise_attenuations,
    project_to_surface,
    resample_schaefer_onto_fs_lr,
    split_data,
)
import bids

class DefaultPipeline(DataPreparationBrain):
    """
    DefaultPipeline is a class that extends the DataPreparationBrain class to handle
    the preprocessing of brain data for the CamCAN pipeline.
    Attributes:
        base_dir (Path): The directory containing CamCANb data.
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
        self.base_dir = Path(dataset_config.base_dir)
        self.in_derivatives = self.base_dir / "derivatives"
        self.results_root = Path(dataset_config.results_dir) / "default"
        self.metric = dataset_config.metric_to_compute
        self.scale = dataset_config.scale
        self.schaefer_resampled = resample_schaefer_onto_fs_lr(self.scale)
        self.big_delta = dataset_config.big_delta
        self.small_delta = dataset_config.small_delta
        self.big_delta_per_bvalue = dataset_config.big_delta_per_bvalue
        
        # NEW ATTRIBUTE TO STORE RESULTS
        self.layout = bids.BIDSLayout(str(self.base_dir), derivatives=self.in_derivatives, validate=False)
        
    def verify_raw_files(self, subject_id: str) -> bool:
        aparcaseg = self.layout.get(subject=subject_id, desc='aparcaseg', suffix='dseg', return_type='file')[0]
        dwi_bids = self.layout.get(subject=subject_id, suffix='dwi', extension='.nii.gz', desc='eddycorrected+bbreg')[0]
        dwi_file = dwi_bids.path
        entities_ = dwi_bids.get_entities()
        dwi_file_bvals = self.layout.get(
            subject=entities_['subject'], 
            extension='bval', return_type='file'
        )[0]
        dwi_file_bvecs = self.layout.get(
            subject=entities_['subject'],
            extension='bvec', return_type='file'
        )[0]

        required_files = {
            "DWI data": dwi_file,
            "bvals": dwi_file_bvals,
            "bvecs": dwi_file_bvecs,
            "aparc+aseg": aparcaseg,
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
        """Compute microstructure metrics for the given subject and save the results."""
        try:
            derivatives_dir = (
                self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
            )
            derivatives_dir.mkdir(parents=True, exist_ok=True)
            
            aparc_aseg = self.layout.get(subject=subject_id, desc='aparcaseg', suffix='dseg', return_type='file')[0]
            dwi_bids = self.layout.get(subject=subject_id, suffix='dwi', extension='.nii.gz', desc='eddycorrected+bbreg')[0]
            dwi_file = dwi_bids.path
            
            entities_ = dwi_bids.get_entities()
            dwi_file_bvals = self.layout.get(
                subject=entities_['subject'], 
                extension='bval', return_type='file'
            )[0]
            dwi_file_bvecs = self.layout.get(
                subject=entities_['subject'],
                extension='bvec', return_type='file'
            )[0]
            
            dwi_nib = nib.load(dwi_file)
            bvals = np.loadtxt(dwi_file_bvals)
            bvecs = np.loadtxt(dwi_file_bvecs)

            labels = extract_selected_labels(aparc_aseg)
    
            selected_labels = [
                k for k in labels
                if (
                    ('ctx' in k) or
                    ('thalamus' in k) or
                    ('caudate' in k) or
                    ('putamen' in k) or
                    ('pallidum' in k)
                )
            ]
            aparc_resampled = nimage.resample_img(
                aparc_aseg,
                target_affine=dwi_nib.affine,
                target_shape=dwi_nib.shape[:3],
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )

            # ctx_mask, vent_mask = create_masks(aparc_resampled, labels) # CHECK
            from diff_benchmark.preprocessing.wrapper_utils_brain_data import create_masks_bids
            ctx_mask, vent_mask = create_masks_bids(aparc_resampled, labels, selected_labels)

            surfaces = {
                f"{h}.{s}": self.layout.get(
                subject=subject_id, suffix=s, hemi=h, space=None,
                extension='.surf.gii', return_type='files' #, density='32k'

            )[0]
                for s in ("white", "pial", "inflated")
                for h in ("L", "R")
            }

            if self.metric == "rtop":
                from diff_benchmark.preprocessing.wrapper_utils_brain_data import compute_rtop_bids
                rtop_img = compute_rtop_bids(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                    self.big_delta_per_bvalue,
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
                from diff_benchmark.preprocessing.wrapper_utils_brain_data import compute_md_bids
                md_img = compute_md_bids(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                    self.big_delta_per_bvalue,
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

class DefaultMulticenterPipeline(DataPreparationBrain):
    """
    DefaultPipeline is a class that extends the DataPreparationBrain class to handle
    the preprocessing of brain data for the CamCAN pipeline.
    Attributes:
        base_dir (Path): The directory containing CamCANb data.
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
        self.base_dir = Path(dataset_config.base_dir)
        self.in_derivatives = self.base_dir / "derivatives"
        self.results_root = Path(dataset_config.results_dir) / "default"
        self.metric = dataset_config.metric_to_compute
        self.scale = dataset_config.scale
        self.schaefer_resampled = resample_schaefer_onto_fs_lr(self.scale)
        self.big_delta = dataset_config.big_delta
        self.small_delta = dataset_config.small_delta
        self.big_delta_per_bvalue = dataset_config.big_delta_per_bvalue
        
        # NEW ATTRIBUTE TO STORE RESULTS
        # self.layout = bids.BIDSLayout(str(self.base_dir), derivatives=self.in_derivatives, validate=False)
        self.center_dirs = [p for p in self.base_dir.iterdir() if p.is_dir()]
        
        # Create a layout per center
        self.layouts = [
            bids.BIDSLayout(str(center_dir), derivatives=center_dir / "derivatives", validate=False)
            for center_dir in self.center_dirs
        ]
        
    def get_subjects(self):
        subjects = []
        for layout in self.layouts:
            subjects += layout.get_subjects()
        return sorted(list(set(subjects)))
    
    def get_layout_for_subject(self, subject_id):
        # find the layout containing this subject
        for layout in self.layouts:
            if subject_id in layout.get_subjects():
                return layout
        raise ValueError(f"Subject {subject_id} not found in any center")
    
    def verify_raw_files(self, subject_id: str) -> bool:
        self.layout = self.get_layout_for_subject(subject_id)

        aparcaseg = self.layout.get(subject=subject_id, desc='aparcaseg', suffix='dseg', return_type='file')[0]
        dwi_bids = self.layout.get(subject=subject_id, suffix='dwi', extension='.nii.gz', desc='eddycorrected+bbreg')[0]
        dwi_file = dwi_bids.path
        entities_ = dwi_bids.get_entities()
        dwi_file_bvals = self.layout.get(
            subject=entities_['subject'], 
            extension='bval', return_type='file'
        )[0]
        dwi_file_bvecs = self.layout.get(
            subject=entities_['subject'],
            extension=r'bvec.*', return_type='file' # for abide is extension='bvecs_absolute' the rest is bvec
        )[0]

        required_files = {
            "DWI data": Path(dwi_file),
            "bvals": Path(dwi_file_bvals),
            "bvecs": Path(dwi_file_bvecs),
            "aparc+aseg": Path(aparcaseg),
        }

        missing_or_empty = []
        for name, path in required_files.items():
            breakpoint()
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
        """Compute microstructure metrics for the given subject and save the results."""
        try:
            derivatives_dir = (
                self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
            )
            derivatives_dir.mkdir(parents=True, exist_ok=True)

            self.layout = self.get_layout_for_subject(subject_id)
            aparc_aseg = self.layout.get(subject=subject_id, desc='aparcaseg', suffix='dseg', return_type='file')[0]
            dwi_bids = self.layout.get(subject=subject_id, suffix='dwi', extension='.nii.gz', desc='eddycorrected+bbreg')[0]
            dwi_file = dwi_bids.path
            
            entities_ = dwi_bids.get_entities()
            dwi_file_bvals = self.layout.get(
                subject=entities_['subject'], 
                extension='bval', return_type='file'
            )[0]
            dwi_file_bvecs = self.layout.get(
                subject=entities_['subject'],
                extension='bvecs_image', return_type='file' # For abide is extension ='bvecs_image'
            )[0]
            
            dwi_nib = nib.load(dwi_file)
            bvals = np.loadtxt(dwi_file_bvals)
            bvecs = np.loadtxt(dwi_file_bvecs)

            labels = extract_selected_labels(aparc_aseg)

            selected_labels = [
                k for k in labels
                if (
                    ('ctx' in k) or
                    ('thalamus' in k) or
                    ('caudate' in k) or
                    ('putamen' in k) or
                    ('pallidum' in k)
                )
            ]
            aparc_resampled = nimage.resample_img(
                aparc_aseg,
                target_affine=dwi_nib.affine,
                target_shape=dwi_nib.shape[:3],
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )

            # ctx_mask, vent_mask = create_masks(aparc_resampled, labels) # CHECK
            from diff_benchmark.preprocessing.wrapper_utils_brain_data import create_masks_bids
            ctx_mask, vent_mask = create_masks_bids(aparc_resampled, labels, selected_labels)

            surfaces = {
                f"{h}.{s}": self.layout.get(
                subject=subject_id, suffix=s, hemi=h, space=None,
                extension='.surf.gii', return_type='files' #, density='32k'

            )[0]
                for s in ("white", "pial", "inflated")
                for h in ("L", "R")
            }

            if self.metric == "rtop":
                from diff_benchmark.preprocessing.wrapper_utils_brain_data import compute_rtop_bids
                breakpoint()
                rtop_img = compute_rtop_bids(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                    self.big_delta_per_bvalue,
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
                from diff_benchmark.preprocessing.wrapper_utils_brain_data import compute_md_bids
                md_img = compute_md_bids(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                    self.big_delta_per_bvalue,
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

class ImageCamcanPipeline(DataPreparationBrain):
    """
    ImageCamcanPipeline is a class that extends the DataPreparationBrain class to handle
    the preprocessing of brain data for the CamCAN pipeline.
    Attributes:
        camcan_dir (Path): The directory containing CamCAN data.
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
        self.base_dir = Path(dataset_config.base_dir)
        self.in_derivatives = self.base_dir / "derivatives"
        self.results_root = Path(dataset_config.derivatives_dir) / "default"
        self.metric = dataset_config.metric_to_compute
        self.scale = dataset_config.get("scale", 1000)
        self.schaefer_resampled = resample_schaefer_onto_fs_lr(self.scale)
        self.big_delta = dataset_config["big_delta"]
        self.small_delta = dataset_config["small_delta"]
        self.big_delta_per_bvalue = dataset_config.get("big_delta_per_bvalue_camcan", None)
        
        self.layout = bids.BIDSLayout(str(self.base_dir), derivatives=self.in_derivatives, validate=False)

    def verify_raw_files(self, subject_id: str) -> bool:
        aparcaseg = self.layout.get(subject=subject_id, desc='aparcaseg', suffix='dseg', return_type='file')[0]
        dwi_bids = self.layout.get(subject=subject_id, suffix='dwi', extension='.nii.gz', desc='eddycorrected+bbreg')[0]
        dwi_file = dwi_bids.path
        entities_ = dwi_bids.get_entities()
        dwi_file_bvals = self.layout.get(
            subject=entities_['subject'], 
            extension='bval', return_type='file'
        )[0]
        dwi_file_bvecs = self.layout.get(
            subject=entities_['subject'],
            extension='bvec', return_type='file'
        )[0]

        required_files = {
            "DWI data": dwi_file,
            "bvals": dwi_file_bvals,
            "bvecs": dwi_file_bvecs,
            "aparc+aseg": aparcaseg,
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
        Check if whole brain .nii.gii files exist for the given subject and metric.
        """
        derivatives_dir = (
            self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
        )
        file = derivatives_dir / f"sub-{subject_id}_param-{metric}_dwimap.nii.gz"

        return file.exists()

    def compute_microstructure(self, subject_id: str):
        """Compute microstructure metrics for the given subject and save the results."""
        try:
            derivatives_dir = (
                self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
            )
            derivatives_dir.mkdir(parents=True, exist_ok=True)
            
            aparc_aseg = self.layout.get(subject=subject_id, desc='aparcaseg', suffix='dseg', return_type='file')[0]
            dwi_bids = self.layout.get(subject=subject_id, suffix='dwi', extension='.nii.gz', desc='eddycorrected+bbreg')[0]
            dwi_file = dwi_bids.path
            
            entities_ = dwi_bids.get_entities()
            dwi_file_bvals = self.layout.get(
                subject=entities_['subject'], 
                extension='bval', return_type='file'
            )[0]
            dwi_file_bvecs = self.layout.get(
                subject=entities_['subject'],
                extension='bvec', return_type='file'
            )[0]
            
            dwi_nib = nib.load(dwi_file)
            bvals = np.loadtxt(dwi_file_bvals)
            bvecs = np.loadtxt(dwi_file_bvecs).T

            breakpoint()
            labels = extract_selected_labels(aparc_aseg)
            
            selected_labels = [
                k for k in labels
                if (
                    ('ctx' in k) or
                    ('thalamus' in k) or
                    ('caudate' in k) or
                    ('putamen' in k) or
                    ('pallidum' in k)
                )
            ]
            aparc_resampled = nimage.resample_img(
                aparc_aseg,
                target_affine=dwi_nib.affine,
                target_shape=dwi_nib.shape[:3],
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )

            # ctx_mask, vent_mask = create_masks(aparc_resampled, labels) # CHECK
            from diff_benchmark.preprocessing.wrapper_utils_brain_data import create_masks_bids
            ctx_mask, vent_mask = create_masks_bids(aparc_resampled, labels, selected_labels)

            if self.metric == "rtop":
                from diff_benchmark.preprocessing.wrapper_utils_brain_data import compute_rtop_bids
                rtop_img = compute_rtop_bids(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                    self.big_delta_per_bvalue,
                )
                nib.save(
                    rtop_img,
                    derivatives_dir / f"sub-{subject_id}_param-rtop_dwimap.nii.gz",
                )
                
            elif self.metric == "md":
                from diff_benchmark.preprocessing.wrapper_utils_brain_data import compute_md_bids
                md_img = compute_md_bids(
                    dwi_nib,
                    ctx_mask,
                    vent_mask,
                    bvals,
                    bvecs,
                    self.big_delta,
                    self.small_delta,
                    self.big_delta_per_bvalue,
                )
                nib.save(
                    md_img, derivatives_dir / f"sub-{subject_id}_param-md_dwimap.nii.gz"
                )

        except (FileNotFoundError, OSError, ImageFileError, KeyError, ValueError) as e:
            print(f"[{subject_id}] Expected error during microstructure: {e}")

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

    def extract_features(self):
        pass