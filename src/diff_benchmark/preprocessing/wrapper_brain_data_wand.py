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
from diff_benchmark.preprocessing.wrapper_brain_base import DataPreparationBrain
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

LABELS = {'???': 0, 'left-cerebral-white-matter': 2, 'left-lateral-ventricle': 4, 'left-inf-lat-vent': 5, 
          'left-cerebellum-white-matter': 7, 'left-cerebellum-cortex': 8, 'left-thalamus-proper': 10, 
          'left-caudate': 11, 'left-putamen': 12, 'left-pallidum': 13, '3rd-ventricle': 14, 
          '4th-ventricle': 15, 'brain-stem': 16, 'left-hippocampus': 17, 'left-amygdala': 18, 'csf': 24, 
          'left-accumbens-area': 26, 'left-ventraldc': 28, 'left-vessel': 30, 'left-choroid-plexus': 31, 
          'right-cerebral-white-matter': 41, 'right-lateral-ventricle': 43, 'right-inf-lat-vent': 44, 
          'right-cerebellum-white-matter': 46, 'right-cerebellum-cortex': 47, 'right-thalamus-proper': 49, 
          'right-caudate': 50, 'right-putamen': 51, 'right-pallidum': 52, 'right-hippocampus': 53, 
          'right-amygdala': 54, 'right-accumbens-area': 58, 'right-ventraldc': 60, 'right-vessel': 62, 
          'right-choroid-plexus': 63, 'wm-hypointensities': 77, 'non-wm-hypointensities': 80, 
          'optic-chiasm': 85, 'cc_posterior': 251, 'cc_mid_posterior': 252, 'cc_central': 253, 
          'cc_mid_anterior': 254, 'cc_anterior': 255, 'ctx-lh-unknown': 1000, 'ctx-lh-bankssts': 1001, 
          'ctx-lh-caudalanteriorcingulate': 1002, 'ctx-lh-caudalmiddlefrontal': 1003, 'ctx-lh-cuneus': 1005, 
          'ctx-lh-entorhinal': 1006, 'ctx-lh-fusiform': 1007, 'ctx-lh-inferiorparietal': 1008, 
          'ctx-lh-inferiortemporal': 1009, 'ctx-lh-isthmuscingulate': 1010, 'ctx-lh-lateraloccipital': 1011, 
          'ctx-lh-lateralorbitofrontal': 1012, 'ctx-lh-lingual': 1013, 'ctx-lh-medialorbitofrontal': 1014, 
          'ctx-lh-middletemporal': 1015, 'ctx-lh-parahippocampal': 1016, 'ctx-lh-paracentral': 1017, 
          'ctx-lh-parsopercularis': 1018, 'ctx-lh-parsorbitalis': 1019, 'ctx-lh-parstriangularis': 1020, 
          'ctx-lh-pericalcarine': 1021, 'ctx-lh-postcentral': 1022, 'ctx-lh-posteriorcingulate': 1023, 
          'ctx-lh-precentral': 1024, 'ctx-lh-precuneus': 1025, 'ctx-lh-rostralanteriorcingulate': 1026, 
          'ctx-lh-rostralmiddlefrontal': 1027, 'ctx-lh-superiorfrontal': 1028, 'ctx-lh-superiorparietal': 1029, 
          'ctx-lh-superiortemporal': 1030, 'ctx-lh-supramarginal': 1031, 'ctx-lh-frontalpole': 1032, 
          'ctx-lh-temporalpole': 1033, 'ctx-lh-transversetemporal': 1034, 'ctx-lh-insula': 1035, 
          'ctx-rh-unknown': 2000, 'ctx-rh-bankssts': 2001, 'ctx-rh-caudalanteriorcingulate': 2002, 
          'ctx-rh-caudalmiddlefrontal': 2003, 'ctx-rh-cuneus': 2005, 'ctx-rh-entorhinal': 2006, 
          'ctx-rh-fusiform': 2007, 'ctx-rh-inferiorparietal': 2008, 'ctx-rh-inferiortemporal': 2009, 
          'ctx-rh-isthmuscingulate': 2010, 'ctx-rh-lateraloccipital': 2011, 'ctx-rh-lateralorbitofrontal': 2012, 
          'ctx-rh-lingual': 2013, 'ctx-rh-medialorbitofrontal': 2014, 'ctx-rh-middletemporal': 2015, 
          'ctx-rh-parahippocampal': 2016, 'ctx-rh-paracentral': 2017, 'ctx-rh-parsopercularis': 2018, 
          'ctx-rh-parsorbitalis': 2019, 'ctx-rh-parstriangularis': 2020, 'ctx-rh-pericalcarine': 2021, 
          'ctx-rh-postcentral': 2022, 'ctx-rh-posteriorcingulate': 2023, 'ctx-rh-precentral': 2024, 
          'ctx-rh-precuneus': 2025, 'ctx-rh-rostralanteriorcingulate': 2026, 'ctx-rh-rostralmiddlefrontal': 2027, 
          'ctx-rh-superiorfrontal': 2028, 'ctx-rh-superiorparietal': 2029, 'ctx-rh-superiortemporal': 2030, 
          'ctx-rh-supramarginal': 2031, 'ctx-rh-frontalpole': 2032, 'ctx-rh-temporalpole': 2033, 
          'ctx-rh-transversetemporal': 2034, 'ctx-rh-insula': 2035}
class DefaultWandPipeline(DataPreparationBrain):
    """
    DefaultWandPipeline is a class that extends the DataPreparationBrain class to handle
    the preprocessing of brain data for the Wand pipeline.
    Attributes:
        wand_dir (Path): The directory containing Wand data.
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
        self.wand_dir = Path(config["data_paths"]["wand_base"])
        self.in_derivatives = Path(config["data_paths"]["wand_base"]) / "derivatives"
        self.results_root = Path(config["data_paths"]["wand_results"]) / "default"
        self.metric = config["metric_to_compute"]
        self.scale = config.get("scale", 1000)
        self.schaefer_resampled = resample_schaefer_onto_fs_lr(scale=1000)
        self.big_delta = config["big_delta_wand"]
        self.small_delta = config["small_delta_wand"]
        
        # NEW ATTRIBUTE TO STORE RESULTS
        self.layout = bids.BIDSLayout(str(self.wand_dir), derivatives=self.in_derivatives, validate=False)

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
                )
                nib.save(
                    rtop_img,
                    derivatives_dir / f"sub-{subject_id}_param-rtop_dwimap.nii.gz",
                )
                breakpoint()
                project_to_surface(
                    rtop_img,
                    ctx_mask,
                    surfaces,
                    derivatives_dir,
                    subject_id,
                    self.metric,
                )
            # elif self.metric == "md":
            #     md_img = compute_md(
            #         dwi_nib,
            #         ctx_mask,
            #         vent_mask,
            #         bvals,
            #         bvecs,
            #         self.big_delta,
            #         self.small_delta,
            #     )
            #     nib.save(
            #         md_img, derivatives_dir / f"sub-{subject_id}_param-md_dwimap.nii.gz"
            #     )
            #     project_to_surface(
            #         md_img, ctx_mask, surfaces, derivatives_dir, subject_id, self.metric
            #     )

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