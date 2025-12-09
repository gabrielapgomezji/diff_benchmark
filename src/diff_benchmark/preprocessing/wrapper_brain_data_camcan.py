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


class DefaultCamcanPipeline(DataPreparationBrain):
    """
    DefaultCamcanPipeline is a class that extends the DataPreparationBrain class to handle
    the preprocessing of brain data for the CamCAN pipeline.
    Attributes:
        camcan_dir (Path): The directory containing CamCANb data.
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
        self.hcp_dir = Path(config["data_paths"]["camcan_base"])
        self.results_root = Path(config["data_paths"]["camcan_results"]) / "default"
        self.metric = config["metric_to_compute"]
        self.scale = config.get("scale", 1000)
        self.schaefer_resampled = resample_schaefer_onto_fs_lr(scale=1000)
        self.big_delta = config["big_delta_camcan"]
        self.small_delta = config["small_delta_camcan"]
        
        # NEW ATTRIBUTE TO STORE RESULTS
        # self.layout = bids.BIDSLayout(str(CAMCAN_ROOT), derivatives=derivatives, validate=False)

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
        
        # subject_dir = self.hcp_dir / subject_id
        # diffusion_dir = subject_dir / "T1w" / "Diffusion"

        required_files = {
            "DWI data": dwi_file,
            "bvals": dwi_file_bvals,
            "bvecs": dwi_file_bvecs,
            # "nodif mask": diffusion_dir / "nodif_brain_mask.nii.gz",
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
            bvecs = np.loadtxt(dwi_file_bvecs).T
            
            # nodif_mask = diffusion_dir / "nodif_brain_mask.nii.gz" # NOT USED IN CAMCAN?

            # labels = extract_selected_labels(aparc_aseg) # CHECK BUT IS CORRECT
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
            from xml import etree
            header = nib.load(aparc_aseg).header
            labels = {
                n.text.lower(): int(n.get('Key'))
                for n in etree.ElementTree.fromstring(header.extensions[0].text).findall('.//Label')
            }
            aparc_resampled = nimage.resample_to_img(
                aparc_aseg,
                target_affine=dwi_nib.affine,
                target_shape=dwi_nib.shape[:3],
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )

            # ctx_mask, vent_mask = create_masks(aparc_resampled, labels) # CHECK
            # BELOW IS THE WORKING WAY FOR CAMCAN
            from scipy import ndimage # in utils for create_mask
            ctx_mask = nimage.math_img(
                " + ".join(
                    f"(x == {labels[k]})"
                    for k in selected_labels
                ),
                x=aparc_resampled
            )

            vent_mask = nimage.new_img_like(
                aparc_resampled,
                ndimage.binary_erosion(
                    nimage.get_data(
                        nimage.math_img(
                            " + ".join(
                                f"(x == {v})" for k, v in labels.items()
                                if 'vent' in k
                            ), x=aparc_resampled
                        )
                    )
                )
            )
            from nilearn import maskers # Also imported in utils
            b0 = nimage.index_img(dwi_nib, 0)
            ctx_masker = maskers.NiftiMasker(ctx_mask)
            ctx_masker.fit(b0)

            ventricle_masker = maskers.NiftiMasker(vent_mask)
            ventricle_masker.fit(b0)

            surfaces = {
                f"{h}.{s}": self.layout.get(
                subject=subject_id, suffix=s, hemi=h, space=None,
                extension='.surf.gii', return_type='files' #, density='32k'

            )[0]
                for s in ("white", "pial", "inflated")
                for h in ("L", "R")
            }

            if self.metric == "rtop":
                # rtop_img = compute_rtop(
                #     dwi_nib,
                #     ctx_mask,
                #     vent_mask,
                #     bvals,
                #     bvecs,
                #     self.big_delta,
                #     self.small_delta,
                # )
                # nib.save(
                #     rtop_img,
                #     derivatives_dir / f"sub-{subject_id}_param-rtop_dwimap.nii.gz",
                # )
                # SMALL MODIFICATIONS
                big_delta_per_bvalue = {
                    1000: 24,
                    2000: 30,
                }
                selected_bvals = [0] + [k for k, v in big_delta_per_bvalue.items() if v == self.big_delta * 1000]
                bvals_mask = np.any([bvals == s for s in selected_bvals], axis=0)

                dwi_insula = ctx_masker.transform(dwi_nib)
                dwi_ventricles = ventricle_masker.transform(dwi_nib)
                dwi_insula_bmasked = dwi_insula[bvals_mask, :]
                dwi_ventricles_bmasked = dwi_ventricles[bvals_mask, :]

                gtab = gradient_table(bvals=bvals[bvals_mask], bvecs=bvecs[bvals_mask], small_delta=self.small_delta, big_delta=self.big_delta)
                from dipy.reconst.mapmri import MapmriModel
                radial_order = 6
                map_model_laplacian_aniso = MapmriModel(
                    gtab,
                    radial_order=radial_order,
                    laplacian_regularization=True,
                    laplacian_weighting=0.2,
                    positivity_constraint=False,
                    
                )

                rtop_insula_bmasked = map_model_laplacian_aniso.fit(dwi_insula_bmasked.T).rtop()
                rtop_ventricles_masked = map_model_laplacian_aniso.fit(dwi_ventricles_bmasked.T).rtop()
                # Need an adaptation of the following 2 lines
                nrtop = rtop_insula_bmasked / rtop_ventricles_masked[~np.isnan(rtop_ventricles_masked)].mean()
                nrtop = nrtop.clip(0, np.percentile(nrtop[~np.isnan(nrtop)], 99))
                nrtop_img = ctx_masker.inverse_transform(nrtop)
                rtop_img = nrtop_img
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