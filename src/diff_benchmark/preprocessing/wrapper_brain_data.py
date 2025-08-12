import json
from pathlib import Path

from diff_benchmark.preprocessing.wrapper_utils_brain_data import compute_microstructure, project_volume_to_surface, extract_by_parcel  # or extract_by_vertex

from diff_benchmark.preprocessing.base_brain_data import BrainDataPreprocessor

# rtop_pipeline.py
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

from diff_benchmark.preprocessing.wrapper_utils_brain_data import (
    extract_selected_labels, create_masks, compute_rtop,
    project_to_surface, resample_schaefer_onto_fs_LR,
    average_per_parcel, compute_md
)
from diff_benchmark.preprocessing.wrapper_brain_base import DataPreparationBrain

class RTOP_HCPPipeline(DataPreparationBrain):

    def __init__(self, config):
        super().__init__(config)
        self.hcp_dir = Path(config["hcp_path"])
        self.results_root = Path(config["results_path"])
        self.metric = config["metric_to_compute"]
        self.schaefer_resampled = resample_schaefer_onto_fs_LR(scale=1000)
        self.big_delta = config["big_delta"]
        self.small_delta = config["small_delta"]

    def compute_microstructure(self, subject_id: str):
        try:
            derivatives_dir = self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
            derivatives_dir.mkdir(parents=True, exist_ok=True)
        
            subject_dir = self.hcp_dir / subject_id
            # output_dir = self.results_root / subject_id / "processed"
            # output_dir.mkdir(parents=True, exist_ok=True)

            diffusion_dir = subject_dir / "T1w" / "Diffusion"
            dwi_nib = nib.load(diffusion_dir / "data.nii.gz")
            bvals, bvecs = diffusion_dir / "bvals", diffusion_dir / "bvecs"
            bvals = np.loadtxt(bvals)
            bvecs = np.loadtxt(bvecs).T
            nodif_mask = diffusion_dir / "nodif_brain_mask.nii.gz"
            
            surfaces = {
                f"{h}.{s}": subject_dir / "T1w" / "fsaverage_LR32k" / f"{subject_id}.{h}.{s}.32k_fs_LR.surf.gii"
                for s in ("white", "pial") for h in ("L", "R")
            }

            aparc_aseg = subject_dir / "T1w" / "aparc+aseg.nii.gz"
            labels = extract_selected_labels(aparc_aseg)
            aparc_resampled = nib.processing.resample_to_img(aparc_aseg, nodif_mask, interpolation='nearest')
            ctx_mask, vent_mask = create_masks(aparc_resampled, labels)
            if self.metric == "rtop":
                rtop_img = compute_rtop(dwi_nib, ctx_mask, vent_mask, bvals, bvecs, self.big_delta, self.small_delta)
                nib.save(rtop_img, derivatives_dir / f"sub-{subject_id}_param-rtop_dwimap.nii.gz")
                project_to_surface(rtop_img, ctx_mask, surfaces, derivatives_dir, subject_id, self.metric)
            elif self.metric == "md":
                md_img = compute_md(dwi_nib, ctx_mask, vent_mask, bvals, bvecs, self.big_delta, self.small_delta)
                nib.save(md_img, derivatives_dir / f"sub-{subject_id}_param-md_dwimap.nii.gz")
                project_to_surface(md_img, ctx_mask, surfaces, derivatives_dir, subject_id, self.metric)
                
        except Exception as e:
            print(f"[{subject_id}] Error during microstructure: {e}")

    def run_analysis(self):
        scalar_files = sorted(self.results_root.glob(f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}.scalar.gii"))
        for left_file in tqdm(scalar_files, desc="Running analysis"):
            try:
                subject_id = left_file.stem.split("_")[0]
                right_file = left_file.with_name(left_file.name.replace(".L.", ".R."))

                rtop_left = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(0, 7)
                rtop_right = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(0, 7)

                rtop_avg = average_per_parcel(rtop_left, rtop_right, self.schaefer_resampled)
                self.results[subject_id] = rtop_avg
            except Exception as e:
                print(f"[{subject_id}] Error during analysis: {e}")

    def extract_features(self):
        pass
