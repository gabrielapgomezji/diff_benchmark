# from pathlib import Path

# import nibabel as nib
# import numpy as np
# from nilearn import image as nimage
# from tqdm import tqdm

# from diff_benchmark.preprocessing.wrapper_brain_base import DataPreparationBrain
# from diff_benchmark.preprocessing.wrapper_utils_brain_data import (
#     average_per_parcel,
#     compute_md,
#     compute_rtop,
#     create_masks,
#     extract_region_data,
#     extract_selected_labels,
#     project_to_surface,
#     resample_schaefer_onto_fs_lr,
# )

# class DefaultWandPipeline(DataPreparationBrain):
#     """
#     DefaultWandPipeline is a class that extends the DataPreparationBrain class to handle
#     the preprocessing of brain data for the Wand pipeline.
#     Attributes:
#         wand_dir (Path): The directory containing Wand data.
#         results_root (Path): The root directory for storing results.
#         metric (str): The metric to compute (e.g., 'rtop', 'md').
#         schaefer_resampled: Resampled Schaefer atlas onto fs_LR.
#         big_delta (float): The big delta value for diffusion metrics.
#         small_delta (float): The small delta value for diffusion metrics.
#     Methods:
#         verify_subject_files(subject_id: str, metric: str) -> bool:
#             Checks if both hemispheres' .scalar.gii files exist for the given subject and metric.
#         compute_microstructure(subject_id: str):
#             Computes microstructure metrics for the given subject and saves the results.
#         run_analysis():
#             Runs the analysis on the scalar files and computes average data per parcel.
#         extract_features():
#             Placeholder method for extracting features (to be implemented).
#     """

#     def __init__(self, config):
#         super().__init__(config)
#         self.wand_dir = Path(config["data_paths"]["wand_base"])
#         self.derivatives_in = self.wand_dir / "derivatives"
#         self.results_root = Path(config["data_paths"]["wand_results"]) / "default"
#         self.metric = config["metric_to_compute"]
#         # breakpoint()
#         self.scale = config.get("scale", 1000)
#         self.schaefer_resampled = resample_schaefer_onto_fs_lr(scale=1000)
#         self.big_delta = config.get("big_delta_wand", 24e-3)
#         self.small_delta = config.get("small_delta_wand", 7e-3)
#         self.big_delta_per_bvalue = config.get(
#             "big_delta_per_bvalue",
#             {
#                 2200: 24,
#                 4000: 30,
#                 4400: 24,
#                 8000: 30,
#                 5800: 42,
#                 7750: 55,
#                 11600: 42,
#                 15500: 55,
#             },
#         )

#     def verify_raw_files(self, subject_id: str) -> bool:
#         pass

#     def verify_subject_files(self, subject_id: str, metric: str) -> bool:
#         """
#         Check if both hemispheres' .scalar.gii files exist for the given subject and metric.
#         """
#         derivatives_dir = (
#             self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
#         )
#         left_file = (
#             derivatives_dir / f"sub-{subject_id}_hemi-L_param-{metric}.scalar.gii"
#         )
#         right_file = (
#             derivatives_dir / f"sub-{subject_id}_hemi-R_param-{metric}.scalar.gii"
#         )

#         return left_file.exists() and right_file.exists()

#     def compute_microstructure(self, subject_id: str):
#         """Compute microstructure metrics for the given subject and save the results."""
#         try:
#             # breakpoint()
#             # layout = bids.BIDSLayout(self.wand_dir, derivatives=self.derivatives_in, validate=False)
#             derivatives_dir = (
#                 self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
#             )
#             derivatives_dir.mkdir(parents=True, exist_ok=True)
#             ####
#             # derivatives_dir = self.derivatives_in / f"sub-{subject_id}" / "dwi"

#             # subject_dir = self.wand_dir / subject_id / "ses-02"

#             # diffusion_dir = subject_dir / "dwi"

#             aparcaseg_path = (
#                 self.derivatives_in
#                 / f"smriprep/sub-{subject_id}/ses-02/anat/sub-{subject_id}_ses-02_desc-aparcaseg_dseg.nii.gz"
#             )
#             dwi_path = (
#                 self.derivatives_in
#                 / f"preprocess/sub-{subject_id}/sub-{subject_id}_ses-02_acq-AxCaliberConcat_space-individualT1_desc-eddycorrected_bbreg_dwi.nii.gz"
#             )
#             bvals_path = (
#                 self.wand_dir
#                 / f"sub-{subject_id}/ses-02/dwi/sub-{subject_id}_ses-02_acq-AxCaliberConcat_dwi.bval"
#             )
#             bvecs_path = (
#                 self.derivatives_in
#                 / f"preprocess/sub-{subject_id}/ses-02/dwi/sub-{subject_id}_ses-02_acq-AxCaliberConcat_desc-rotated_dwi.bvec"
#             )
#             bvecs = np.loadtxt(bvecs_path).T
#             bvals = np.loadtxt(bvals_path)
#             dwi_nib = nib.load(dwi_path)

#             parcellation_dwi = nimage.resample_img(
#                 aparcaseg_path,
#                 target_affine=dwi_nib.affine,
#                 target_shape=dwi_nib.shape[:3],
#                 interpolation="nearest",
#                 force_resample=True,
#                 copy_header=True,
#             )

#             b0 = nimage.index_img(dwi_nib, 0)

#             # dwi_nib = nib.load(diffusion_dir / "data.nii.gz")
#             # bvals, bvecs = diffusion_dir / "bvals", diffusion_dir / "bvecs"
#             # bvals = np.loadtxt(bvals)
#             # bvecs = np.loadtxt(bvecs).T

#             labels = extract_selected_labels(aparc_aseg)
#             aparc_resampled = nimage.resample_to_img(
#                 aparc_aseg,
#                 nodif_mask,
#                 interpolation="nearest",
#                 force_resample=True,
#                 copy_header=True,
#             )

#             ctx_mask, vent_mask = create_masks(aparc_resampled, labels)

#             surfaces = {
#                 f"{h}.{s}": subject_dir
#                 / "T1w"
#                 / "fsaverage_LR32k"
#                 / f"{subject_id}.{h}.{s}.32k_fs_LR.surf.gii"
#                 for s in ("white", "pial")
#                 for h in ("L", "R")
#             }

#             if self.metric == "rtop":
#                 rtop_img = compute_rtop(
#                     dwi_nib,
#                     ctx_mask,
#                     vent_mask,
#                     bvals,
#                     bvecs,
#                     self.big_delta,
#                     self.small_delta,
#                 )
#                 nib.save(
#                     rtop_img,
#                     derivatives_dir / f"sub-{subject_id}_param-rtop_dwimap.nii.gz",
#                 )
#                 project_to_surface(
#                     rtop_img,
#                     ctx_mask,
#                     surfaces,
#                     derivatives_dir,
#                     subject_id,
#                     self.metric,
#                 )
#             elif self.metric == "md":
#                 md_img = compute_md(
#                     dwi_nib,
#                     ctx_mask,
#                     vent_mask,
#                     bvals,
#                     bvecs,
#                     self.big_delta,
#                     self.small_delta,
#                 )
#                 nib.save(
#                     md_img, derivatives_dir / f"sub-{subject_id}_param-md_dwimap.nii.gz"
#                 )
#                 project_to_surface(
#                     md_img, ctx_mask, surfaces, derivatives_dir, subject_id, self.metric
#                 )

#         except Exception as e:
#             print(f"[{subject_id}] Error during microstructure: {e}")

#     # def run_analysis(self):  # Doing a test
#     def run_analysis_good(self):
#         scalar_files = sorted(
#             self.results_root.glob(
#                 f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}.scalar.gii"
#             )
#         )
#         for left_file in tqdm(scalar_files, desc="Running analysis"):
#             try:
#                 subject_id = left_file.stem.split("_")[0].replace("sub-", "")
#                 right_file = left_file.with_name(
#                     left_file.name.replace("hemi-L", "hemi-R")
#                 )

#                 left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(
#                     0, 7
#                 )
#                 right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(
#                     0, 7
#                 )

#                 avg_data = average_per_parcel(
#                     left_data, right_data, self.schaefer_resampled
#                 )
#                 self.results[subject_id] = avg_data
#             except Exception as e:
#                 print(f"[{subject_id}] Error during analysis: {e}")

#     # def run_analysis_region(self):
#     def run_analysis(self):
#         scalar_files = sorted(
#             self.results_root.glob(
#                 f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}.scalar.gii"
#             )
#         )
#         for left_file in tqdm(scalar_files, desc="Running analysis"):
#             try:
#                 subject_id = left_file.stem.split("_")[0].replace("sub-", "")
#                 right_file = left_file.with_name(
#                     left_file.name.replace("hemi-L", "hemi-R")
#                 )

#                 left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(
#                     0, 7
#                 )
#                 right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(
#                     0, 7
#                 )

#                 # target = "VisCent_Striate"
#                 # target = self.config["region_name"]
#                 target = self.config["models"][0]["params"]["region_name"]
#                 # target = None
#                 avg_data = extract_region_data(
#                     left_data,
#                     right_data,
#                     self.schaefer_resampled,
#                     target_substring=target,
#                     average=False,
#                 )
#                 self.results[subject_id] = avg_data
#             except Exception as e:
#                 print(f"[{subject_id}] Error during analysis: {e}")

#     def extract_features(self):
#         pass


# # class DefaultCamcanPipeline(DataPreparationBrain):
# #     """
# #     DefaultCamcanPipeline is a class that extends the DataPreparationBrain class to handle
# #     the preprocessing of brain data for the CamCAN pipeline.
# #     Attributes:
# #         camcan_dir (Path): The directory containing CamCAN data.
# #         results_root (Path): The root directory for storing results.
# #         metric (str): The metric to compute (e.g., 'rtop', 'md').
# #         schaefer_resampled: Resampled Schaefer atlas onto fs_LR.
# #         big_delta (float): The big delta value for diffusion metrics.
# #         small_delta (float): The small delta value for diffusion metrics.
# #     Methods:
# #         verify_subject_files(subject_id: str, metric: str) -> bool:
# #             Checks if both hemispheres' .scalar.gii files exist for the given subject and metric.
# #         compute_microstructure(subject_id: str):
# #             Computes microstructure metrics for the given subject and saves the results.
# #         run_analysis():
# #             Runs the analysis on the scalar files and computes average data per parcel.
# #         extract_features():
# #             Placeholder method for extracting features (to be implemented).
# #     """

# #     def __init__(self, config):
# #         super().__init__(config)
# #         self.camcan_dir = Path(config["data_paths"]["camcan_base"])
# #         self.results_root = Path(config["data_paths"]["camcan_results"]) / "default"
# #         self.metric = config["metric_to_compute"]
# #         breakpoint()
# #         self.scale = config.get("scale", 1000)
# #         self.schaefer_resampled = resample_schaefer_onto_fs_lr(scale=1000)
# #         self.big_delta = config["big_delta"]
# #         self.small_delta = config["small_delta"]
# #    def verify_raw_files(self, subject_id: str) -> bool:
# #        pass
# #     def verify_subject_files(self, subject_id: str, metric: str) -> bool:
# #         """
# #         Check if both hemispheres' .scalar.gii files exist for the given subject and metric.
# #         """
# #         derivatives_dir = (
# #             self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
# #         )
# #         left_file = (
# #             derivatives_dir / f"sub-{subject_id}_hemi-L_param-{metric}.scalar.gii"
# #         )
# #         right_file = (
# #             derivatives_dir / f"sub-{subject_id}_hemi-R_param-{metric}.scalar.gii"
# #         )

# #         return left_file.exists() and right_file.exists()

# #     def compute_microstructure(self, subject_id: str):
# #         """Compute microstructure metrics for the given subject and save the results."""
# #         try:
# #             derivatives_dir = (
# #                 self.results_root / "derivatives" / f"sub-{subject_id}" / "dwi"
# #             )
# #             derivatives_dir.mkdir(parents=True, exist_ok=True)

# #             subject_dir = self.hcp_dir / subject_id

# #             diffusion_dir = subject_dir / "T1w" / "Diffusion"
# #             dwi_nib = nib.load(diffusion_dir / "data.nii.gz")
# #             bvals, bvecs = diffusion_dir / "bvals", diffusion_dir / "bvecs"
# #             bvals = np.loadtxt(bvals)
# #             bvecs = np.loadtxt(bvecs).T
# #             nodif_mask = diffusion_dir / "nodif_brain_mask.nii.gz"

# #             aparc_aseg = subject_dir / "T1w" / "aparc+aseg.nii.gz"

# #             labels = extract_selected_labels(aparc_aseg)
# #             aparc_resampled = nimage.resample_to_img(
# #                 aparc_aseg,
# #                 nodif_mask,
# #                 interpolation="nearest",
# #                 force_resample=True,
# #                 copy_header=True,
# #             )

# #             ctx_mask, vent_mask = create_masks(aparc_resampled, labels)

# #             surfaces = {
# #                 f"{h}.{s}": subject_dir
# #                 / "T1w"
# #                 / "fsaverage_LR32k"
# #                 / f"{subject_id}.{h}.{s}.32k_fs_LR.surf.gii"
# #                 for s in ("white", "pial")
# #                 for h in ("L", "R")
# #             }

# #             if self.metric == "rtop":
# #                 rtop_img = compute_rtop(
# #                     dwi_nib,
# #                     ctx_mask,
# #                     vent_mask,
# #                     bvals,
# #                     bvecs,
# #                     self.big_delta,
# #                     self.small_delta,
# #                 )
# #                 nib.save(
# #                     rtop_img,
# #                     derivatives_dir / f"sub-{subject_id}_param-rtop_dwimap.nii.gz",
# #                 )
# #                 project_to_surface(
# #                     rtop_img,
# #                     ctx_mask,
# #                     surfaces,
# #                     derivatives_dir,
# #                     subject_id,
# #                     self.metric,
# #                 )
# #             elif self.metric == "md":
# #                 md_img = compute_md(
# #                     dwi_nib,
# #                     ctx_mask,
# #                     vent_mask,
# #                     bvals,
# #                     bvecs,
# #                     self.big_delta,
# #                     self.small_delta,
# #                 )
# #                 nib.save(
# #                     md_img, derivatives_dir / f"sub-{subject_id}_param-md_dwimap.nii.gz"
# #                 )
# #                 project_to_surface(
# #                     md_img, ctx_mask, surfaces, derivatives_dir, subject_id, self.metric
# #                 )

# #         except Exception as e:
# #             print(f"[{subject_id}] Error during microstructure: {e}")

# #     # def run_analysis(self):  # Doing a test
# #     def run_analysis_good(self):
# #         scalar_files = sorted(
# #             self.results_root.glob(
# #                 f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}.scalar.gii"
# #             )
# #         )
# #         for left_file in tqdm(scalar_files, desc="Running analysis"):
# #             try:
# #                 subject_id = left_file.stem.split("_")[0].replace("sub-", "")
# #                 right_file = left_file.with_name(
# #                     left_file.name.replace("hemi-L", "hemi-R")
# #                 )

# #                 left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(
# #                     0, 7
# #                 )
# #                 right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(
# #                     0, 7
# #                 )

# #                 avg_data = average_per_parcel(
# #                     left_data, right_data, self.schaefer_resampled
# #                 )
# #                 self.results[subject_id] = avg_data
# #             except Exception as e:
# #                 print(f"[{subject_id}] Error during analysis: {e}")

# #     # def run_analysis_region(self):
# #     def run_analysis(self):
# #         scalar_files = sorted(
# #             self.results_root.glob(
# #                 f"derivatives/sub-*/dwi/*_hemi-L_param-{self.metric}.scalar.gii"
# #             )
# #         )
# #         for left_file in tqdm(scalar_files, desc="Running analysis"):
# #             try:
# #                 subject_id = left_file.stem.split("_")[0].replace("sub-", "")
# #                 right_file = left_file.with_name(
# #                     left_file.name.replace("hemi-L", "hemi-R")
# #                 )

# #                 left_data = np.nan_to_num(nib.load(left_file).darrays[0].data).clip(
# #                     0, 7
# #                 )
# #                 right_data = np.nan_to_num(nib.load(right_file).darrays[0].data).clip(
# #                     0, 7
# #                 )

# #                 # target = "VisCent_Striate"
# #                 # target = self.config["region_name"]
# #                 target = self.config["models"][0]["params"]["region_name"]
# #                 # target = None
# #                 avg_data = extract_region_data(
# #                     left_data, right_data, self.schaefer_resampled, target_substring=target, average=False
# #                 )
# #                 self.results[subject_id] = avg_data
# #             except Exception as e:
# #                 print(f"[{subject_id}] Error during analysis: {e}")

# #     def extract_features(self):
# #         pass
