# base_pipeline.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Dict, Union

import bids
import nibabel as nib
import numpy as np
from nibabel.filebasedimages import ImageFileError
from nilearn import image as nimage

from diff_benchmark.preprocessing.wrapper_utils_brain_data import (
    extract_selected_labels,
    project_to_surface,
    resample_schaefer_onto_fs_lr,
)
from diff_benchmark.preprocessing.wrapper_utils_brain_data import (
                    compute_md_bids,
                    compute_rtop_bids,
                    create_masks_bids,
                    compute_save_and_project_metric
                )
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig

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

# DataPreparationBrain
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
        return sorted({
            subject
            for layout in self.layouts
            for subject in layout.get_subjects()
        })

    def get_layout_for_subject(self, subject_id: str) -> bids.BIDSLayout:
        # find the layout containing this subject
        for layout in self.layouts:
            if subject_id in layout.get_subjects():
                return layout
        raise ValueError(f"Subject {subject_id} not found in any center")
       
    def _get_required_raw_files(self, subject_id: str) -> Dict[str, Union[Path, Dict[str, Path]]]:
        """
        Return a dict mapping logical names -> file paths
        """
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
            

            return {
                "DWI data": diffusion_dir / self.dwi_desc,
                "bvals": diffusion_dir / self.bval_extensions,
                "bvecs": diffusion_dir / self.bvec_extensions,
                "nodif mask": diffusion_dir / self.nodif_mask_extension,
                "aparc+aseg": subject_dir / self.aparcaseg_extension,
                **{f"surface:{k}": v for k, v in surfaces.items()},
            }
        elif "bids" in self.data_reading:
            layout = self.get_layout_for_subject(subject_id)

            aparcaseg = layout.get(
                subject=subject_id, desc="aparcaseg", suffix="dseg", return_type="file"
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
                f"{h}.{s}": self.layout.get(
                    subject=subject_id,
                    suffix=s,
                    hemi=h,
                    space=None,
                    extension=".surf.gii",
                    return_type="files",  # , density='32k'
                )[0]
                for s in ("white", "pial", "inflated")
                for h in ("L", "R")
            }

            return {
                "DWI data": Path(dwi_bids.path),
                "bvals": Path(bvals),
                "bvecs": Path(bvecs),
                "aparc+aseg": Path(aparcaseg),
                **{f"surface:{k}": v for k, v in surfaces.items()},
            }

  
    def verify_raw_files(self, subject_id: str) -> bool:
        """
        Verifies the existence of raw files for a given subject ID.
        Args:
            subject_id (str): The unique identifier for the subject whose raw files are to be verified.
        """
        required_files = self._get_required_raw_files(subject_id)

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
            aparc_aseg = files["aparc+aseg"]
            labels = extract_selected_labels(aparc_aseg)
            dwi_nib = nib.load(files["DWI data"])
            bvals = np.loadtxt(files["bvals"])
            # surfaces = files["surfaces"]
            surfaces = {k.split("surface:")[1]: v for k, v in files.items() if k.startswith("surface:")}
            if self.data_reading == "hcp":
                bvecs = np.loadtxt(files["bvecs"]).T
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
                bvecs = np.loadtxt(files["bvecs"])
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

            ctx_mask, vent_mask = create_masks_bids(
                aparc_resampled, labels, selected_labels
            )
            
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
            
            # if self.metric == "rtop":
            #     rtop_img = compute_rtop_bids(
            #         dwi_nib,
            #         ctx_mask,
            #         vent_mask,
            #         bvals,
            #         bvecs,
            #         self.big_delta,
            #         self.small_delta,
            #         self.big_delta_per_bvalue,
            #     )
            #     nib.save(
            #         rtop_img,
            #         derivatives_dir / f"sub-{subject_id}_param-rtop_dwimap.nii.gz",
            #     )

            #     project_to_surface(
            #         rtop_img,
            #         ctx_mask,
            #         surfaces,
            #         derivatives_dir,
            #         subject_id,
            #         self.metric,
            #     )
            # elif self.metric == "md":
            #     md_img = compute_md_bids(
            #         dwi_nib,
            #         ctx_mask,
            #         vent_mask,
            #         bvals,
            #         bvecs,
            #         self.big_delta,
            #         self.small_delta,
            #         self.big_delta_per_bvalue,
            #     )
            #     nib.save(
            #         md_img, derivatives_dir / f"sub-{subject_id}_param-md_dwimap.nii.gz"
            #     )
            #     project_to_surface(
            #         md_img, ctx_mask, surfaces, derivatives_dir, subject_id, self.metric
            #     )

        except (FileNotFoundError, OSError, ImageFileError, KeyError, ValueError) as e:
            print(f"[{subject_id}] Expected error during microstructure: {e}")


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
        df = pd.DataFrame.from_dict(self.results, orient="index")
        df.index.name = "subject_id"
        return df

    def run_pipeline(self, recompute: bool = False) -> pd.DataFrame:
        """
        Main orchestration: ensures all required files exist before running analysis.
        """
        subject_list = sorted(
            [
                p.name
                for p in Path(self.dataset_config.base_dir).iterdir()
                if p.is_dir() and p.name.isdigit()
            ]
        )

        def process_subject(subject_id):
            """Processes a single subject by checking for required files"""
            # if not self.verify_subject_files(
            if self.verify_raw_files(subject_id):
                if (
                    self.verify_subject_files(
                        subject_id, self.dataset_config.metric_to_compute
                    )
                    and recompute
                ):
                    print(f"[{subject_id}] Recomputing microstructure.")
                    self.compute_microstructure(subject_id)
                else:
                    print(f"[{subject_id}] Missing files — computing microstructure.")
                    self.compute_microstructure(subject_id)
            # else:
            #     print(f"[{subject_id}] All required files found.")

        Parallel(n_jobs=50)(
            delayed(process_subject)(subject_id)
            for subject_id in tqdm(subject_list, desc="Processing subjects")
        )

        # Once all files are ready, run the analysis
        print("All required files are ready. Now you can run analysis!")
        # self.run_analysis()
        # df = self.export_to_csv()
        # return df

    def run_microstructure_pipeline(self) -> pd.DataFrame:
        """
        Main orchestration: ensures all required files exist before running analysis.
        """
        print(
            "All data should be preprocessed already. Getting microstructure files..."
        )
        self.run_analysis()
        df = self.export_to_csv()
        return df
