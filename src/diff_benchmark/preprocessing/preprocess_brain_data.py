from pathlib import Path

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.core.subdivide_octahedron import create_unit_sphere

from diff_benchmark.preprocessing.base_brain_data import BrainDataPreprocessor
from diff_benchmark.preprocessing.models_configuration import get_model_configs
from diff_benchmark.preprocessing.utils_brain import (
    compute_data,
    create_subgraph,
    extract_data,
    save_output,
)


class DefaultBrainPreprocessor(BrainDataPreprocessor):
    """
    DefaultBrainPreprocessor is a class that extends the BrainDataPreprocessor
    to handle the preprocessing of brain data for individual subjects.
    Methods:
        preprocess_subject(raw_data_path: Path, subject_id: str, save_path: Path):
            Preprocesses the brain data for a given subject by extracting data,
            computing results using specified models, and saving the output to
            the designated path.
        save_subject_info(subject_id: str):
            Saves information related to the specified subject. This is a placeholder
            method that currently only prints a message.
        save_dataset_info():
            Saves information related to the entire dataset. This is a placeholder
            method that currently only prints a message.
    """

    def preprocess_subject(self, raw_data_path: Path, subject_id: str, save_path: Path):
        save_path.mkdir(parents=True, exist_ok=True)

        normalize_input = False
        sphere = create_unit_sphere(6)

        data = extract_data(raw_data_path=raw_data_path)
        gtab_original = gradient_table(bvals=data["bvals"], bvecs=data["bvecs"])
        graph_ins = create_subgraph(data)
        gtab0 = gradient_table(bvals=np.repeat(1, 1), bvecs=np.array([[1, 0, 0]]))
        bvals_to_compute = [1000, 2000, 3000]
        model_configs = get_model_configs(gtab_original)

        for name, model in model_configs.items():
            output_file = save_path / f"{name}_all_bvals.h5"
            if output_file.exists():
                print(
                    f"Skipping {subject_id} model {name} because output already exists."
                )
                continue

            all_results = compute_data(
                data, bvals_to_compute, sphere, model, gtab0, graph_ins, normalize_input
            )
            save_output(all_results, save_path, name, sphere, data, subject_id)

    # def preprocess_dataset(self):
    #     base_path = Path(self.config["results_path"])
    #     subjects = [p.name for p in base_path.iterdir() if p.is_dir()]
    #     for sub in subjects:
    #         raw_path = base_path / sub / "raw_surface_data.h5"
    #         save_path = Path(self.config["results_path_2"]) / sub / "processed"
    #         self.preprocess_subject(raw_path, sub, save_path)

    def save_subject_info(self, subject_id: str):
        print(f"[{subject_id}] Saving subject info... (placeholder)")

    def save_dataset_info(self):
        print("Saving dataset info... (placeholder)")


class RtopVolumePreprocessor(BrainDataPreprocessor):
    def preprocess_subject(self, raw_data_path: Path, subject_id: str, save_path: Path):
        save_path.mkdir(parents=True, exist_ok=True)

        from os.path import expanduser

        import yaml

        config_path = Path(__file__).parent.parent / "configuration.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # === Load DWI data ===
        folder = Path(expanduser(config["base_path"]))
        diffusion_folder = folder / subject_id / "T1w" / "Diffusion"
        dwi = nib.load(diffusion_folder / "data.nii.gz")
        data = dwi.get_fdata()
        affine = dwi.affine

        bvals = np.loadtxt(diffusion_folder / "bvals")
        bvecs = np.loadtxt(diffusion_folder / "bvecs")
        gtab = gradient_table(bvals, bvecs)

        print(f"[{subject_id}] Fitting MAPMRI model...")
        model_configs = get_model_configs(gtab)

        for name, model in model_configs.items():
            # output_file = save_path / f"{name}_all_bvals.h5"
            # === Fit MAP-MRI model ===
            map_model = model(gtab)
            map_fit = map_model.fit(data)

            # === Compute RTOP ===
            rtop = map_fit.rtop()

            # === Save RTOP image ===
            rtop_img = nib.Nifti1Image(rtop.astype(np.float32), affine)
            nib.save(rtop_img, save_path / f"{name}_RTOP.nii.gz")

            print(
                f"[{subject_id}] RTOP map saved to: {save_path / f'{name}_RTOP.nii.gz'}"
            )

    def save_subject_info(self, subject_id: str):
        print(f"[{subject_id}] Saving subject info... (placeholder)")

    def save_dataset_info(self):
        print("Saving dataset info... (placeholder)")


class TensorVolumePreprocessor(BrainDataPreprocessor):
    def preprocess_subject(self, raw_data_path: Path, subject_id: str, save_path: Path):
        save_path.mkdir(parents=True, exist_ok=True)

        from os.path import expanduser

        import yaml
        from dipy.reconst.dti import TensorModel

        config_path = Path(__file__).parent.parent / "configuration.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        folder = Path(expanduser(config["base_path"]))
        diffusion_folder = folder / subject_id / "T1w" / "Diffusion"

        # === Load DWI data ===
        dwi = nib.load(diffusion_folder / "data.nii.gz")
        data = dwi.get_fdata()
        affine = dwi.affine

        bvals = np.loadtxt(diffusion_folder / "bvals")
        bvecs = np.loadtxt(diffusion_folder / "bvecs")
        gtab = gradient_table(bvals, bvecs)

        print(f"[{subject_id}] Fitting DTI model...")
        # === Fit Tensor Model ===
        tensor_model = TensorModel(gtab)
        tensor_fit = tensor_model.fit(data)

        # === Extract FA and MD ===
        fa = tensor_fit.fa
        md = tensor_fit.md

        # === Save maps ===
        fa_img = nib.Nifti1Image(fa.astype(np.float32), affine)
        md_img = nib.Nifti1Image(md.astype(np.float32), affine)

        nib.save(fa_img, save_path / f"FA.nii.gz")
        nib.save(md_img, save_path / f"MD.nii.gz")

        print(f"[{subject_id}] FA map saved to: {save_path / f'FA.nii.gz'}")
        print(f"[{subject_id}] MD map saved to: {save_path / f'MD.nii.gz'}")

    def save_subject_info(self, subject_id: str):
        print(f"[{subject_id}] Saving subject info... (placeholder)")

    def save_dataset_info(self):
        print("Saving dataset info... (placeholder)")


class DWIMetricPreprocessor(BrainDataPreprocessor):
    def __init__(self, config: dict, metrics_to_compute: list):
        super().__init__(config)
        self.config = config
        self.metrics_to_compute = set(metrics_to_compute)

    def preprocess_subject(self, raw_data_path: Path, subject_id: str, save_path: Path):
        from os.path import expanduser
        from pathlib import Path

        import nibabel as nib
        import numpy as np
        import yaml
        from dipy.core.gradients import gradient_table
        from dipy.reconst.dti import TensorModel
        from dipy.reconst.mapmri import MapmriModel

        from diff_benchmark.preprocessing.utils_brain import dti_measure, mapmri_measure
        save_path.mkdir(parents=True, exist_ok=True)

        folder = Path(expanduser(self.config["base_path"]))
        diffusion_folder = folder / subject_id / "T1w" / "Diffusion"
        breakpoint()    
        # === Load DWI data ===
        dwi = nib.load(diffusion_folder / "data.nii.gz")
        data = dwi.get_fdata()
        affine = dwi.affine

        bvals = np.loadtxt(diffusion_folder / "bvals")
        bvecs = np.loadtxt(diffusion_folder / "bvecs")
        gtab = gradient_table(bvals, bvecs)
        if any(m in self.metrics_to_compute for m in {"FA", "MD", "AD", "RD"}):
            print("Fitting DTI model...")
            dti_model = TensorModel(gtab)
            dti_fit = dti_model.fit(data)
            dti_measure(dti_fit, affine, self.metrics_to_compute, save_path)

        if any(m in self.metrics_to_compute for m in {"RTOP", "RTAP", "RTPP"}):
            print("Fitting MAPMRI model...")
            map_model = MapmriModel(gtab)
            map_fit = map_model.fit(data)
            mapmri_measure(map_fit, affine, self.metrics_to_compute, save_path)

        print(f"[{subject_id}] Metrics saved to: {save_path}")

    def save_subject_info(self, subject_id: str):
        print(f"[{subject_id}] Saving subject info... (placeholder)")

    def save_dataset_info(self):
        print("Saving dataset info... (placeholder)")
