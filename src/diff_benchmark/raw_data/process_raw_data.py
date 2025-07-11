import nibabel as nib
import numpy as np
import h5py
from pathlib import Path
from nilearn import surface
from os.path import expanduser

from diff_benchmark.raw_data.base import RawDataProcessor

class DWIProcessor(RawDataProcessor):
    def save_subject_info(self, sub: str):
        # You can implement saving subject-level metadata here
        print(f"[{sub}] Subject info saved (stub)")

    def save_dataset_info(self):
        # You can implement dataset-wide logging or metadata export
        print("Dataset info saved (stub)")

    def project_dwi_to_cortex(self, sub: str):
        config = self.config
        folder = Path(expanduser(config["base_path"]))
        data_folder = Path(config["data_path"])
        diffusion_folder = folder / sub / "T1w" / "Diffusion"
        surface_folder = folder / sub / "T1w" / "fsaverage_LR32k"
        ribbon_path = folder / sub / "MNINonLinear" / "ribbon.nii.gz"
        surface_left_pial = surface_folder / f"{sub}.L.pial_MSMAll.32k_fs_LR.surf.gii"
        surface_left_white = surface_folder / f"{sub}.L.white_MSMAll.32k_fs_LR.surf.gii"
        surface_left = surface_folder / f"{sub}.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
        deen_left = Path(expanduser(config["deen_path"]))
        save_dir = Path(config["results_path"]) / sub
        raw_data_output = save_dir / "raw_surface_data.h5"

        if raw_data_output.exists():
            print(f"[{sub}] Skipped (already processed)")
            return

        mask_path = data_folder / sub / "deen_subject.nii.gz"
        dwi_path = diffusion_folder / "data.nii.gz"
        bvecs_path = diffusion_folder / "bvecs"
        bvals_path = diffusion_folder / "bvals"

        required_files = [
            ribbon_path, surface_left_pial, surface_left_white, surface_left,
            mask_path, dwi_path, bvecs_path, bvals_path, deen_left,
        ]
        if not all(f.exists() for f in required_files):
            print(f"[{sub}] Skipped (missing required file)")
            return

        try:
            dwi = nib.load(dwi_path)
            bvecs = np.loadtxt(bvecs_path)
            bvals = np.loadtxt(bvals_path)

            _ = nib.load(mask_path).get_fdata()  # Placeholder for possible use

            left_dwi_surface = surface.vol_to_surf(
                dwi,
                surface_left_pial,
                interpolation="linear",
                kind="depth",
                inner_mesh=surface_left_white,
                mask_img=ribbon_path,
            )

            mesh_left = surface.load_surf_mesh(surface_left)
            left_labels = surface.load_surf_data(deen_left)
            nodes_left = left_labels.nonzero()[0]

            save_dir.mkdir(parents=True, exist_ok=True)
            with h5py.File(raw_data_output, "w") as f:
                f.create_dataset("left_dwi_surface", data=left_dwi_surface)
                f.create_dataset("surface_labels", data=left_labels)
                f.create_dataset("nodes_left", data=nodes_left)
                f.create_dataset("surface_coordinates", data=mesh_left.coordinates)
                f.create_dataset("surface_faces", data=mesh_left.faces)
                f.create_dataset("bvals", data=bvals)
                f.create_dataset("bvecs", data=bvecs)

                f.attrs["subject"] = sub
                f.attrs["hemisphere"] = "left"
                f.attrs["source"] = "projected DWI on surface"
                f.attrs["description"] = (
                    "Raw DWI signal projected on cortical surface using nilearn.surface.vol_to_surf."
                )
            print(f"[{sub}] Saved to {raw_data_output}")
        except Exception as e:
            print(f"[{sub}] Failed with error: {e}")
