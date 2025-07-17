from os.path import expanduser
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from nilearn import surface

from diff_benchmark.raw_data.base import RawDataProcessor


class DWIProcessor(RawDataProcessor):
    """
    DWIProcessor is a class that processes diffusion-weighted imaging (DWI) data for subjects.
    This class inherits from RawDataProcessor and provides methods to save subject-level and dataset-wide metadata,
    as well as to project DWI data onto the cortical surface.
    Methods:
        save_subject_info(sub: str):
            Saves subject-level metadata for the given subject.
        save_dataset_info():
            Logs or exports dataset-wide metadata.
        project_dwi_to_cortex(sub: str):
            Projects DWI data onto the cortical surface for the specified subject.
            This method checks for the existence of required files, loads DWI and associated data,
            performs the projection, and saves the results in an HDF5 file.
            If the output file already exists or if any required files are missing, it skips processing.
    """

    def save_subject_info(self, sub):
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
        surface_left = (
            surface_folder / f"{sub}.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
        )
        deen_left = Path(expanduser(config["deen_left_path"]))
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
            ribbon_path,
            surface_left_pial,
            surface_left_white,
            surface_left,
            mask_path,
            dwi_path,
            bvecs_path,
            bvals_path,
            deen_left,
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

class DWIHemispheresProcessor(RawDataProcessor):
    """
    DWIProcessor is a class that processes diffusion-weighted imaging (DWI) data for subjects.
    This class inherits from RawDataProcessor and provides methods to save subject-level and dataset-wide metadata,
    as well as to project DWI data onto the cortical surface.
    Methods:
        save_subject_info(sub: str):
            Saves subject-level metadata for the given subject.
        save_dataset_info():
            Logs or exports dataset-wide metadata.
        project_dwi_to_cortex(sub: str):
            Projects DWI data onto the cortical surface for the specified subject.
            This method checks for the existence of required files, loads DWI and associated data,
            performs the projection, and saves the results in an HDF5 file.
            If the output file already exists or if any required files are missing, it skips processing.
    """

    def save_subject_info(self, sub):
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
        
        # Left Hemisphere
        surface_left_pial = surface_folder / f"{sub}.L.pial_MSMAll.32k_fs_LR.surf.gii"
        surface_left_white = surface_folder / f"{sub}.L.white_MSMAll.32k_fs_LR.surf.gii"
        surface_left = (
            surface_folder / f"{sub}.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
        )
        deen_left = Path(expanduser(config["deen_left_path"]))
        
        # Right Hemisphere
        surface_right_pial = surface_folder / f"{sub}.R.pial_MSMAll.32k_fs_LR.surf.gii"
        surface_right_white = surface_folder / f"{sub}.R.white_MSMAll.32k_fs_LR.surf.gii"
        surface_right = surface_folder / f"{sub}.R.midthickness_MSMAll.32k_fs_LR.surf.gii"
        deen_right = Path(expanduser(config["deen_right_path"]))

        
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
        ribbon_path,
        surface_left_pial, surface_left_white, surface_left, deen_left,
        surface_right_pial, surface_right_white, surface_right, deen_right,
        mask_path, dwi_path, bvecs_path, bvals_path,
    ]
        if not all(f.exists() for f in required_files):
            print(f"[{sub}] Skipped (missing required file)")
            return

        try:
            dwi = nib.load(dwi_path)
            bvecs = np.loadtxt(bvecs_path)
            bvals = np.loadtxt(bvals_path)

            _ = nib.load(mask_path).get_fdata()  # Placeholder for possible use

            # Process Left Hemisphere
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

            # Process Right Hemisphere
            right_dwi_surface = surface.vol_to_surf(
                dwi,
                surface_right_pial,
                interpolation="linear",
                kind="depth",
                inner_mesh=surface_right_white,
                mask_img=ribbon_path,
            )

            mesh_right = surface.load_surf_mesh(surface_right)
            right_labels = surface.load_surf_data(deen_right)
            nodes_right = right_labels.nonzero()[0]

            save_dir.mkdir(parents=True, exist_ok=True)
            with h5py.File(raw_data_output, "w") as f:
                # Save Left Hemisphere Data
                f.create_dataset("left_dwi_surface", data=left_dwi_surface)
                f.create_dataset("surface_labels", data=left_labels)
                f.create_dataset("nodes_left", data=nodes_left)
                f.create_dataset("surface_coordinates", data=mesh_left.coordinates)
                f.create_dataset("surface_faces", data=mesh_left.faces)
                
                # Save Right Hemisphere Data
                f.create_dataset("right_dwi_surface", data=right_dwi_surface)
                f.create_dataset("surface_labels_right", data=right_labels)
                f.create_dataset("nodes_right", data=nodes_right)
                f.create_dataset("surface_coordinates_right", data=mesh_right.coordinates)
                f.create_dataset("surface_faces_right", data=mesh_right.faces)

                # Common info
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
            
class DWISchaeferProcessor(RawDataProcessor):
    """
    Processes DWI data by projecting it to the cortical surface and aggregating it by Schaefer parcels.
    """

    def __init__(self, config: dict, schaefer_resampled: dict):
        super().__init__(config)
        self.schaefer_resampled = schaefer_resampled  # dict with 'left.data', 'right.data'

    def save_subject_info(self, sub: str):
        print(f"[{sub}] Subject info saved (stub)")

    def save_dataset_info(self):
        print("Dataset info saved (stub)")

    def project_dwi_to_cortex(self, sub: str):
        cfg = self.config
        schaefer = self.schaefer_resampled

        folder = Path(expanduser(cfg["base_path"]))
        diffusion_folder = folder / sub / "T1w" / "Diffusion"
        surface_folder = folder / sub / "T1w" / "fsaverage_LR32k"
        ribbon_path = folder / sub / "MNINonLinear" / "ribbon.nii.gz"
        save_dir = Path(cfg["results_path"]) / sub
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / "parcellated_dwi_schaefer.h5"

        if output_path.exists():
            print(f"[{sub}] Skipped: already processed.")
            return

        # Surface paths
        surface_L_pial = surface_folder / f"{sub}.L.pial_MSMAll.32k_fs_LR.surf.gii"
        surface_L_white = surface_folder / f"{sub}.L.white_MSMAll.32k_fs_LR.surf.gii"
        surface_R_pial = surface_folder / f"{sub}.R.pial_MSMAll.32k_fs_LR.surf.gii"
        surface_R_white = surface_folder / f"{sub}.R.white_MSMAll.32k_fs_LR.surf.gii"

        # DWI paths
        dwi_path = diffusion_folder / "data.nii.gz"
        bvecs_path = diffusion_folder / "bvecs"
        bvals_path = diffusion_folder / "bvals"

        required = [
            ribbon_path,
            surface_L_pial, surface_L_white,
            surface_R_pial, surface_R_white,
            dwi_path, bvecs_path, bvals_path,
        ]

        if not all(p.exists() for p in required):
            print(f"[{sub}] Skipped: missing files.")
            return

        try:
            # Load data
            dwi_img = nib.load(dwi_path)
            bvecs = np.loadtxt(bvecs_path)
            bvals = np.loadtxt(bvals_path)

            # Project to surface
            surf_L = surface.vol_to_surf(dwi_img, surface_L_pial, inner_mesh=surface_L_white, kind="depth", mask_img=ribbon_path)
            surf_R = surface.vol_to_surf(dwi_img, surface_R_pial, inner_mesh=surface_R_white, kind="depth", mask_img=ribbon_path)

            # Aggregate by Schaefer parcels
            parcels_L = schaefer["left.data"]
            parcels_R = schaefer["right.data"]
            unique_L = np.unique(parcels_L)
            unique_R = np.unique(parcels_R)

            n_directions = surf_L.shape[1]
            n_parcels = len(unique_L) + len(unique_R)
            parcellated_dwi = np.zeros((n_parcels, n_directions))

            for i, p in enumerate(unique_L):
                mask = parcels_L == p
                parcellated_dwi[i] = surf_L[mask].mean(axis=0)

            for i, p in enumerate(unique_R):
                mask = parcels_R == p
                parcellated_dwi[i + len(unique_L)] = surf_R[mask].mean(axis=0)

            # Save
            with h5py.File(output_path, "w") as f:
                f.create_dataset("dwi_parcellated", data=parcellated_dwi)
                f.create_dataset("bvecs", data=bvecs)
                f.create_dataset("bvals", data=bvals)
                f.attrs["subject"] = sub
                f.attrs["n_parcels"] = n_parcels
                f.attrs["description"] = "DWI projected to surface and averaged using Schaefer parcels"

            print(f"[{sub}] Saved to {output_path}")
        except Exception as e:
            print(f"[{sub}] Failed with error: {e}")
