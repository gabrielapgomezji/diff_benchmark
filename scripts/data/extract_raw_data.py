from os.path import expanduser
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import yaml

# from dipy.core.gradients import gradient_table
from joblib import Parallel, delayed
from nilearn import surface


def process_subject(sub: str, config: dict):
    """Processes diffusion-weighted imaging (DWI) data for a given subject by projecting
    the volumetric data onto the cortical surface and saving the results in HDF5 format.
    Args:
        sub (str): Subject identifier.
        config (dict): Configuration dictionary containing paths and settings.
            Expected keys:
                - "base_path" (str): Base folder containing subject data.
                - "data_path" (str): Folder containing additional data files.
                - "deen_left" (str): Path to left hemisphere surface labels.
                - "results_path" (str): Directory to save processed data.

    Returns:
        None: The function saves processed data to an HDF5 file and skips processing
        if the output file already exists or required files are missing.

    Raises:
        Exception: If any error occurs during processing, it is caught and logged.
    """
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

    # Skip if output already exists
    if raw_data_output.exists():
        print(f"[{sub}] Skipped (already processed)")
        return

    # Ensure all required files exist
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
        # Load data
        dwi = nib.load(dwi_path)
        bvecs = np.loadtxt(bvecs_path)
        bvals = np.loadtxt(bvals_path)
        # gtab = gradient_table(bvals=bvals, bvecs=bvecs) # UNUSED VARIABLE

        image = nib.load(mask_path)
        _ = image.get_fdata()  # If you need this later, assign it

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


def main():
    """
    Main function to extract and process raw data for multiple subjects in parallel.
    This function reads configuration settings from a YAML file, identifies subject folders
    within a specified base directory, and processes each subject using parallel processing.
    Steps:
    1. Load configuration settings from 'scripts/configuration.yml'.
    2. Determine the base folder path and expand it to an absolute path.
    3. Identify all subject directories within the base folder.
    4. Process each subject in parallel using the `process_subject` function.
    Note:
    - The number of parallel jobs is set to 10.
    - Ensure the configuration file and subject directories exist before running this function.
    Raises:
        FileNotFoundError: If the configuration file or base folder does not exist.
        KeyError: If the 'folder' key is missing in the configuration file.
    Dependencies:
        - yaml: For loading the configuration file.
        - pathlib.Path: For handling file paths.
        - joblib.Parallel and joblib.delayed: For parallel processing.
    """

    with open("configuration.yaml", "r") as file:
        config = yaml.safe_load(file)
    base_folder = Path(expanduser(config["base_path"]))
    subjects = [p.name for p in base_folder.iterdir() if p.is_dir()]

    Parallel(n_jobs=10)(delayed(process_subject)(sub, config) for sub in subjects)


if __name__ == "__main__":
    main()
