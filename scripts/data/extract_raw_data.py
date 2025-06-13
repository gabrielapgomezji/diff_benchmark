import h5py
from pathlib import Path
import yaml
from os.path import expanduser
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from nilearn import surface
from joblib import Parallel, delayed

def process_subject(sub: str, config: dict):
    folder = Path(expanduser(config["folder"]))
    data_folder = Path(config["data_folder"])
    diffusion_folder = folder / sub / "T1w" / "Diffusion"
    surface_folder = folder / sub / "T1w" / "fsaverage_LR32k"
    ribbon_path = folder / sub / "MNINonLinear" / "ribbon.nii.gz"
    surface_left_pial = surface_folder / f"{sub}.L.pial_MSMAll.32k_fs_LR.surf.gii"
    surface_left_white = surface_folder / f"{sub}.L.white_MSMAll.32k_fs_LR.surf.gii"
    surface_left = surface_folder / f"{sub}.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
    deen_left = Path(expanduser(config["deen_left"]))
    save_dir = Path(config["save_folder"]) / sub
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
        ribbon_path, surface_left_pial, surface_left_white,
        surface_left, mask_path, dwi_path, bvecs_path,
        bvals_path, deen_left
    ]
    if not all(f.exists() for f in required_files):
        print(f"[{sub}] Skipped (missing required file)")
        return

    try:
        # Load data
        dwi = nib.load(dwi_path)
        bvecs = np.loadtxt(bvecs_path)
        bvals = np.loadtxt(bvals_path)
        gtab = gradient_table(bvals=bvals, bvecs=bvecs)

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
    config = yaml.safe_load(open("scripts/configuration.yml"))
    base_folder = Path(expanduser(config["folder"]))
    subjects = [p.name for p in base_folder.iterdir() if p.is_dir()]
    
    Parallel(n_jobs=10)(delayed(process_subject)(sub, config) for sub in subjects)

if __name__ == "__main__":
    main()
