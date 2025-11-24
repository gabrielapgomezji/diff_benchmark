import json
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

# def load_surface_coordinates(subject_dir, hemisphere="L", surface="pial"):
#     """
#     Load the physical XYZ coordinates from an HCP fsLR32k surface file.

#     Parameters
#     ----------
#     subject_dir : str or Path
#         Base directory of the subject (e.g., /HCP/100307)
#     hemisphere : "L" or "R"
#     surface : "white" or "pial"

#     Returns
#     -------
#     coords : ndarray of shape (n_vertices, 3)
#         XYZ coordinates for each vertex.
#     """
#     subject_dir = Path(subject_dir)
#     surf_path = (
#         subject_dir / "T1w" / "fsaverage_LR32k" /
#         f"{subject_dir.name}.{hemisphere}.{surface}.32k_fs_LR.surf.gii"
#     )

#     if not surf_path.exists():
#         raise FileNotFoundError(f"Surface file not found: {surf_path}")

#     gii = nib.load(str(surf_path))
#     coords = gii.darrays[0].data  # vertex coordinates

#     return coords


def load_surface_coordinates(subject_dir, hemisphere="L", surface="pial"):
    """
    Load the physical XYZ coordinates from an HCP fsLR32k surface file.

    Parameters
    ----------
    subject_dir : str or Path
        Base directory of the subject (e.g., /HCP/100307)
    hemisphere : "L" or "R"
    surface : "white" or "pial"

    Returns
    -------
    coords : ndarray of shape (n_vertices, 3)
        XYZ coordinates for each vertex.
    coord_dict : dict
        Dictionary mapping vertex index -> [x, y, z].
    """

    subject_dir = Path(subject_dir)

    surf_path = (
        subject_dir
        / "T1w"
        / "fsaverage_LR32k"
        / f"{subject_dir.name}.{hemisphere}.{surface}.32k_fs_LR.surf.gii"
    )

    if not surf_path.exists():
        raise FileNotFoundError(f"Surface file not found: {surf_path}")

    gii = nib.load(str(surf_path))
    coords = gii.darrays[0].data  # vertex coordinates (N, 3)
    R_OFFSET = 32492 if hemisphere.upper() == "R" else 0
    global_indices = np.arange(coords.shape[0]) + R_OFFSET
    coords_reindexed = coords.copy()
    # Convert to dictionary: {index: [x,y,z]}
    # coord_dict = {i: coords[i].tolist() for i in range(coords.shape[0])}
    coord_dict = {
        int(global_indices[i]): coords[i].tolist()
        for i in range(coords.shape[0])
    }
    return coords, coord_dict


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    config_path = (
        Path(__file__).parent.parent / "config/configuration_general.yaml"
    )
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    subject_id = "100307"
    hcp_root = Path(config["data_paths"]["hcp_base"])
    subject_dir = hcp_root / subject_id
    # coords_L = load_surface_coordinates(subject_dir, hemisphere="L", surface="pial")
    # coords_R = load_surface_coordinates(subject_dir, hemisphere="R", surface="pial")
    coords_L, dict_L = load_surface_coordinates(
        subject_dir, hemisphere="L", surface="pial"
    )
    coords_R, dict_R = load_surface_coordinates(
        subject_dir, hemisphere="R", surface="pial"
    )
    print("Left hemisphere coordinates:", coords_L.shape)
    print("Right hemisphere coordinates:", coords_R.shape)
    output_dir = Path(__file__).parent.parent / "aux_files/vertex_coordinates"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "coords_L.json", "w") as f:
        json.dump(dict_L, f)

    with open(output_dir / "coords_R.json", "w") as f:
        json.dump(dict_R, f)
