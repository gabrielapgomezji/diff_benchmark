import json
from pathlib import Path

import nibabel as nib
import numpy as np

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

    # Convert to dictionary: {index: [x,y,z]}
    coord_dict = {i: coords[i].tolist() for i in range(coords.shape[0])}

    return coords, coord_dict


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    subject_id = "100307"
    hcp_root = Path("/data/parietal/store4/data/HCP")
    subject_dir = hcp_root / subject_id
    # breakpoint()
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

    # Optionally save as numpy arrays:
    np.save("coords_L.npy", coords_L)
    np.save("coords_R.npy", coords_R)

    with open("coords_L.json", "w") as f:
        json.dump(dict_L, f)

    with open("coords_R.json", "w") as f:
        json.dump(dict_R, f)
