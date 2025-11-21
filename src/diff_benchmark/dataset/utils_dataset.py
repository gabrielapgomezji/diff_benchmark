import numpy as np
import json
from pathlib import Path


def is_valid_embedding(embeddings):
    """Return True if embeddings are not all NaNs."""
    for array in embeddings.values():
        if not np.isnan(array).all():
            return True
    return False

def load_precomputed_coordinates():
    """
    Load precomputed left/right hemisphere vertex coordinate files.

    Expected files:
        base_dir/coords_L.json
        base_dir/coords_R.json

    Returns:
        dict_L, dict_R

    Raises:
        FileNotFoundError if either file is missing.
    """
    coord_dir =  Path(__file__).parent.parent.parent.parent / "aux_files" / "vertex_coordinates"

    file_L = coord_dir / "coords_L.json"
    file_R = coord_dir / "coords_R.json"

    if not file_L.exists() or not file_R.exists():
        raise FileNotFoundError(
            f"Missing coordinate files.\n"
            f"Expected:\n  {file_L}\n  {file_R}\n"
            f"Please compute them first. Run the command: run python scripts/get_coordinates.py."
        )

    with open(file_L, "r") as f:
        dict_L = json.load(f)
    with open(file_R, "r") as f:
        dict_R = json.load(f)

    return dict_L, dict_R
