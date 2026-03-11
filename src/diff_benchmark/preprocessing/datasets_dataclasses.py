from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass
class DatasetConfig:
    """Configuration for a diffusion MRI dataset."""

    name: str
    base_dir: Path
    results_dir: Path
    data_reading: str
    metric_to_compute: str
    scale: int

    # Optional diffusion parameters
    big_delta: Optional[float] = None
    small_delta: Optional[float] = None
    big_delta_per_bvalue: Optional[Dict[int, float]] = None

    # Raw file descriptor extensions
    dwi_desc: str = "eddycorrected+bbreg"
    bvec_extensions: str | Iterable[str] = ("bvec",)
    bval_extensions: str | Iterable[str] = ("bval",)

    nodif_mask_extension: str | None = None
    aparcaseg_extension: str | None = None

    region: str | None = None
    tissue_type: str = "gray"  # "gray" or "white"

    # Target surface space: "fslr_32k" (HCP) or "fsaverage" (FreeSurfer/CamCAN)
    surface_space: str = "fslr_32k"

    # ---- Mesh-pipeline options (only used when data_type == "mesh") ----
    # Surface geometry to load from TemplateFlow for the mesh representation.
    # Common choices: "midthickness" (default), "inflated", "pial", "white".
    mesh_surface_type: str = "midthickness"
