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

    # Files extensions (nodif, aparcaseg for hcp)
    dwi_desc: str = "eddycorrected+bbreg"
    bvec_extensions: str | Iterable[str] = ("bvec",)
    bval_extensions: str | Iterable[str] = ("bval",)

    nodif_mask_extension: str | None = None
    aparcaseg_extension: str | None = None

    region: str | None = None
    tissue_type: str = "gray" #"white"
    
    # Surface space configuration
    surface_space: str = "fslr_32k"  # "fslr_32k" for HCP, "fsaverage" for CamCAN/FreeSurfer
