from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from typing import Iterable, Tuple, Union


@dataclass
class DatasetConfig:
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
    bvec_extensions: Union[str, Iterable[str]] = ("bvec",)
    bval_extensions: Union[str, Iterable[str]] = ("bval",)

    nodif_mask_extension: Optional[str] = None
    aparcaseg_extension: Optional[str] = None