from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

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
    
    