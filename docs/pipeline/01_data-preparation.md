# Preprocessing

> **Pipeline position:** Step 1 — must be completed before any model training.  
> **CLI:** `diffbenchmark-features`  
> **← [Back to index](index.md)**

---

## Overview

The preprocessing component converts raw diffusion MRI (dMRI) data into subject-level microstructure feature maps suitable for downstream machine learning. It handles both cortical surface projections (gray matter) and white matter skeleton projections.

**Inputs:**
- Raw dMRI data in BIDS format (CamCAN, ABIDE) or HCP format
- Dataset configuration (`DatasetConfig`)

**Outputs:**
- Per-subject `.scalar.gii` files (GIFTI format) containing projected microstructure maps for both hemispheres
```
sub-<id>_hemi-<L|R>_param-<metric>_tissue-<type>.scalar.gii
```
- Per-subject `.nii.gz` files containing dwi_maps of the microstructure
```
sub-<id>_param-<metric>_tissue-<type>_dwimap.nii.gz
```
- Files are stored under `<results_dir>/default/derivatives/sub-<id>/dwi/`



---

## Configuration

All parameters are loaded from Hydra config files (`src/diff_benchmark/configs/dataset/`).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | ✓ | Dataset identifier |
| `data_reading` | `str` | ✓ | Format: `"hcp"`, `"bids"`, `"multicenter-bids"` |
| `metric_to_compute` | `str` | ✓ | One of `md`, `mk`, `sh` |
| `scale` | `int` | ✓ | Schaefer parcellation resolution |
| `tissue_type` | `str` | ✓ | `"gray"` or `"white"` |
| `surface_space` | `str` | ✗ | `"fslr_32k"` (default for HCP), `"fsaverage"` |
| `big_delta` / `small_delta` | `float` | ✗ | Diffusion gradient timing in milliseconds |
| `dwi_desc` | `str` | ✗ | DWI file descriptor suffix |

Cluster settings (parallelism, SLURM config, paths to the data) come from `configs/cluster/`.

---

## Usage Example

```python
from pathlib import Path
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.preprocessing.brain_feature_extraction import DefaultPipeline

dataset = DatasetConfig(
    name="camcan",
    base_dir=Path("/data/camcan/raw"),
    results_dir=Path("/data/camcan/processed"),
    data_reading="bids",
    metric_to_compute="md",
    scale=200,
    tissue_type="gray",
    surface_space="fsaverage",
)

pipeline = DefaultPipeline(dataset)

# Process a single subject
pipeline.compute_microstructure("CC110033")

# Process all subjects (with optional SLURM dispatch)
pipeline.run_pipeline(recompute=False)
```

**CLI:**
```bash
diffbenchmark-features \
    dataset=camcan \
    dataset.metric_to_compute=md \
    dataset.tissue_type=gray \
    cluster=my_env
```
