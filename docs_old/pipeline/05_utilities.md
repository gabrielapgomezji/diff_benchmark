# Utilities

> **← [Back to index](index.md)**

This page covers the three shared utility modules used across the pipeline:

- [Feature Caching](#1-feature-caching) — `cache_features.py` / `cached_features.py`
- [Job Manager](#2-job-manager) — `job_manager.py`
- [Run ID & Experiment Fingerprinting](#3-run-id--experiment-fingerprinting) — `run_id.py`

---

## 1. Feature Caching

> **CLI:** `diffbenchmark-cache`

### Overview

Feature caching dramatically accelerates training for models with frozen, compute-intensive backbones (DINOv2 and CURIA). The backbone is run once over the full dataset and the resulting embeddings are stored as a Parquet file. Subsequent training runs load embeddings directly, bypassing the backbone forward pass entirely.

**Inputs:**
- A frozen backbone model
- A source `DataLoader` producing raw (normalised) MRI slices

**Outputs:**
- `<cache_dir>/dl_features/<model>_<dataset>_<hash>.parquet` — flat embedding table
- `<cache_dir>/dl_features/<model>_<dataset>_<hash>.meta.json` — metadata (sample count, embedding dim, augmentations)

### Configuration
Using the `configs/data` files from the Hydra configuration.

| Parameter | Default | Description |
|---|---|---|
| `num_augmentations` | `10` | Number of augmented copies per subject |
| `resize_shape` | `None` | Resize slices before caching (e.g., `[256, 256]` for HCP or None) |
| `norm_mean` / `norm_std` | `0.5` | Normalisation parameters applied before caching |

### Usage Example

```bash
# Pre-compute and cache DINOv2 features for HCP
diffbenchmark-cache dataset=hcp model=dinov2 data.num_augmentations=10

# Force recompute with resized inputs
diffbenchmark-cache dataset=hcp model=dinov2 data.resize_shape=[256,256] force_recompute=true
```

---

## 2. Job Manager

### Overview

`job_manager.py` provides a single `run_jobs()` function that dispatches a list of function calls either sequentially, in parallel via `joblib`, or distributed via SLURM using `submitit`. All backends share identical error handling through `fn_error_catcher`.

**Used by:**
- `run.py` — dispatching per-experiment training jobs
- `brain_feature_extraction.py` — dispatching per-subject preprocessing jobs

---

## 3. Run ID & Experiment Fingerprinting

### Overview

`run_id.py` generates deterministic, human-readable experiment identifiers and SHA-1 configuration fingerprints. This ensures that identical configurations map to the same experiment directory, enabling automatic cache lookup and deduplication of experiments in sweeps.

**Used by:** `main()` in `run.py`, before any job is dispatched.

### Run ID Format

```
<model>_<dataset><tissue><metric><target>_<8-char-hash>
```

**Example:** `pca_forest_hcwmda_3f2a91b0`

When `force=True`, a timestamp suffix is appended to guarantee uniqueness:
```
pca_forest_hcwmda_3f2a91b0_20260302-143512
```

### Abbreviation Tables

| Table | Example mappings |
|---|---|
| `TARGET_ABBR` | `"gender"` → `"g"`, `"age"` → `"a"`, `"diagnosis"` → `"d"` |
| `MICROSTRUCTURE_ABBR` | `"md"` → `"md"`, `"sh"` → `"sh"` |
| `TISSUE_TYPE_ABBR` | `"white_matter"` → `"w"`, `"gray_matter"` → `"g"` |
| `DATASET_ABBR` | `"hcp"` → `"h"`, `"camcan"` → `"c"`, `"abide"` → `"a"` |

### Keys Excluded from Fingerprint

The following top-level config keys are always excluded before hashing to prevent experiment IDs from changing due to infrastructure-only differences:

```python
EXCLUDE_KEYS = {"runtime", "hydra", "cluster", "slurm", "paths", "choices", "analysis"}
```

### Usage Example

```python
from omegaconf import OmegaConf
from pathlib import Path
from diff_benchmark.utils.run_id import make_run_id, is_cached, get_learning_curve_id

cfg = OmegaConf.load("configs/main.yaml")

run_id, exp_hash = make_run_id(cfg)
print(run_id)    # e.g. "pca_forest_hcwmda_3f2a91b0"
print(exp_hash)  # e.g. "3f2a91b0"

lc_id = get_learning_curve_id(cfg)  # groups all points of a learning curve

if is_cached(run_id, Path("exp_outputs/experiments")):
    print("Already computed — skipping.")
```