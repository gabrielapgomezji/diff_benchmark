# Main Run

> **Pipeline position:** Step 3 — orchestrates data preparation, training, and result persistence end-to-end.  
> **CLI:** `diffbenchmark-run`  
> **← [Back to index](index.md)**

---

## Overview

`run.py` is the orchestration layer that coordinates data preparation, model instantiation, cross-validated training, metric collection, and result persistence. It is the primary entry point for training experiments.

**Inputs:**
- A fully-resolved Hydra configuration (generated from `configs/main.yaml`)

**Outputs:**  
Per-experiment directory under `exp_outputs/experiments/exp_<run_id>/`:
```
├── config.yaml          # Full resolved configuration
├── metadata.yaml        # Experiment status, timing, job metadata
├── metrics/
│   └── fold_metrics.parquet   # One row per (fold, split, metric)
├── predictions/
│   ├── predictions.parquet    # Per-sample predictions for all folds
│   └── targets.parquet        # Ground-truth targets
├── debug/               # Training curves (if debug=True)
└── logs/
```

---


### `metadata.yaml` Lifecycle

| Status | Meaning |
|---|---|
| `"running"` | Written at job start |
| `"success"` | All `n_splits` folds completed without error |
| `"partial"` | Some folds completed; no crash detected |
| `"crashed"` | An exception was raised in a fold; `error` field populated |

---

## Configuration

Parameters are set in `configs/main.yaml` and its included sub-configs.

| Parameter | Location | Description |
|---|---|---|
| `random_state` | `main.yaml` | Define the random state of the experiment |
| `runtime.run_id` | `main.yaml` | Will store the run_id of the experiment |
| `runtime.force` | `main.yaml` | Skip cache check and always re-run |
| `runtime.learning_curve_exp` | `main.yaml` | Mark run as part of a learning curve sweep |
| `runtime.learning_curve_id` | `main.yaml` | Will store the assigned id if a data size analysis wants to be run |
| `experiment_hash` | `main.yaml` | Experiment identifier |

**Environment variables used at runtime:**
- `SLURM_JOB_ID` — recorded in `metadata.yaml` when running under SLURM

---

## Usage Example

**CLI:**
```bash
# Single experiment
diffbenchmark-run \
    dataset=hcp \
    model=pca_forest \
    pred_head=binary_classification \
    target=gender \
    dataset.metric_to_compute=md \
    dataset.tissue_type=gray

# Force rerun of a cached experiment
diffbenchmark-run dataset=hcp model=linear runtime.force=true

# Sweep over all configs defined in configs/main.yaml choices
diffbenchmark-run
```
