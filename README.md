# DiffBench — Diffusion MRI Prediction Benchmark

**DiffBench** is an open-source, reproducible benchmarking framework for evaluating machine learning and deep learning models on diffusion MRI (dMRI) microstructure-based prediction tasks. It enables controlled, systematic comparison across:

- **Datasets**: HCP, CamCAN, ABIDE
- **Tissue types**: gray matter (cortical surface), white matter skeleton
- **Microstructural metrics**: MD, MK, SH power, b0
- **Model families**: classical ML pipelines, 2D/3D CNNs, and vision foundation models (DINOv2, MedicalNet)
- **Prediction tasks**: binary classification and regression of cognitive/demographic targets

---

## Installation

Requires **Python ≥ 3.12**.

```bash
# Clone the repository
git clone <repo-url>
cd diff_benchmark

# Install with pip (editable)
pip install -e .

# Or with Poetry
poetry install
```

---

## Workflow Overview

The benchmark follows a three-step pipeline:

```
1. Preprocessing (raw dMRI)  →  2. Feature extraction  →  3. Training & evaluation
```

### Step 1 — Preprocessing

Raw dMRI data must be preprocessed before running the benchmark. This is handled by the `diffusion-preprocessing` submodule, which wraps dMRIPrep-based pipelines and can be run via Docker or Singularity.

➡️ See [`diffusion-preprocessing/README.md`](diffusion-preprocessing/README.md) for full instructions.

### Step 2 — Feature Extraction

Compute microstructural maps (MD, MK, SH, b0) and project them to the cortical surface or white matter skeleton for all subjects in a dataset:

```bash
diffbenchmark-features dataset=hcp dataset.metric_to_compute=md dataset.tissue_type=gray
```

Optionally, pre-cache features from frozen deep learning backbone models to disk to speed up subsequent training:

```bash
diffbenchmark-cache dataset=hcp model=dinov2
```

### Step 3 — Run Benchmark

Train and evaluate models across cross-validation folds:

```bash
diffbenchmark-run dataset=hcp model=pca_forest pred_head=binary_classification target=gender dataset.metric_to_compute=md dataset.tissue_type=gray
```

---

## CLI Commands

All commands use [Hydra](https://hydra.cc/) for configuration. Dataset paths and compute settings are defined in `src/diff_benchmark/configs/cluster/`. Overrides are passed as `key=value` arguments.

### `diffbenchmark-features`

Compute and store microstructural features for all subjects in a dataset.

```bash
diffbenchmark-features dataset=hcp dataset.metric_to_compute=md dataset.tissue_type=gray
```

### `diffbenchmark-run`

Run a full benchmark experiment (cross-validation, metrics, per-fold predictions).

```bash
diffbenchmark-run \
    dataset=hcp \
    model=linear \
    pred_head=regression \
    target=age \
    dataset.metric_to_compute=md \
    dataset.tissue_type=white
```

### `diffbenchmark-cache`

Pre-compute and cache features from a frozen backbone model to disk.

```bash
diffbenchmark-cache dataset=hcp model=dinov2
```

### `diffbenchmark-analysis`

Analyse experiment results and generate summary tables and plots.

```bash
# Both tables and plots (default)
diffbenchmark-analysis

# Only summary tables
diffbenchmark-analysis plots=false

# Only plots
diffbenchmark-analysis tables=false

# Force recompute of all plots
diffbenchmark-analysis force_plots=true

# Include debug plots for incomplete/failed runs
diffbenchmark-analysis analysis.debug=true
```

---

## Configuration

The benchmark is configured via Hydra YAML files in `src/diff_benchmark/configs/`. The root config is [`main.yaml`](src/diff_benchmark/configs/main.yaml).

### Key configuration groups

| Group | Available options |
|---|---|
| `dataset` | `hcp`, `camcan`, `abide` |
| `model` | `linear`, `pca_linear`, `lasso`, `forest`, `pca_forest`, `svm`, `pca_svm`, `dummy_classifier`, `dummy_regressor`,  `dinov2`, `curia`, `medicalnet`|
| `pred_head` | `binary_classification`, `regression` |
| `target` | `gender`, `age`, `dx_group` |
| `dataset.tissue_type` | `gray`, `white` |
| `dataset.metric_to_compute` | `md`, `mk`, `sh`, `b0` |

### Setting up dataset paths

Dataset paths are set per compute environment in `src/diff_benchmark/configs/cluster/`. Create or copy an existing cluster config file and point it to your data:

```yaml
# src/diff_benchmark/configs/cluster/my_env.yaml
name: my_env
paths:
  hcp:
    base_dir: /path/to/HCP/raw
    results_dir: /path/to/HCP/preprocessed
    csv_file: /path/to/HCP/demographics.csv
  camcan:
    base_dir: /path/to/camcan/raw
    results_dir: /path/to/camcan/preprocessed
    csv_file: /path/to/camcan/demographics.csv
```

Then activate it with `cluster=my_env` in any command.

---

## Outputs

Experiment results are saved under `exp_outputs/experiments/exp_<run_id>/`:

```
exp_outputs/experiments/exp_<run_id>/
├── config.yaml         # Full Hydra configuration used
├── metadata.yaml       # Run metadata (model, dataset, status, timing)
├── metrics/            # Per-fold metrics (Parquet)
├── predictions/        # Per-fold predictions (Parquet)
├── debug/              # Debug training curves
└── logs/               # Run logs
```

Summary tables and plots from `diffbenchmark-analysis` are saved under `exp_outputs/`.

---

## Adding a New Model

1. Create a new script under `src/diff_benchmark/models/` (in `deep_models/` or `sklearn_models/`).
2. Subclass one of the three abstract base classes from `src/diff_benchmark/models/`:
   - `NumpyAbstractModel` — for classical `sklearn`-style pipelines
   - `TorchAbstractModel` — for PyTorch models with a custom training loop
3. Implement the required methods: `fit`, `predict`, and `_dataloader_to_numpy`.
4. Register your model in `src/diff_benchmark/models/model_configurations.py`.
5. Add a corresponding YAML config in `src/diff_benchmark/configs/model/`. 
