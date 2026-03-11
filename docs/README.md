# DiffBench Pipeline — Technical Documentation

This section provides a detailed technical reference for each component of the DiffBench pipeline. It is intended for developers onboarding to the project who need to understand the internals, extend the code, or debug issues.

## Pipeline Overview

The benchmark follows a four-stage pipeline:

```
Raw dMRI data
    │
    ▼
[1. Preprocessing]  ── brain_feature_extraction.py, preparation_pipeline.py
    │                   Computes microstructure maps and projects to surface
    ▼
[2. Feature Extraction / Caching]  ── cache_features.py, cached_features.py
    │                                  Optionally pre-computes backbone embeddings
    ▼
[3. Model Training & Evaluation]  ── run.py, trainer.py
    │                                 Cross-validated training and prediction
    ▼
[4. Analysis]  ── analysis.py
                   Aggregation, tables, plots, learning curves
```

## Sections

| Document | Component | CLI entry point |
|---|---|---|
| [00 — Preprocessing](../diffusion-preprocessing/README.md) | Computation of required dMRI files | Docker/singularity |
| [01 — Data Preparation](pipeline/01_data-preparation.md) | Raw dMRI → microstructure feature maps | `diffbenchmark-features` |
| [02 — Model Training](pipeline/02_model_training.md) | Trainer backends, model registry | *(called by run)* |
| [03 — Main Run](pipeline/03_main_run.md) | Orchestration, cross-validation loop | `diffbenchmark-run` |
| [04 — Analysis](pipeline/04_analysis.md) | Aggregation, reports, plots | `diffbenchmark-analysis` |
| [05 — Utilities](pipeline/05_utilities.md) | Feature caching, job manager, run ID | `diffbenchmark-cache` |


## Metrics Reference

### Classification (`binary_classification`)

| Metric key | Description |
|---|---|
| `accuracy` | Standard unweighted accuracy |
| `accuracy_weighted` | Accuracy with balanced class weights |
| `precision` | Binary precision |
| `recall` | Binary recall |
| `f1` | Binary F1 score |

### Regression (`regression`)

| Metric key | Description |
|---|---|
| `rmse` | Root mean squared error |
| `rmse_weighted` | RMSE with balanced sample weights |
| `mae` | Mean absolute error |
| `mae_weighted` | MAE with balanced sample weights |
| `r2` | R² (coefficient of determination) |
| `explained_variance` | Explained variance score |
| `mape` | Mean absolute percentage error |
| `pearson_correlation` | Pearson correlation coefficient |

All metrics are computed via `diff_benchmark.utils.scores.compute_metrics()`. Class weights are computed with `sklearn.utils.class_weight.compute_sample_weight("balanced", y_true)`.
