# DiffBench Pipeline Overview

DiffBench follows a modular pipeline that transforms diffusion MRI data into benchmark results. Each stage can be used and understood independently, depending on whether you are interested in data preparation, feature extraction, model training, evaluation, or result analysis.

This page provides a high-level view of how the different components of the library interact. For implementation details and usage instructions, follow the links to the corresponding sections of the user guide.

## Pipeline overview

The benchmark is organized into four main stages:

```text
Raw dMRI data
      │
      ▼
1. Preprocessing and data preparation
      │
      ▼
2. Feature extraction and caching
      │
      ▼
3. Model training and evaluation
      │
      ▼
4. Analysis and reporting
```

Each stage produces the inputs required by the next one, while remaining sufficiently modular to allow individual components to be modified or extended.

---

## 1. Preprocessing and data preparation

The first stage prepares diffusion MRI data for use by the benchmark.

This stage is responsible for generating the imaging representations required by the downstream machine learning pipelines. Depending on the dataset and configuration, this may include the computation of microstructure maps and their projection or transformation into the spaces used by DiffBench.

The main components involved are:

```text
brain_feature_extraction.py
preparation_pipeline.py
```

At a high level:

```text
Raw diffusion MRI data
        │
        ▼
Microstructure computation
        │
        ▼
Spatial transformation / projection
        │
        ▼
Prepared representations
```

Users interested in preparing a new dataset or modifying the preprocessing workflow can start with the corresponding user guide sections.

See:

* [Data Preparation](../user_guide/data_preparation.md)
* [Preprocessing](../user_guide/preprocessing.md)

---

## 2. Feature extraction and caching

Once the imaging data has been prepared, DiffBench converts it into representations that can be used by machine learning models.

Depending on the selected pipeline, features may correspond directly to imaging measurements or may be obtained through a feature extractor or backbone model.

For computationally expensive feature extractors, DiffBench can optionally pre-compute and cache the resulting representations. This avoids recomputing the same features during repeated experiments or cross-validation runs.

The main components involved are:

```text
cache_features.py
cached_features.py
```

The general flow is:

```text
Prepared imaging data
        │
        ▼
Feature extraction
        │
        ├── Direct features
        │
        └── Backbone embeddings
                │
                ▼
             Caching
        │
        ▼
Model-ready features
```

See:

* [Feature Extraction](../user_guide/feature_extraction.md)

---

## 3. Model training and evaluation

The training stage runs the benchmark experiments.

DiffBench orchestrates cross-validation, model fitting, prediction, and evaluation according to the selected experimental configuration.

The main components involved are:

```text
run.py
trainer.py
```

`run.py` coordinates the execution of the experiment, while the trainer components handle the interaction with the selected machine learning or deep learning model.

The general flow is:

```text
Model-ready features
        │
        ▼
Cross-validation split
        │
        ▼
Model training
        │
        ▼
Prediction
        │
        ▼
Metric computation
```

The model implementation itself is separated from the experiment orchestration. This makes it possible to compare different modelling approaches under the same data splits and evaluation procedure.

Users who want to configure or understand model training can see:

* [Training](../user_guide/training.md)
* [Evaluation](../user_guide/evaluation.md)
* [Models Reference](../reference/models.md)
* [Metrics Reference](../reference/metrics.md)

Developers who want to integrate a new model should see:

* [Adding a New Model](../tutorials/add_model.md)

---

## 4. Analysis and reporting

After the benchmark runs have completed, DiffBench provides tools for aggregating and analysing the generated results.

The main analysis component is:

```text
analysis.py
```

This stage operates on the predictions and metrics generated during training and evaluation.

Typical outputs include:

```text
Cross-validation results
        │
        ▼
Aggregation across folds and runs
        │
        ├── Summary tables
        ├── Performance plots
        ├── Model comparisons
        └── Learning curves
```

This separation between experiment execution and analysis makes it possible to reuse previously generated results without rerunning model training.

See:

* [Analysis](../user_guide/analysis.md)
* [Evaluation](../user_guide/evaluation.md)

---

## Pipeline components

The table below summarizes the main components of the pipeline and their corresponding command-line entry points.

| Stage                          | Main component                                           | Purpose                                                                      | CLI entry point                 |
| ------------------------------ | -------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------- |
| Preprocessing                  | `brain_feature_extraction.py`, `preparation_pipeline.py` | Compute the required dMRI representations and prepare them for the benchmark | `diffbenchmark-features`        |
| Feature extraction and caching | `cache_features.py`, `cached_features.py`                | Extract and optionally cache model-ready features                            | `diffbenchmark-cache`           |
| Model training                 | `trainer.py`                                             | Implement the training backend and model interaction                         | Called by `diffbenchmark-run`   |
| Experiment orchestration       | `run.py`                                                 | Run cross-validation, training, prediction, and evaluation                   | `diffbenchmark-run`             |
| Analysis                       | `analysis.py`                                            | Aggregate results and generate tables, plots, and learning curves            | `diffbenchmark-analysis`        |
| Utilities                      | Job management, run identifiers, caching utilities       | Support experiment execution and reproducibility                             | Used internally by the pipeline |

---

## Where should I start?

You do not need to understand every stage of DiffBench before using or extending the library.

If you want to **run an existing benchmark**, start with the [Quickstart](../getting_started/quickstart.md).

If you want to **prepare a new dataset**, see [Data Preparation](../user_guide/data_preparation.md) and [Preprocessing](../user_guide/preprocessing.md).

If you want to **change how features are generated**, see [Feature Extraction](../user_guide/feature_extraction.md).

If you want to **configure or compare models**, see [Training](../user_guide/training.md) and the [Models Reference](../reference/models.md).

If you want to **understand how model performance is measured**, see [Evaluation](../user_guide/evaluation.md) and the [Metrics Reference](../reference/metrics.md).

If you want to **add a new machine learning or deep learning model**, follow the [Adding a New Model](../tutorials/add_model.md) tutorial.

If you want to **work directly with the Python implementation**, see the [API Reference](../api/index.md).

## Next steps

For a first use of DiffBench, the recommended path is:

```text
Installation
    │
    ▼
Quickstart
    │
    ▼
Pipeline Overview
    │
    ▼
Relevant User Guide section
```

The individual stages of the pipeline can then be explored independently depending on the experiment or component you want to modify.
