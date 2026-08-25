# Data Preparation

Data preparation is the first stage of the DiffBench workflow. It converts diffusion MRI data into subject-level imaging representations that can be used by the downstream benchmarking pipeline.

This stage must be completed before feature extraction and model training.

For an overview of the complete workflow, see [Pipeline Overview](../overview/pipeline.md).

## Overview

The data preparation pipeline processes diffusion MRI data and computes the representations required by DiffBench.

Depending on the selected configuration, it can generate:

* cortical surface representations for gray matter;
* white matter representations;
* diffusion-derived microstructure maps;
* subject-level files that can later be used for feature extraction and model training.

At a high level, the workflow is:

```text
Diffusion MRI data
        │
        ▼
Microstructure computation
        │
        ▼
Spatial transformation / projection (if required)
        │
        ▼
Prepared subject-level representations
```

The exact processing depends on the dataset and experiment configuration.

---

## Inputs

The pipeline requires:

* diffusion MRI data;
* a dataset configuration describing how the data should be read and processed;
* paths to the input data and output directory.

DiffBench supports different dataset layouts, including BIDS-compatible datasets and HCP-style data.

Dataset-specific options are defined through Hydra configuration files.

For the complete list of available dataset configuration options, see the [Configuration Reference](../reference/configuration.md).

---

## Outputs

Data preparation generates subject-level diffusion-derived representations that are consumed by later stages of the benchmark.

Typical outputs include cortical surface files in GIFTI format:

```text
sub-<id>_hemi-<L|R>_param-<metric>_tissue-<type>.scalar.gii
```

and volumetric diffusion maps:

```text
sub-<id>_param-<metric>_tissue-<type>_dwimap.nii.gz
```

These files are organized by subject in the configured results directory.

The exact files produced depend on the selected tissue type, metric, and spatial representation.

---

## Running data preparation

The recommended way to run this stage is through the command-line interface:

```bash
diffbenchmark-features \
    dataset=camcan \
    dataset.metric_to_compute=md \
    dataset.tissue_type=gray \
    cluster=my_env
```

Hydra overrides can be used to select the dataset and modify individual configuration options directly from the command line.

For example:

```bash
diffbenchmark-features \
    dataset=camcan \
    dataset.metric_to_compute=mk
```

The available options depend on the dataset configuration.

---

## Main configuration choices

The main choices controlling data preparation include:

* **Dataset** — the dataset to process.
* **Data format** — how DiffBench locates and reads the imaging files.
* **Microstructure metric** — the diffusion-derived quantity to compute.
* **Tissue type** — for example, gray or white matter.
* **Surface or spatial representation** — where the resulting measurements should be projected.
* **Output location** — where processed representations are stored.

More detailed configuration parameters, including dataset paths and cluster-specific settings, are documented in the [Configuration Reference](../reference/configuration.md).

---

## Parallel execution

Subjects can be processed independently, allowing the data preparation stage to be parallelized across available compute resources.

Cluster-specific settings, including SLURM configuration and execution paths, are handled separately from the dataset definition.

See the [Configuration Reference](../reference/configuration.md) for details about execution and cluster settings.

---

## Using the Python API

The data preparation pipeline can also be used directly from Python.

For example, a single subject can be processed with:

```python
from diff_benchmark.preprocessing.brain_feature_extraction import DefaultPipeline

pipeline = DefaultPipeline(dataset_config)

pipeline.compute_microstructure("subject_id")
```

The complete dataset can be processed with:

```python
pipeline.run_pipeline(recompute=False)
```

For the full class and method signatures, see the [Preprocessing API Reference](../api/preprocessing.md).

---

## Next step

Once the required subject-level representations have been generated, they can be used by the feature extraction stage.

Continue with [Feature Extraction](feature_extraction.md).
