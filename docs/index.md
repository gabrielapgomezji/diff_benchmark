# DiffBench

DiffBench is a Python library for benchmarking machine learning pipelines for diffusion MRI.

It provides a reproducible and modular framework for preparing diffusion MRI data, extracting features, training predictive models, evaluating their performance, and comparing experiments.

## Benchmark workflow

DiffBench follows a modular workflow from diffusion MRI data to benchmark results:

```text
Diffusion MRI data
        │
        ▼
1. Preprocessing
        │
        ▼
2. Data Preparation
        │
        ▼
3. Feature Extraction
        │
        ▼
4. Model Training & Evaluation
        │
        ▼
5. Analysis
```

Each stage can be configured and used independently depending on the experiment.

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} 1. Preprocessing
:link: user_guide/preprocessing
:link-type: doc

Preprocess diffusion MRI data and generate the inputs required by the benchmark.
:::

:::{grid-item-card} 2. Data Preparation
:link: user_guide/data_preparation
:link-type: doc

Prepare diffusion MRI representations required by the benchmark.
:::

:::{grid-item-card} 3. Feature Extraction
:link: user_guide/feature_extraction
:link-type: doc

Transform prepared imaging data into model-ready features.
:::

:::{grid-item-card} 4. Model Training
:link: user_guide/training
:link-type: doc

Train and compare machine learning and deep learning models.
:::

:::{grid-item-card} 5. Analysis
:link: user_guide/analysis
:link-type: doc

Aggregate metrics, compare experiments, and generate reports and plots.
:::

::::

For a more detailed explanation of how these stages interact, see the
[Pipeline Overview](overview/pipeline.md).

---

## Documentation

Choose where you want to start:

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} 🚀 Getting Started
:link: getting_started/index
:link-type: doc

Install DiffBench and run your first experiment.
:::

:::{grid-item-card} 📖 User Guide
:link: user_guide/index
:link-type: doc

Learn how data preparation, feature extraction, training, evaluation, and analysis work.
:::

:::{grid-item-card} 🧠 Pipeline Overview
:link: overview/pipeline
:link-type: doc

Understand the architecture and how the different components interact.
:::

:::{grid-item-card} 🔧 Reference
:link: reference/index
:link-type: doc

Explore configuration options, available models, metrics, and generated results.
:::

:::{grid-item-card} 🧩 Tutorials
:link: tutorials/index
:link-type: doc

Follow step-by-step guides for extending DiffBench.
:::

:::{grid-item-card} 💻 API Reference
:link: api/index
:link-type: doc

Explore the Python classes, functions, and implementation interfaces.
:::


::::

```{toctree}
:hidden:
:maxdepth: 1

overview/index
getting_started/index
user_guide/index
reference/index
tutorials/index
api/index
```