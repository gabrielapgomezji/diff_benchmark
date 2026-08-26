# Model Training

Model training is the stage where DiffBench fits predictive models on the prepared features and generates predictions for evaluation.

This stage runs after data preparation and, when applicable, feature extraction or feature caching.

For an overview of the complete workflow, see [Pipeline Overview](../overview/pipeline.md).

## Overview

DiffBench provides a common training workflow for both classical machine learning and deep learning models.

Although these models rely on different underlying frameworks, the benchmark handles them through a consistent interface. This allows different modelling approaches to be evaluated using the same cross-validation splits and experimental procedure.

At a high level, the training workflow is:

```text id="w2e16d"
Prepared features
        │
        ▼
Cross-validation split
        │
        ▼
Model initialization
        │
        ▼
Model fitting
        │
        ▼
Prediction
        │
        ▼
Evaluation
```

The exact training procedure depends on the selected model and backend.

---

## Inputs

The training stage receives:

* the prepared input features;
* the prediction target;
* the train and test samples for the current cross-validation fold;
* the selected model configuration;
* any training parameters required by the selected backend.

The input representation depends on the model. Classical machine learning models typically operate on tabular feature arrays, while deep learning models may consume data through PyTorch datasets and data loaders.

---

## Model types

DiffBench supports both:

* **classical machine learning models**, based on a scikit-learn-style interface;
* **deep learning models**, trained through PyTorch-based backends.

This common abstraction allows the main experiment workflow to remain independent of the specific model implementation.

For the complete list of available models and their options, see the [Models Reference](../reference/models.md).

---

## Cross-validation training

Model training is performed independently for each cross-validation fold.

For every fold, DiffBench:

1. creates the corresponding training and test subsets;
2. initializes the configured model;
3. fits the model using the training data;
4. generates predictions;
5. sends those predictions to the evaluation stage.

```text id="4o8amp"
Dataset
   │
   ▼
Cross-validation
   │
   ├── Fold 1 → Train → Predict
   ├── Fold 2 → Train → Predict
   ├── Fold 3 → Train → Predict
   └── ...
                  │
                  ▼
              Evaluation
```

Using the same cross-validation structure across models makes their results directly comparable within an experiment.

---

## Training configuration

Training behavior is controlled through Hydra configuration files.

The configuration determines, among other things:

* the model to train;
* model-specific hyperparameters;
* the training backend;
* optimization settings for deep learning models;
* experiment-specific training options.

For the complete configuration options, see the [Configuration Reference](../reference/configuration.md).

---

## Outputs

For each cross-validation fold, the training stage produces predictions for the corresponding data splits.

These predictions are then used by the evaluation stage to compute the configured performance metrics.

The resulting fold-level predictions and metrics can later be aggregated during the analysis stage.

---

## Related documentation

For more detailed information:

* See the [Models Reference](../reference/models.md) for the available models and their model-specific options.
* See the [Configuration Reference](../reference/configuration.md) for training and backend parameters.
* See the [Adding a New Model](../tutorials/add_model.md) tutorial if you want to integrate a new model.
* See the [Models API Reference](../api/models.md) for implementation details.

---

## Next step

Once the model has generated predictions for each fold, DiffBench evaluates them using the configured metrics.

Continue with [Evaluation](evaluation.md).
