# Adding a New Model

DiffBench can be extended with additional machine learning or deep learning models.

This tutorial describes the main steps required to integrate a new model into the benchmark.

Before starting, it is useful to understand the standard [Model Training](../user_guide/training.md) workflow.

## Model interface

A model integrated into DiffBench must expose the interface expected by the experiment runner.

At minimum, the model must support:

```python
fit(...)
predict(...)
```

The exact implementation depends on whether the model uses a scikit-learn-style estimator or a PyTorch `nn.Module`.

## Step 1 — Implement the model

Add the model implementation to the appropriate module under the DiffBench model package.

The model should follow the conventions of the existing implementations for the corresponding backend.

## Step 2 — Register the model

Add the model to the model registry so that it can be selected through the experiment configuration.

This allows the experiment runner to instantiate it from its configured name.

## Step 3 — Add its configuration

Create or update the corresponding Hydra configuration with the parameters required by the new model.

Model-specific parameters should remain in the model configuration rather than being hard-coded in the training pipeline.

## Step 4 — Run an experiment

Once registered, the model can be selected in the same way as any existing model.

For example:

```bash
diffbenchmark-run model=my_new_model
```

The model will then participate in the standard cross-validation, prediction, and evaluation workflow.

## Next steps

See the [Models Reference](../reference/models.md) for examples of existing models and the [Models API](../api/models.md) for details about the expected model interface.
