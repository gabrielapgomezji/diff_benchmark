# Models API

This page describes the model interface used internally by DiffBench.

For a conceptual overview of model training, see [Model Training](../user_guide/training.md).

## Model creation

Models are instantiated through the model configuration utilities.

```python
from diff_benchmark.models.model_configurations import get_model
from omegaconf import OmegaConf

cfg = OmegaConf.load(
    "exp_outputs/experiments/exp_myrun/config.yaml"
)

model = get_model(
    cfg.model.name,
    OmegaConf.to_container(cfg, resolve=True),
)
```

## Training interface

Models expose a common training and prediction interface:

```python
model.set_fold(0)

model.fit(train_loader)

predictions = model.predict(test_loader)
```

`fit()` trains the model using the data associated with the current fold.

`predict()` returns the model predictions in a common format so that downstream evaluation does not depend on the underlying modelling framework.

This abstraction allows DiffBench to support both scikit-learn-style estimators and PyTorch-based models through the same experiment orchestration.
