# Configuration

## Training Configuration

Training behavior in DiffBench is controlled through Hydra configuration files.

For an overview of the training workflow, see [Model Training](../user_guide/training.md).

## Backend configuration

Parameters used by deep learning trainers are defined in the `backend` configuration group.

Typical options include:

| Parameter       | Description                                      |
| --------------- | ------------------------------------------------ |
| `epochs`        | Number of training epochs                        |
| `learning_rate` | Initial learning rate used by the optimizer      |
| `weight_decay`  | L2 regularization coefficient                    |
| `debug`         | Enables additional debugging or logging behavior |

Some models may expose additional model-specific parameters.

## Model-specific configuration

Model hyperparameters are defined through the corresponding model configuration.

These can include parameters such as:

* regularization strength;
* number of estimators;
* kernel settings;
* architecture-specific options;
* optimizer settings.

For the available model families, see the [Models Reference](models.md).

## Hydra overrides

Configuration values can be overridden directly from the command line.

For example:

```bash
diffbenchmark-run \
    model=medicalnet \
    backend.epochs=20 \
    backend.learning_rate=1e-4
```

This allows experiment settings to be changed without modifying the underlying configuration files.


## Analysis configuration

Analysis behavior is controlled through the `analysis` Hydra configuration group and command-line overrides.

| Parameter        | Default | Description                                                     |
| ---------------- | ------: | --------------------------------------------------------------- |
| `plots`          |  `true` | Generate visualisation plots                                    |
| `tables`         |  `true` | Build and save aggregated result tables                         |
| `force_plots`    | `false` | Regenerate plots even if they already exist                     |
| `analysis.debug` | `false` | Include incomplete or failed runs when generating debug outputs |

The analysis command reads experiment results from `exp_outputs/experiments/` by default.

The location of experiment outputs is configured through the paths configuration.
