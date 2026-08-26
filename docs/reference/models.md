# Models Reference

DiffBench supports both classical machine learning models and deep learning models. The available model is selected through the experiment configuration.

For an overview of how models are trained within the benchmark, see [Model Training](../user_guide/training.md).

## Classical machine learning models

| Model              | Description                                                     |
| ------------------ | --------------------------------------------------------------- |
| `linear`           | Linear or logistic regression, depending on the prediction task |
| `pca_linear`       | PCA followed by a linear model                                  |
| `lasso`            | Lasso regression                                                |
| `forest`           | Random forest                                                   |
| `pca_forest`       | PCA followed by a random forest                                 |
| `svm`              | Support vector machine                                          |
| `pca_svm`          | PCA followed by a support vector machine                        |
| `dummy_classifier` | Majority-class baseline for classification                      |
| `dummy_regressor`  | Mean-prediction baseline for regression                         |

## Deep learning models

| Model        | Description                        |
| ------------ | ---------------------------------- |
| `dinov2`     | DINOv2 Vision Transformer backbone |
| `curia`      | CURIA backbone                     |
| `medicalnet` | MedicalNet 3D backbone             |

## Model selection

The model is selected through the corresponding Hydra configuration.

For example, an experiment may select a model using:

```text
model=linear
```

or:

```text
model=medicalnet
```

The exact configuration structure depends on the model implementation.

For training-related parameters, see the [Configuration Reference](configuration.md).

For details about extending DiffBench with a new model, see [Adding a New Model](../tutorials/add_model.md).
