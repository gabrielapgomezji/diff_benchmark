# Metrics Reference

DiffBench evaluates predictive models using task-specific performance metrics.

The metric used in an experiment depends on whether the prediction task is a **classification** or **regression** problem. Metrics are computed independently for each cross-validation fold and can later be aggregated across folds during the analysis stage.

For an overview of the evaluation workflow, see [Evaluation](../user_guide/evaluation.md).

---

## Overview

DiffBench currently distinguishes between two main types of prediction tasks:

| Task                  | Main metric       | Better value |
| --------------------- | ----------------- | ------------ |
| Binary classification | Balanced accuracy | Higher       |
| Regression            | R² score          | Higher       |

These metrics are used to compare models under the same cross-validation procedure.

At a high level:

```text id="22saqc"
Model predictions
        │
        ▼
Prediction task
        │
        ├── Classification ──► Balanced accuracy
        │
        └── Regression ──────► R²
        │
        ▼
Fold-level metric
        │
        ▼
Aggregation across folds
```

The aggregation and comparison of metrics across experiments is handled by the [Analysis](../user_guide/analysis.md) stage.

---

## Classification metrics

### Balanced accuracy

Balanced accuracy is the main classification metric used by DiffBench.

It measures classification performance while giving equal importance to each class.

For binary classification, it is defined as the average of the recall obtained for the two classes:

```text id="239x81"
Balanced Accuracy = (Sensitivity + Specificity) / 2
```

where:

* **Sensitivity** measures the proportion of positive samples correctly classified;
* **Specificity** measures the proportion of negative samples correctly classified.

Balanced accuracy ranges from `0` to `1`.

| Value   | Interpretation                                         |
| ------- | ------------------------------------------------------ |
| `1.0`   | Perfect classification                                 |
| `0.5`   | Chance-level performance for a balanced binary problem |
| `< 0.5` | Performance below chance                               |
| `0.0`   | Predictions are completely incorrect                   |

#### Why balanced accuracy?

Standard accuracy can give misleading results when the target classes are imbalanced.

Consider a dataset containing:

```text id="qaqmkp"
90 controls
10 patients
```

A model predicting every participant as a control would obtain:

```text id="f9msn3"
Accuracy = 90%
```

despite completely failing to identify patients.

Balanced accuracy evaluates performance independently for each class before averaging them, making it more appropriate when the number of samples per class differs.

#### Example

Suppose a model obtains:

```text id="3viixo"
Sensitivity = 0.80
Specificity = 0.90
```

Then:

```text id="dc77yv"
Balanced Accuracy = (0.80 + 0.90) / 2
                  = 0.85
```

The resulting balanced accuracy is therefore `0.85`.

#### In DiffBench

Balanced accuracy is computed independently for each cross-validation fold.

For example:

```text id="23ud7x"
Fold 1 ──► 0.81
Fold 2 ──► 0.84
Fold 3 ──► 0.79
Fold 4 ──► 0.83
Fold 5 ──► 0.82
              │
              ▼
       Mean ± variability
```

The fold-level values are preserved in the experiment outputs, while their aggregated statistics are generated during analysis.

See the [Results Reference](results.md) for the corresponding output files.

---

## Regression metrics

### R² score

The coefficient of determination, commonly referred to as **R²**, is the main regression metric used by DiffBench.

R² measures how well model predictions explain the variability of the target variable.

It is defined as:

```text id="xxwf5i"
           Σ (yᵢ - ŷᵢ)²
R² = 1 - ───────────────
           Σ (yᵢ - ȳ)²
```

where:

* `yᵢ` is the true target value;
* `ŷᵢ` is the predicted value;
* `ȳ` is the mean of the true target values.

#### Interpretation

Unlike many performance metrics, R² is not restricted to the interval `[0, 1]`.

| R²      | Interpretation                                         |
| ------- | ------------------------------------------------------ |
| `1.0`   | Perfect predictions                                    |
| `0.0`   | Equivalent to always predicting the mean of the target |
| `< 0.0` | Worse than predicting the target mean                  |

For example:

```text id="xbryv8"
R² = 0.70
```

means that the model explains approximately 70% of the variance in the target values for the evaluated data.

A negative value is possible:

```text id="rl1pmg"
R² = -0.20
```

This indicates that the predictions perform worse than a simple baseline that always predicts the mean of the target.

#### In DiffBench

R² is computed independently for each cross-validation fold:

```text id="vdh948"
Fold 1 ──► R²
Fold 2 ──► R²
Fold 3 ──► R²
Fold 4 ──► R²
Fold 5 ──► R²
              │
              ▼
       Mean ± variability
```

As with classification metrics, fold-level results are stored by the evaluation pipeline and aggregated during analysis.

---

## Baseline models and metric interpretation

Performance metrics should generally be interpreted relative to an appropriate baseline.

DiffBench includes baseline models that provide a reference against which predictive models can be compared.

For classification, a dummy classifier can represent a simple prediction strategy that does not learn meaningful relationships between the imaging features and target.

For regression, a dummy regressor can provide a baseline based on a simple target statistic such as the training-set mean.

The purpose of these models is not to obtain competitive performance, but to verify that more complex models provide useful predictive information beyond trivial prediction strategies.

For the available baseline models, see the [Models Reference](models.md).

---

## Metrics and cross-validation

DiffBench computes evaluation metrics separately for each cross-validation fold rather than calculating a single metric from the complete dataset.

For example:

```text id="0p4ktl"
                    Dataset
                       │
                       ▼
                Cross-validation
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
          Fold 1     Fold 2      ...
            │          │
            ▼          ▼
         Predict     Predict
            │          │
            ▼          ▼
         Metric      Metric
            └──────────┬──────────┘
                       ▼
                 Aggregation
```

This makes it possible to inspect both:

* average predictive performance;
* variability in performance across data splits.

The analysis stage produces the corresponding cross-fold summaries.

See [Analysis](../user_guide/analysis.md) for details.

---

## Fold-level and aggregated metrics

It is useful to distinguish between **evaluation** and **analysis** in DiffBench.

During evaluation:

```text id="iqs25h"
predictions
    │
    ▼
metric computation
    │
    ▼
fold-level metrics
```

During analysis:

```text id="o3tos6"
fold-level metrics
       │
       ▼
aggregation across folds
       │
       ▼
mean / variability
       │
       ▼
experiment comparisons
```

This separation preserves the individual fold results while also providing concise summaries for comparing experiments.

The generated metric tables are described in the [Results Reference](results.md).

---

## Choosing the appropriate metric

The evaluation metric must correspond to the prediction task.

For a binary classification task:

```text id="ghbkb7"
task = classification
        │
        ▼
Balanced accuracy
```

For a continuous prediction task:

```text id="94a29v"
task = regression
        │
        ▼
R²
```

Experiments with different prediction tasks should not be compared directly using their raw metric values.

For example, a balanced accuracy of `0.80` and an R² of `0.80` represent fundamentally different quantities.

---

## Related documentation

For more information:

* See [Evaluation](../user_guide/evaluation.md) for how predictions and metrics are produced during an experiment.
* See [Analysis](../user_guide/analysis.md) for how fold-level metrics are aggregated and compared.
* See the [Results Reference](results.md) for the files containing stored and aggregated metrics.
* See the [Models Reference](models.md) for available predictive and baseline models.
* See the [Configuration Reference](configuration.md) for experiment and evaluation settings.
