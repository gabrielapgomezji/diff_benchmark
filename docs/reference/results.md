# Results Reference

DiffBench stores experiment-level and aggregated outputs in structured files so that results can be inspected, compared, or processed programmatically.

For an overview of the analysis workflow, see [Analysis](../user_guide/analysis.md).

## Analysis outputs

The analysis pipeline typically writes aggregated results under:

```text
exp_outputs/summary/
```

### `global_metrics.parquet`

Contains the fold-level metrics collected across experiments.

This table can be used when analyses require access to individual cross-validation folds rather than only experiment-level summaries.

### `summary_metrics.parquet`

Contains metrics aggregated across folds for each experiment.

Typical summaries include the mean and standard deviation of the evaluation metrics.

### `comprehensive_table.parquet`

Contains one row per experiment and combines:

* aggregated performance metrics;
* experiment identifiers;
* flattened experiment configuration values.

This table is useful for comparing experiment settings and filtering benchmark results.

### Dataset reports

Files following the pattern:

```text
<dataset>_report.txt
```

provide human-readable summaries for individual datasets, including experiment comparisons and best-performing runs.

### Coverage table

```text
coverage_table.txt
```

summarizes which combinations of datasets, models, targets, and tissue representations are available in the experiment results.

## Coverage table encoding

Coverage cells use a compact two-character code describing the tissue representation and prediction target.

The first character represents the tissue:

```text
g = gray matter
w = white matter
```

The second character represents the target:

```text
g = gender
a = age
d = diagnosis
```

For example:

```text
gg
```

represents gray matter with gender as the prediction target, while:

```text
wa
```

represents white matter with age as the prediction target.

## Plot outputs

Per-run plots are stored under:

```text
exp_outputs/plots/<run_id>/
```

The exact visualisations depend on the experiment and the selected analysis configuration.
