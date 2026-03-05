import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path


def is_successful_experiment(exp_dir: Path) -> bool:
    """Check if experiment completed successfully or has usable partial results."""
    metadata_path = exp_dir / "metadata.yaml"
    if not metadata_path.exists():
        return False

    metadata = OmegaConf.load(metadata_path)
    status = metadata.get("status")

    # Accept success, partial, or experiments with completed folds
    if status in ["success", "partial"]:
        return True

    # Also accept experiments that have metrics file (even if status is "crashed" or "running")
    metrics_file = exp_dir / "metrics" / "fold_metrics.parquet"
    return metrics_file.exists()


def table_all_runs(df: pd.DataFrame, primary_metric: str = "accuracy") -> pd.DataFrame:
    """
    Show ALL runs for each model (not just the best one).

    Args:
        df: Summary dataframe with mean/std per run
        primary_metric: Metric to filter on (accuracy, mae, rmse, etc.)

    Returns:
        DataFrame with all runs, sorted by model and metric value
    """
    # Find all metrics containing the primary metric string
    related_metrics = df[
        df["metric"].str.contains(primary_metric, case=False, na=False)
    ]["metric"].unique()

    df_filt = df[(df["metric"].isin(related_metrics)) & (df["split"] == "test")].copy()

    if df_filt.empty:
        return df_filt

    # Sort by model name and then by metric value
    # For accuracy/correlation: descending (higher is better)
    # For error metrics (mae, rmse, mse): ascending (lower is better)
    is_error_metric = any(
        err in primary_metric.lower() for err in ["mae", "rmse", "mse", "error"]
    )
    df_filt = df_filt.sort_values(
        ["model_name", "mean"], ascending=[True, is_error_metric]
    )

    # Select columns to display
    display_cols = ["model_name", "dataset", "run_id", "metric", "mean", "std"]
    if "tissue_type" in df_filt.columns:
        display_cols.insert(2, "tissue_type")
    if "prediction_task" in df_filt.columns:
        display_cols.insert(3, "prediction_task")

    # Keep only existing columns
    display_cols = [c for c in display_cols if c in df_filt.columns]

    return df_filt[display_cols].reset_index(drop=True)


def table_model_aggregate(
    df: pd.DataFrame, primary_metric: str = "accuracy"
) -> pd.DataFrame:
    """
    Show aggregate statistics for each model across all its runs.

    Args:
        df: Summary dataframe with mean/std per run
        primary_metric: Metric to filter on (accuracy, mae, rmse, etc.)

    Returns:
        DataFrame with mean/std/min/max across all runs per model
    """
    # Find all metrics containing the primary metric string
    related_metrics = df[
        df["metric"].str.contains(primary_metric, case=False, na=False)
    ]["metric"].unique()

    df_filt = df[(df["metric"].isin(related_metrics)) & (df["split"] == "test")].copy()

    if df_filt.empty:
        return df_filt

    # Aggregate by model name
    agg_dict = {"mean": ["mean", "std", "min", "max", "count"]}

    df_agg = df_filt.groupby(["model_name", "metric"]).agg(agg_dict).reset_index()

    # Flatten column names
    df_agg.columns = ["model_name", "metric", "mean", "std", "min", "max", "n_runs"]

    # Sort by model name
    # For accuracy/correlation: descending (higher is better)
    # For error metrics (mae, rmse, mse): ascending (lower is better)
    is_error_metric = any(
        err in primary_metric.lower() for err in ["mae", "rmse", "mse", "error"]
    )
    df_agg = df_agg.sort_values(["mean"], ascending=[is_error_metric])

    return df_agg.reset_index(drop=True)


def table_best_means(
    df: pd.DataFrame, primary_metric: str = "accuracy"
) -> pd.DataFrame:
    """Return the best-performing run per (dataset, task, model) group.

    Rows are first sorted by ``"mean"`` descending so that ``groupby.first()``
    selects the highest mean for each group.

    Args:
        df: Summary DataFrame with one row per run/metric/split.
        primary_metric: Metric name (or substring) to filter on.

    Returns:
        DataFrame containing the best run for each model/dataset group,
        restricted to test-split rows matching *primary_metric*.
    """

    # Find all metrics containing the primary metric string
    related_metrics = df[
        df["metric"].str.contains(primary_metric, case=False, na=False)
    ]["metric"].unique()

    df_filt = df[(df["metric"].isin(related_metrics)) & (df["split"] == "test")]

    df_sorted = df_filt.sort_values("mean", ascending=False)

    # Include tissue_type in grouping if it exists
    group_cols = ["dataset", "prediction_task", "metric", "model_name"]
    if "tissue_type" in df.columns:
        group_cols.insert(1, "tissue_type")
    if "primary_metric" in df.columns:
        group_cols.insert(2, "primary_metric")

    df_best = df_sorted.groupby(group_cols, as_index=False).first()

    return df_best


def select_best_runs(df: pd.DataFrame, primary_metric: str = "accuracy") -> dict:
    """Select the run_id with the highest mean test score per model/dataset group.

    Args:
        df: Summary DataFrame with columns including ``"metric"``, ``"split"``,
            ``"mean"``, and ``"run_id"``.
        primary_metric: Exact metric name to compare runs on.

    Returns:
        Dict mapping ``(dataset[, tissue_type], prediction_task, model_name)``
        tuples to the best ``run_id`` string.
    """

    df_test = df[(df["metric"] == primary_metric) & (df["split"] == "test")]

    best = {}

    group_cols = ["dataset", "prediction_task", "model_name"]
    if "tissue_type" in df.columns:
        group_cols.insert(1, "tissue_type")

    for keys, df_group in df_test.groupby(group_cols):
        best_row = df_group.loc[df_group["mean"].idxmax()]
        best[keys] = best_row["run_id"]

    return best


def table_detailed(
    df_metrics: pd.DataFrame, best_runs: dict, primary_metric: str = "accuracy"
) -> pd.DataFrame:
    """Build a long-format table of per-fold values for the best runs.

    Args:
        df_metrics: Full per-fold metrics DataFrame with columns
            ``"run_id"``, ``"metric"``, ``"fold"``, ``"split"``, ``"value"``.
        best_runs: Dict returned by :func:`select_best_runs` mapping group
            keys to ``run_id`` strings.
        primary_metric: Metric name (or substring) to include.

    Returns:
        Long-format DataFrame sorted by dataset/task/model/metric/fold/split.
    """

    rows = []

    # Find all related metrics
    related_metrics = df_metrics[
        df_metrics["metric"].str.contains(primary_metric, case=False, na=False)
    ]["metric"].unique()

    for keys, run_id in best_runs.items():
        # Handle both (dataset, task, model) and (dataset, tissue_type, task, model)
        if len(keys) == 4:
            dataset, tissue_type, task, model = keys
        else:
            dataset, task, model = keys
            tissue_type = None

        df_run = df_metrics[
            (df_metrics["run_id"] == run_id)
            & (df_metrics["metric"].isin(related_metrics))
        ]

        for _, row in df_run.iterrows():
            row_dict = {
                "dataset": dataset,
                "task": task,
                "model_name": model,
                "metric": row["metric"],
                "fold": row["fold"],
                "split": row["split"],
                "value": row["value"],
            }
            if tissue_type is not None:
                row_dict["tissue_type"] = tissue_type
            rows.append(row_dict)

    sort_cols = ["dataset", "task", "model_name", "metric", "fold", "split"]
    if any("tissue_type" in r for r in rows):
        sort_cols.insert(1, "tissue_type")

    return pd.DataFrame(rows).sort_values(sort_cols)


def table_folds_wide(
    df_metrics: pd.DataFrame,
    best_runs: dict,
    split: str = "test",
    primary_metric: str = "accuracy",
) -> pd.DataFrame:
    """Build a wide-format table with one column per fold for the best runs.

    Args:
        df_metrics: Full per-fold metrics DataFrame.
        best_runs: Dict returned by :func:`select_best_runs`.
        split: Which data split to include (``"train"`` or ``"test"``).
        primary_metric: Metric name (or substring) to include.

    Returns:
        Wide DataFrame with columns
        ``[dataset, [tissue_type,] task, model, metric, fold0, fold1, …, mean, std]``.
    """

    rows = []

    # Find all related metrics
    related_metrics = df_metrics[
        df_metrics["metric"].str.contains(primary_metric, case=False, na=False)
    ]["metric"].unique()

    for keys, run_id in best_runs.items():
        # Handle both (dataset, task, model) and (dataset, tissue_type, task, model)
        if len(keys) == 4:
            dataset, tissue_type, task, model = keys
        else:
            dataset, task, model = keys
            tissue_type = None

        for metric in related_metrics:
            df_run = df_metrics[
                (df_metrics["run_id"] == run_id)
                & (df_metrics["metric"] == metric)
                & (df_metrics["split"] == split)
            ]

            if df_run.empty:
                continue

            row = {
                "dataset": dataset,
                "task": task,
                "model": model,
                "metric": metric,
            }

            if tissue_type is not None:
                row["tissue_type"] = tissue_type

            for _, r in df_run.iterrows():
                row[f"fold{r['fold']}"] = r["value"]

            row["mean"] = df_run["value"].mean()
            row["std"] = df_run["value"].std()

            rows.append(row)

    df = pd.DataFrame(rows)

    fold_cols = sorted(
        [c for c in df.columns if c.startswith("fold")],
        key=lambda x: int(x.replace("fold", "")),
    )

    base_cols = ["dataset", "task", "model", "metric"]
    if "tissue_type" in df.columns:
        base_cols.insert(1, "tissue_type")

    return df[base_cols + fold_cols + ["mean", "std"]]


def print_table(df: pd.DataFrame) -> None:
    """Pretty-print *df* to stdout with 3 decimal places for floats.

    Args:
        df: DataFrame to print.  Prints ``"(empty table)"`` when *df* is empty.
    """
    if df.empty:
        print("(empty table)")
        return
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def table_weighted_aggregate(
    df_folds: pd.DataFrame,
    df_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute weighted aggregate metrics for each model across all datasets and tissue types.

    For classification: weighted accuracy
    For regression: weighted MAE

    Weights are based on the number of test samples in each dataset.
    """
    rows = []

    # Get test split data only
    df_test_folds = df_folds[df_folds["split"] == "test"]

    # Group by model and prediction task
    for (model_name, prediction_task), df_model in df_test_folds.groupby(
        ["model_name", "prediction_task"]
    ):

        # Determine which metric to use
        if prediction_task == "binary_classification":
            target_metric = "accuracy"
        elif prediction_task == "regression":
            target_metric = "mae"
        else:
            continue  # Skip unknown task types

        # Filter for the target metric
        df_metric = df_model[df_model["metric"] == target_metric]

        if df_metric.empty:
            continue

        # Count samples per dataset/tissue_type combination for weighting
        sample_counts = df_metric.groupby(["dataset", "tissue_type"]).size().to_dict()
        total_samples = sum(sample_counts.values())

        # Calculate weighted mean across all datasets and tissue types
        weighted_sum = 0
        for (dataset, tissue_type), count in sample_counts.items():
            # Get mean value for this dataset/tissue combination
            df_subset = df_metric[
                (df_metric["dataset"] == dataset)
                & (df_metric["tissue_type"] == tissue_type)
            ]
            subset_mean = df_subset["value"].mean()
            weight = count / total_samples
            weighted_sum += subset_mean * weight

        row = {
            "model_name": model_name,
            "prediction_task": prediction_task,
            "metric": target_metric,
            "weighted_mean": weighted_sum,
            "n_datasets": len(sample_counts),
            "total_folds": len(df_metric),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Sort by task and weighted mean (descending for accuracy, ascending for mae)
    df_classification = df[
        df["prediction_task"] == "binary_classification"
    ].sort_values("weighted_mean", ascending=False)
    df_regression = df[df["prediction_task"] == "regression"].sort_values(
        "weighted_mean", ascending=True  # Lower MAE is better
    )

    return pd.concat([df_classification, df_regression], ignore_index=True)
