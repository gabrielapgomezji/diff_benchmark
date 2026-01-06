import pandas as pd

def metrics_to_rows(
    metrics: dict,
    *,
    run_id: str,
    model_name: str,
    dataset: str,
    prediction_task: str,
    fold: int,
    split: str,
):
    rows = []
    for metric_name, metric_value in metrics.items():
        rows.append(
            {
                "run_id": run_id,
                "model_name": model_name,
                "dataset": dataset,
                "prediction_task": prediction_task,
                "fold": fold,
                "split": split,
                "metric": metric_name,
                "value": float(metric_value),
            }
        )
    return rows
