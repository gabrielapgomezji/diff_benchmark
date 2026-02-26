from pathlib import Path

import pandas as pd


def metrics_to_rows(
    metrics: dict,
    *,
    run_id: str,
    model_name: str,
    dataset: str,
    prediction_task: str,
    tissue_type: str,
    primary_metric: str,
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
                "tissue_type": tissue_type,
                "primary_metric": primary_metric,
                "fold": fold,
                "split": split,
                "metric": metric_name,
                "value": float(metric_value),
            }
        )
    return rows


class ParquetSaver:
    """
    Utility to handle parquet saving with duplicate avoidance.
    Can be used for targets, predictions, or metrics.
    """

    def __init__(self, path: Path, key_columns: list, columns: list = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_columns = key_columns
        self.columns = columns
        self._load_existing()
        self._pending_rows = []

    def _load_existing(self):
        if self.path.exists():
            self.df_existing = pd.read_parquet(self.path)
        else:
            self.df_existing = pd.DataFrame(
                columns=self.columns if self.columns else []
            )

    def add_rows(self, rows: list[dict]):
        """Add new rows to be saved later; filter duplicates when saving."""
        self._pending_rows.extend(rows)

    def save(self):
        """Concatenate pending rows with existing data and save to parquet"""
        if not self._pending_rows:
            return  # nothing to save

        df_new = pd.DataFrame(self._pending_rows)
        if not self.df_existing.empty and self.key_columns:
            merged = df_new.merge(
                self.df_existing[self.key_columns],
                on=self.key_columns,
                how="left",
                indicator=True,
            )
            df_to_add = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
        else:
            df_to_add = df_new

        if not df_to_add.empty:
            df_out = pd.concat([self.df_existing, df_to_add], ignore_index=True)
            df_out.to_parquet(self.path, index=False)
            self.df_existing = df_out  # update existing
        self._pending_rows = []  # clear after saving
