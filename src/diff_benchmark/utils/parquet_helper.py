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
) -> list[dict]:
    """Convert a metrics dictionary into a list of flat row dicts for Parquet storage.

    Each entry in the returned list corresponds to one metric and can be
    concatenated directly into a DataFrame.

    Args:
        metrics: Dict mapping metric name to its scalar value.
        run_id: Unique run identifier.
        model_name: Model name string.
        dataset: Dataset name.
        prediction_task: ``"binary_classification"`` or ``"regression"``.
        tissue_type: Tissue type string (e.g. ``"gray_matter"``).
        primary_metric: Primary microstructure metric (e.g. ``"sh"``, ``"md"``).
        fold: Cross-validation fold index.
        split: Data split (``"train"`` or ``"test"``).

    Returns:
        List of row dicts, one per metric.
    """
    return [
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
        for metric_name, metric_value in metrics.items()
    ]


class ParquetSaver:
    """Append-safe Parquet writer with duplicate-row filtering.

    Rows are buffered in memory and flushed to disk via :meth:`save`.
    On each flush, existing rows (identified by *key_columns*) are deduplicated
    so that re-running a fold never creates duplicate entries.

    Args:
        path: Destination ``.parquet`` file path.
        key_columns: Columns used to identify duplicate rows.
        columns: Expected column names (used to initialise an empty DataFrame
            when no file exists yet).
    """

    def __init__(self, path: Path, key_columns: list[str], columns: list[str] | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_columns = key_columns
        self._pending_rows: list[dict] = []
        self._existing = self._load_existing(columns)

    def _load_existing(self, columns: list[str] | None) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_parquet(self.path)
        return pd.DataFrame(columns=columns or [])

    def add_rows(self, rows: list[dict]) -> None:
        """Buffer *rows* for the next :meth:`save` call.

        Args:
            rows: List of row dicts to buffer.
        """
        self._pending_rows.extend(rows)

    def save(self) -> None:
        """Flush buffered rows to disk, skipping any duplicates already on disk.

        Does nothing if there are no pending rows.
        """
        if not self._pending_rows:
            return

        df_new = pd.DataFrame(self._pending_rows)

        if not self._existing.empty and self.key_columns:
            # Keep only rows whose key combination is not already stored.
            merged = df_new.merge(
                self._existing[self.key_columns],
                on=self.key_columns,
                how="left",
                indicator=True,
            )
            df_new = merged[merged["_merge"] == "left_only"].drop(columns="_merge")

        if not df_new.empty:
            combined = pd.concat([self._existing, df_new], ignore_index=True)
            combined.to_parquet(self.path, index=False)
            self._existing = combined

        self._pending_rows = []
