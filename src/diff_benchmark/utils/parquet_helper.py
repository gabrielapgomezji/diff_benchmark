from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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

    Only the key columns are held in RAM; the full dataset is streamed from
    disk at write time so memory usage stays proportional to the number of
    unique key combinations, not the total row count.

    Args:
        path: Destination ``.parquet`` file path.
        key_columns: Columns used to identify duplicate rows.
        columns: Accepted for backward compatibility; no longer used.
    """

    def __init__(self, path: Path, key_columns: list[str], columns: list[str] | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_columns = key_columns
        self._pending_rows: list[dict] = []
        # Only the key columns are kept in RAM for duplicate detection – much
        # cheaper than holding the full DataFrame.
        self._existing_keys: pd.DataFrame = self._load_existing_keys()

    def _load_existing_keys(self) -> pd.DataFrame:
        if self.path.exists() and self.key_columns:
            return pq.read_table(self.path, columns=self.key_columns).to_pandas()
        return pd.DataFrame(columns=self.key_columns)

    def add_rows(self, rows: list[dict]) -> None:
        """Buffer *rows* for the next :meth:`save` call.

        Args:
            rows: List of row dicts to buffer.
        """
        self._pending_rows.extend(rows)

    def save(self) -> None:
        """Flush buffered rows to disk, skipping any duplicates already on disk.

        New rows are streamed into a temp file alongside the existing row
        groups so that the full dataset is never duplicated in RAM.

        Does nothing if there are no pending rows.
        """
        if not self._pending_rows:
            return

        df_new = pd.DataFrame(self._pending_rows)

        if not self._existing_keys.empty and self.key_columns:
            # Deduplicate against the lightweight key-only cache.
            merged = df_new.merge(
                self._existing_keys,
                on=self.key_columns,
                how="left",
                indicator=True,
            )
            df_new = merged[merged["_merge"] == "left_only"].drop(columns="_merge")

        if not df_new.empty:
            new_table = pa.Table.from_pandas(df_new, preserve_index=False)

            if self.path.exists():
                # Stream-copy existing row groups + append new ones without
                # loading the whole dataset into memory.
                tmp_path = self.path.with_suffix(".tmp.parquet")
                existing_pf = pq.ParquetFile(self.path)
                with pq.ParquetWriter(tmp_path, existing_pf.schema_arrow) as writer:
                    for batch in existing_pf.iter_batches():
                        writer.write_batch(batch)
                    writer.write_table(new_table)
                tmp_path.replace(self.path)
            else:
                pq.write_table(new_table, self.path)

            # Extend the in-memory key cache with the newly written keys.
            if self.key_columns:
                new_keys = df_new[self.key_columns]
                self._existing_keys = pd.concat(
                    [self._existing_keys, new_keys], ignore_index=True
                )

        self._pending_rows = []
