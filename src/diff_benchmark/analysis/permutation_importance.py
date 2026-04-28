from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def build_group_permutation_importance_records(
    result: Mapping[str, Any],
    *,
    model_name: str,
    run_id: str,
    dataset: str,
    tissue_type: str,
    primary_metric: str,
    prediction_task: str,
    fold: int,
    split: str,
    metadata_fields: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert a group permutation importance result into flat parquet rows."""
    metadata_fields = dict(metadata_fields or {})

    importances = np.asarray(result.get("importances", []), dtype=float)
    selected_regions = [int(r) for r in result.get("selected_regions", [])]
    region_labels = [int(r) for r in result.get("region_labels", [])]
    baseline_score = float(result.get("baseline_score", np.nan))
    scoring = result.get("scoring", None)
    n_repeats = int(result.get("n_repeats", importances.shape[1] if importances.ndim == 2 else 0))

    if importances.ndim != 2:
        raise ValueError("result['importances'] must be a 2D array-like structure.")

    records: list[dict[str, Any]] = []
    importances_mean = np.asarray(result.get("importances_mean", []), dtype=float)
    importances_std = np.asarray(result.get("importances_std", []), dtype=float)

    for region_idx, region_id in enumerate(selected_regions):
        region_label = region_labels[region_id] if 0 <= region_id < len(region_labels) else region_id
        region_values = importances[region_idx]
        region_mean = float(importances_mean[region_idx]) if importances_mean.size > region_idx else float(np.mean(region_values))
        region_std = float(importances_std[region_idx]) if importances_std.size > region_idx else float(np.std(region_values))

        for repeat_idx, importance_value in enumerate(region_values):
            records.append(
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "dataset": dataset,
                    "prediction_task": prediction_task,
                    "tissue_type": tissue_type,
                    "primary_metric": primary_metric,
                    "fold": int(fold),
                    "split": str(split),
                    "selected_region": int(region_id),
                    "selected_region_label": int(region_label),
                    "repeat": int(repeat_idx),
                    "importance": float(importance_value),
                    "importance_mean": region_mean,
                    "importance_std": region_std,
                    "baseline_score": baseline_score,
                    "scoring": scoring,
                    "n_repeats": n_repeats,
                    **metadata_fields,
                }
            )
    return records


def save_group_permutation_importance(
    records: Sequence[Mapping[str, Any]],
    *,
    output_root: str | Path,
    model_name: str,
    run_id: str,
    fold: int | None = None,
    split: str | None = None,
) -> Path:
    """Save permutation importance records into a cumulative parquet file."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    parquet_path = output_root / "permutation_importance.parquet"
    df_new = pd.DataFrame(list(records))

    if parquet_path.exists():
        df_prev = pd.read_parquet(parquet_path)
        df_all = pd.concat([df_prev, df_new], ignore_index=True)
    else:
        df_all = df_new

    key_cols = [
        c
        for c in ("run_id", "model_name", "subject_id", "fold", "split", "selected_region", "repeat")
        if c in df_all.columns
    ]
    if key_cols:
        df_all = df_all.drop_duplicates(subset=key_cols, keep="last")
    else:
        df_all = df_all.drop_duplicates(keep="last")

    if "fold" in df_all.columns:
        df_all = df_all.sort_values(by=["fold", "selected_region", "repeat"], kind="stable").reset_index(drop=True)

    preferred_prefix = [
        "run_id",
        "model_name",
        "dataset",
        "tissue_type",
        "primary_metric",
        "prediction_task",
        "fold",
        "split",
        "selected_region",
        "selected_region_label",
        "repeat",
        "importance",
        "importance_mean",
        "importance_std",
        "baseline_score",
        "scoring",
        "n_repeats",
    ]
    ordered_prefix = [c for c in preferred_prefix if c in df_all.columns]
    remaining_cols = [c for c in df_all.columns if c not in ordered_prefix]
    df_all = df_all[ordered_prefix + remaining_cols]

    df_all.to_parquet(parquet_path, index=False)
    return parquet_path