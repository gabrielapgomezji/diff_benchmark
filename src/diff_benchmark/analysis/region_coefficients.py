from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from diff_benchmark.preprocessing.utils.utils_brain_feature_extraction import (
    build_parcel_label_vector,
    load_template_surface,
    resample_schaefer_onto_fs_lr,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _cached_surface_atlas(scale: int, surface_space: str) -> dict[str, Any]:
    schaefer = resample_schaefer_onto_fs_lr(scale=scale, target_space=surface_space)
    left_vertices, left_faces = load_template_surface(
        hemi="L", space=surface_space, surf_type="midthickness"
    )
    right_vertices, right_faces = load_template_surface(
        hemi="R", space=surface_space, surf_type="midthickness"
    )
    parcel_labels = build_parcel_label_vector(
        schaefer,
        n_left=left_vertices.shape[0],
        n_right=right_vertices.shape[0],
    )
    return {
        "left_mesh": (left_vertices, left_faces),
        "right_mesh": (right_vertices, right_faces),
        "parcel_labels": parcel_labels,
        "n_left_vertices": int(left_vertices.shape[0]),
        "n_right_vertices": int(right_vertices.shape[0]),
        "n_regions": int(np.unique(parcel_labels[parcel_labels > 0]).size),
        "atlas_meta": schaefer.get("atlas_meta", {}),
    }


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _resolve_estimator(model: Any) -> Any:
    """Return a fitted estimator by unwrapping trainer/model wrappers."""
    candidate = model
    for _ in range(4):
        if hasattr(candidate, "best_estimator_"):
            return candidate.best_estimator_
        if hasattr(candidate, "model"):
            candidate = candidate.model
            continue
        break
    if hasattr(candidate, "best_estimator_"):
        return candidate.best_estimator_
    return candidate


def _resolve_linear_coef(estimator: Any, class_index: int | None = None) -> np.ndarray | None:
    coef = getattr(estimator, "coef_", None)
    if coef is None and hasattr(estimator, "named_steps"):
        steps = estimator.named_steps
        if "classifier" in steps and hasattr(steps["classifier"], "coef_"):
            coef = steps["classifier"].coef_
        elif "group_lasso" in steps and hasattr(steps["group_lasso"], "coef_"):
            coef = steps["group_lasso"].coef_
        elif "group_elastic_net" in steps and hasattr(steps["group_elastic_net"], "coef_"):
            coef = steps["group_elastic_net"].coef_

    if coef is None:
        return None

    coef = _as_numpy(coef)
    if coef.ndim == 1:
        return coef

    if coef.shape[0] == 1:
        return coef[0]

    idx = class_index if class_index is not None else 1
    idx = int(np.clip(idx, 0, coef.shape[0] - 1))
    return coef[idx]


def _region_keys(
    region_order: Sequence[int],
    metadata: Mapping[str, Any] | None = None,
) -> list[str]:
    metadata = metadata or {}
    region_name_map = metadata.get("region_name_map", {}) or {}
    hemisphere_map = metadata.get("hemisphere_map", {}) or {}

    keys: list[str] = []
    for rid in region_order:
        region_name = str(region_name_map.get(rid, rid))
        hemi = hemisphere_map.get(rid, None)
        if hemi is not None:
            keys.append(f"{hemi}:{region_name}")
        else:
            keys.append(region_name)
    return keys


def _group_slices_from_transformer(transformer: Any) -> tuple[list[int], list[slice]]:
    region_order = list(getattr(transformer, "region_order_", []))
    slices: list[slice] = []
    col = 0

    if hasattr(transformer, "region_feature_widths_"):
        for rid in region_order:
            width = int(transformer.region_feature_widths_[rid])
            slices.append(slice(col, col + width))
            col += width
        return region_order, slices

    if hasattr(transformer, "n_components_per_region_"):
        for rid in region_order:
            width = int(transformer.n_components_per_region_[rid])
            slices.append(slice(col, col + width))
            col += width
        return region_order, slices

    return region_order, slices


def _aggregate_region_weights(
    coef_vector: np.ndarray,
    region_order: Sequence[int],
    region_slices: Sequence[slice],
    agg: str = "l2",
) -> dict[int, float]:
    values: dict[int, float] = {}
    for rid, sl in zip(region_order, region_slices):
        block = coef_vector[sl]
        if block.size == 0:
            values[int(rid)] = 0.0
            continue
        if agg == "mean":
            values[int(rid)] = float(np.mean(block))
        elif agg == "abs_mean":
            values[int(rid)] = float(np.mean(np.abs(block)))
        else:
            values[int(rid)] = float(np.linalg.norm(block, ord=2))
    return values


def _extract_static_region_coefficients(
    estimator: Any,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, float] | None:
    metadata = metadata or {}
    agg = str(metadata.get("agg", "l2"))
    class_index = metadata.get("class_index", None)

    coef_vector = _resolve_linear_coef(estimator, class_index=class_index)
    if coef_vector is None:
        return None

    transformer = None
    if hasattr(estimator, "named_steps"):
        for step_name in ("region_features", "region_pca"):
            if step_name in estimator.named_steps:
                transformer = estimator.named_steps[step_name]
                break

    if transformer is None:
        return {"global": float(np.linalg.norm(coef_vector, ord=2))}

    region_order, region_slices = _group_slices_from_transformer(transformer)
    if not region_order or not region_slices:
        return {"global": float(np.linalg.norm(coef_vector, ord=2))}

    region_values = _aggregate_region_weights(coef_vector, region_order, region_slices, agg=agg)
    keys = _region_keys(region_order, metadata)
    return {key: float(region_values[int(rid)]) for key, rid in zip(keys, region_order)}


def _extract_additive_head_subject_coefficients(
    task_model: Any,
    X: Sequence[Any],
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, float]]:
    metadata = metadata or {}
    class_index = metadata.get("class_index", None)

    if not hasattr(task_model, "backbone") or not hasattr(task_model, "head"):
        return []
    if not hasattr(task_model.head, "parcel_contributions"):
        return []

    import torch

    if X is None:
        return []

    if isinstance(X, torch.Tensor):
        X_seq: list[Any] = [sample for sample in X]
    else:
        X_seq = list(X)
    if len(X_seq) == 0:
        return []

    def _chunks(seq: Sequence[Any], size: int):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    def _forward_contrib(model_obj: Any, x_batch: Sequence[Any]):
        x_input: Any = x_batch
        if x_batch:
            first = x_batch[0]
            if isinstance(first, torch.Tensor):
                x_input = torch.stack(list(x_batch), dim=0)
            elif isinstance(first, np.ndarray):
                x_input = torch.as_tensor(np.stack(x_batch, axis=0))
        feats = model_obj.backbone(x_input)
        return model_obj.head.parcel_contributions(feats)

    # Coefficient extraction happens after training and can trigger OOM for
    # large folds; process in chunks and transparently fall back to CPU.
    batch_size = int(metadata.get("coef_batch_size", 8)) if metadata else 8
    batch_size = max(1, batch_size)

    device = next(task_model.parameters()).device
    task_model.eval()

    contrib_blocks: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for x_batch in _chunks(X_seq, batch_size):
                contrib_blocks.append(_as_numpy(_forward_contrib(task_model, x_batch)))
    except RuntimeError as err:
        oom = "out of memory" in str(err).lower()
        is_cuda = str(device).startswith("cuda")
        if not (oom and is_cuda):
            raise

        logger.info(
            "CUDA OOM during region coefficient extraction; retrying on CPU in chunks."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        contrib_blocks = []
        task_model_cpu = task_model.to("cpu")
        with torch.no_grad():
            for x_batch in _chunks(X_seq, batch_size):
                contrib_blocks.append(_as_numpy(_forward_contrib(task_model_cpu, x_batch)))

    contrib_np = np.concatenate(contrib_blocks, axis=0)  # (B, P, C)
    if contrib_np.ndim != 3:
        return []

    n_classes = contrib_np.shape[2]
    if n_classes == 1:
        per_region = contrib_np[:, :, 0]
    else:
        idx = class_index if class_index is not None else 1
        idx = int(np.clip(idx, 0, n_classes - 1))
        per_region = contrib_np[:, :, idx]

    parcel_ids = None
    if hasattr(task_model.backbone, "_parcel_ids") and task_model.backbone._parcel_ids is not None:
        parcel_ids = [int(p) for p in task_model.backbone._parcel_ids]
    if parcel_ids is None:
        parcel_ids = list(range(per_region.shape[1]))

    keys = _region_keys(parcel_ids, metadata)
    out: list[dict[str, float]] = []
    for i in range(per_region.shape[0]):
        out.append({k: float(v) for k, v in zip(keys, per_region[i])})
    return out


def extract_region_coefficients(
    model: Any,
    X: Sequence[Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Extract region-level coefficients/importances for a fitted model.

    For linear/sklearn models this returns static model coefficients aggregated
    per region. For additive deep heads this returns the mean absolute parcel
    contribution across subjects if *X* is provided.
    """
    estimator = _resolve_estimator(model)

    static = _extract_static_region_coefficients(estimator, metadata=metadata)
    if static is not None:
        return static

    if X is not None:
        subject_maps = _extract_additive_head_subject_coefficients(estimator, X, metadata=metadata)
        if subject_maps:
            keys = list(subject_maps[0].keys())
            arr = np.vstack([[row[k] for k in keys] for row in subject_maps])
            return {k: float(np.mean(np.abs(arr[:, i]))) for i, k in enumerate(keys)}

    return {}


def extract_subject_region_coefficients(
    model: Any,
    X: Sequence[Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, float]], bool]:
    """Return per-subject region coefficients and a flag for static/shared values.

    Returns:
        (coefficients_per_subject, is_static)
    """
    estimator = _resolve_estimator(model)

    if X is not None:
        subj = _extract_additive_head_subject_coefficients(estimator, X, metadata=metadata)
        if subj:
            return subj, False

    static = _extract_static_region_coefficients(estimator, metadata=metadata)
    if static is None:
        static = {}

    return [dict(static)], True


def aggregate_subject_region_coefficients(
    region_coefficients: Sequence[Mapping[str, float]],
    *,
    mode: str = "mean_abs",
) -> dict[str, float]:
    """Aggregate subject-specific region maps into one static coefficient map.

    Args:
        region_coefficients: Sequence of per-subject ``{region: value}`` mappings.
        mode: Aggregation mode, one of ``"mean_abs"``, ``"mean"``, ``"median_abs"``.

    Returns:
        ``{region: aggregated_value}``.
    """
    if not region_coefficients:
        return {}

    keys: list[str] = []
    seen: set[str] = set()
    for row in region_coefficients:
        for key in row.keys():
            key_str = str(key)
            if key_str not in seen:
                seen.add(key_str)
                keys.append(key_str)

    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray(
            [float(row.get(key, 0.0)) for row in region_coefficients],
            dtype=np.float64,
        )
        if mode == "mean":
            agg_val = float(np.mean(values))
        elif mode == "median_abs":
            agg_val = float(np.median(np.abs(values)))
        else:
            agg_val = float(np.mean(np.abs(values)))
        out[key] = agg_val
    return out


def build_region_coefficient_records(
    *,
    subject_ids: Sequence[Any] | None,
    model_name: str,
    region_coefficients: Sequence[Mapping[str, float]] | Mapping[str, float],
    y_true: Sequence[Any] | None = None,
    y_pred: Sequence[Any] | None = None,
    fold: int | None = None,
    split: str | None = None,
    is_static: bool = False,
    metadata_fields: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build standardized records for region coefficients."""
    if isinstance(region_coefficients, Mapping):
        coeff_list = [dict(region_coefficients)]
        is_static = True
    else:
        coeff_list = [dict(r) for r in region_coefficients]

    if is_static:
        sid_list = ["__global__"]
    else:
        if subject_ids is None:
            raise ValueError("subject_ids must be provided for subject-specific coefficients.")
        sid_list = [str(s) for s in subject_ids]
        if len(sid_list) != len(coeff_list):
            raise ValueError(
                f"subject_ids length ({len(sid_list)}) does not match coefficient rows ({len(coeff_list)})."
            )

    records: list[dict[str, Any]] = []
    for idx, sid in enumerate(sid_list):
        rec: dict[str, Any] = {
            "subject_id": sid,
            "model_name": model_name,
            "region_coefficients": coeff_list[0] if is_static else coeff_list[idx],
            "is_static": bool(is_static),
        }
        if fold is not None:
            rec["fold"] = int(fold)
        if split is not None:
            rec["split"] = str(split)
        if y_true is not None and not is_static:
            rec["y_true"] = float(_as_numpy(y_true)[idx])
        if y_pred is not None and not is_static:
            rec["y_pred"] = float(_as_numpy(y_pred)[idx])
        if metadata_fields:
            rec.update(dict(metadata_fields))
        records.append(rec)
    return records


def records_to_wide_dataframe(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert coefficient records into one-row-per-subject wide dataframe."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        row: dict[str, Any] = {}
        for key, value in rec.items():
            if key == "region_coefficients":
                continue
            if isinstance(value, (Mapping, list, tuple, set)):
                continue
            row[key] = value
        coeffs = rec.get("region_coefficients", {}) or {}
        for key, val in coeffs.items():
            row[str(key)] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def save_region_coefficients(
    records: Sequence[Mapping[str, Any]],
    *,
    output_root: str | Path,
    model_name: str,
    run_id: str,
    fold: int | None = None,
    split: str | None = None,
) -> Path:
    """Save coefficient records into a cumulative parquet file.

    The file is always named ``coefficients.parquet`` inside the experiment
    coefficients directory. Model/run and other experiment metadata are stored
    as explicit columns in the parquet rows.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    df_new = records_to_wide_dataframe(records)
    parquet_path = output_root / "coefficients.parquet"

    if parquet_path.exists():
        df_prev = pd.read_parquet(parquet_path)
        df_all = pd.concat([df_prev, df_new], ignore_index=True)
    else:
        df_all = df_new

    key_cols = [
        c
        for c in ("run_id", "model_name", "subject_id", "fold", "split")
        if c in df_all.columns
    ]
    if key_cols:
        df_all = df_all.drop_duplicates(subset=key_cols, keep="last")
    else:
        df_all = df_all.drop_duplicates(keep="last")

    if "fold" in df_all.columns:
        df_all = df_all.sort_values(by=["fold"], kind="stable").reset_index(drop=True)

    preferred_prefix = [
        "run_id",
        "model_name",
        "dataset",
        "tissue_type",
        "primary_metric",
        "metric_to_compute",
        "subject_id",
        "is_static",
        "fold",
        "split",
        "y_true",
        "y_pred",
    ]
    ordered_prefix = [c for c in preferred_prefix if c in df_all.columns]
    remaining_cols = [c for c in df_all.columns if c not in ordered_prefix]
    df_all = df_all[ordered_prefix + remaining_cols]

    df_all.to_parquet(parquet_path, index=False)
    return parquet_path


def load_region_coefficients_table(
    *,
    experiment_dir: str | Path,
    model_name: str | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Load saved coefficients parquet and optionally filter by model/run."""
    exp_dir = Path(experiment_dir)
    parquet_path = exp_dir / "coefficients" / "coefficients.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Coefficient parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if run_id is not None and "run_id" in df.columns:
        df = df[df["run_id"].astype(str) == str(run_id)]
    if model_name is not None and "model_name" in df.columns:
        df = df[df["model_name"].astype(str) == str(model_name)]

    if df.empty:
        raise ValueError(
            "No coefficient rows found for the provided filters "
            f"(model_name={model_name}, run_id={run_id})."
        )

    return df


def coefficients_from_table(
    coefficients_df: pd.DataFrame,
    *,
    fold: int | None = None,
    average_folds: bool = False,
) -> dict[str, float]:
    """Extract a region->value mapping from saved coefficient table.

    - If ``fold`` is provided, returns coefficients for that fold.
    - If ``average_folds`` is true (or no fold is provided), returns mean across folds.
    """
    if coefficients_df.empty:
        raise ValueError("Empty coefficients dataframe.")

    df = coefficients_df.copy()
    if "is_static" in df.columns:
        df = df[df["is_static"].astype(bool)]
    if df.empty:
        raise ValueError("No static coefficient rows available for plotting.")

    if fold is not None and "fold" in df.columns:
        df = df[df["fold"] == int(fold)]
        if df.empty:
            raise ValueError(f"No coefficients found for fold={fold}.")
        row = df.iloc[-1]
    else:
        average_folds = True if fold is None else average_folds
        if not average_folds:
            row = df.iloc[-1]
        else:
            meta_cols = {
                "subject_id",
                "model_name",
                "run_id",
                "dataset",
                "tissue_type",
                "primary_metric",
                "metric_to_compute",
                "is_static",
                "fold",
                "split",
                "y_true",
                "y_pred",
            }
            region_cols = [
                c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not region_cols:
                raise ValueError("No region columns found in coefficients dataframe.")
            means = df[region_cols].mean(axis=0)
            return {str(col): float(means[col]) for col in region_cols}

    meta_cols = {
        "subject_id",
        "model_name",
        "run_id",
        "dataset",
        "tissue_type",
        "primary_metric",
        "metric_to_compute",
        "is_static",
        "fold",
        "split",
        "y_true",
        "y_pred",
    }
    region_cols = [
        c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    return {str(col): float(row[col]) for col in region_cols}


def plot_experiment_coefficients(
    *,
    experiment_dir: str | Path,
    model_name: str | None = None,
    atlas_img=None,
    atlas_surface: Mapping[str, Any] | None = None,
    fold: int | None = None,
    average_folds: bool = False,
    label_map: Mapping[Any, Any] | None = None,
    run_id: str | None = None,
    cmap: str = "coolwarm",
    threshold: float | None = None,
    output_file: str | Path | None = None,
):
    """Load experiment coefficients and produce a nilearn figure.

    Saves by default to:
        ``exp_outputs/plots/<run_id>/coefficients/`` when *experiment_dir*
        is an ``exp_outputs/experiments/exp_<run_id>`` path.
    """
    exp_dir = Path(experiment_dir)
    coeff_df = load_region_coefficients_table(
        experiment_dir=exp_dir,
        model_name=model_name,
        run_id=run_id,
    )
    resolved_model_name = model_name
    if resolved_model_name is None:
        if "model_name" not in coeff_df.columns:
            raise ValueError("Could not infer model_name from coefficients table.")
        model_values = sorted(
            {str(v) for v in coeff_df["model_name"].dropna().unique().tolist()}
        )
        if not model_values:
            raise ValueError("Could not infer model_name from coefficients table.")
        if len(model_values) > 1:
            raise ValueError(
                "Multiple models found in coefficients table; please specify model_name. "
                f"Found: {model_values}"
            )
        resolved_model_name = model_values[0]
    coeffs = coefficients_from_table(
        coeff_df,
        fold=fold,
        average_folds=average_folds,
    )

    if output_file is None:
        run_token = str(run_id) if run_id is not None else exp_dir.name.replace("exp_", "")
        if exp_dir.parent.name == "experiments" and exp_dir.name.startswith("exp_"):
            out_dir = exp_dir.parent.parent / "plots" / run_token / "coefficients"
        else:
            out_dir = exp_dir / "plots" / "coefficients"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"fold{fold}" if fold is not None else "mean_folds"
        safe_model = str(resolved_model_name).replace("/", "_")
        output_file = out_dir / f"coefficients_{safe_model}_{suffix}.png"

    title = (
        f"Region coefficients — {resolved_model_name} — fold {fold}"
        if fold is not None
        else f"Region coefficients — {resolved_model_name} — mean across folds"
    )
    if atlas_img is not None:
        disp = plot_subject_coefficients(
            subject_id="__global__",
            coefficients=coeffs,
            atlas_img=atlas_img,
            label_map=label_map,
            title=title,
            output_file=output_file,
            cmap=cmap,
            threshold=threshold,
        )
        return disp, Path(output_file)

    if atlas_surface is not None:
        fig = plot_surface_region_coefficients(
            subject_id="__global__",
            coefficients=coeffs,
            surface_atlas=atlas_surface,
            title=title,
            output_file=output_file,
            cmap=cmap,
            threshold=threshold,
        )
        return fig, Path(output_file)

    raise ValueError(
        "plot_experiment_coefficients requires either `atlas_img` (volumetric) "
        "or `atlas_surface` (surface atlas metadata)."
    )


def load_atlas_from_run(
    run_id: str,
    *,
    experiments_root: str | Path = "./exp_outputs/experiments",
    atlas_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load atlas metadata/resources for one experiment run.

    Resolution priority for volumetric atlas:
      1) explicit ``atlas_path`` override
      2) ``analysis.coefficients_plot.atlas_*`` from experiment config
      3) ``metadata.yaml`` atlas section (if present)

    If no volumetric atlas is available, returns surface Schaefer atlas
    resources inferred from run config.
    """
    exp_dir = Path(experiments_root) / f"exp_{run_id}"
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    def _is_missing(value: Any) -> bool:
        return value in (None, "", "null", "None")

    cfg_path = exp_dir / "config.yaml"
    meta_path = exp_dir / "metadata.yaml"
    cfg = OmegaConf.load(cfg_path) if cfg_path.exists() else None
    meta = OmegaConf.load(meta_path) if meta_path.exists() else None

    # ---- Volumetric atlas path priority ----
    atlas_candidate = None if _is_missing(atlas_path) else str(atlas_path)
    if _is_missing(atlas_candidate) and cfg is not None:
        atlas_candidate = cfg.get("analysis", {}).get("coefficients_plot", {}).get(
            "atlas_path", None
        )
        if _is_missing(atlas_candidate):
            atlas_candidate = cfg.get("analysis", {}).get("coefficients_plot", {}).get(
                "atlas_img", None
            )
    if _is_missing(atlas_candidate) and meta is not None:
        atlas_candidate = meta.get("atlas", {}).get("atlas_path", None)
        if _is_missing(atlas_candidate):
            atlas_candidate = meta.get("atlas", {}).get("atlas_img", None)

    if not _is_missing(atlas_candidate):
        atlas_path_obj = Path(str(atlas_candidate))
        if atlas_path_obj.exists():
            return {
                "atlas_type": "volume",
                "atlas_path": str(atlas_path_obj),
                "experiment_dir": str(exp_dir),
            }

    # ---- Surface Schaefer fallback (automatic) ----
    if cfg is None:
        raise ValueError(
            "Could not load experiment config for atlas inference and no explicit atlas_path was provided."
        )

    scale = int(cfg.get("dataset", {}).get("scale", 100))
    surface_space = str(cfg.get("dataset", {}).get("surface_space", "fslr_32k"))
    cached = _cached_surface_atlas(scale=scale, surface_space=surface_space)

    label_map_path = cfg.get("analysis", {}).get("coefficients_plot", {}).get(
        "label_map_path", None
    )
    if _is_missing(label_map_path):
        default_label_map = Path("aux_materials/fs_labels.json")
        label_map_path = str(default_label_map) if default_label_map.exists() else None

    return {
        "atlas_type": "surface_schaefer",
        "experiment_dir": str(exp_dir),
        "surface_space": surface_space,
        "scale": scale,
        "left_mesh": cached["left_mesh"],
        "right_mesh": cached["right_mesh"],
        "parcel_labels": cached["parcel_labels"],
        "n_left_vertices": cached["n_left_vertices"],
        "n_right_vertices": cached["n_right_vertices"],
        "n_regions": cached["n_regions"],
        "label_map_path": label_map_path,
        "atlas_meta": cached["atlas_meta"],
    }


def _coef_label_set(coefficients: Mapping[Any, float]) -> set[int]:
    out: set[int] = set()
    for key in coefficients.keys():
        try:
            out.add(int(str(key).split(":")[-1]))
        except Exception:
            continue
    return out


def plot_surface_region_coefficients(
    subject_id: str,
    coefficients: Mapping[Any, float],
    surface_atlas: Mapping[str, Any],
    *,
    title: str | None = None,
    output_file: str | Path | None = None,
    cmap: str = "coolwarm",
    threshold: float | None = None,
):
    """Plot region coefficients on surface meshes using nilearn."""
    from matplotlib import pyplot as plt
    from nilearn import plotting

    left_mesh = surface_atlas["left_mesh"]
    right_mesh = surface_atlas["right_mesh"]
    parcel_labels = np.asarray(surface_atlas["parcel_labels"]).astype(np.int32)
    n_left = int(surface_atlas["n_left_vertices"])

    coeff_map = _to_label_value_map(coefficients, label_map=None)
    texture = np.zeros(parcel_labels.shape[0], dtype=np.float32)
    for label_id, value in coeff_map.items():
        texture[parcel_labels == int(label_id)] = float(value)

    atlas_labels = set(np.unique(parcel_labels[parcel_labels > 0]).astype(int).tolist())
    coef_labels = _coef_label_set(coefficients)
    if coef_labels and coef_labels != atlas_labels:
        missing_in_coef = sorted(list(atlas_labels - coef_labels))
        missing_in_atlas = sorted(list(coef_labels - atlas_labels))
        logger.warning(
            "Region alignment mismatch for coefficient plotting: "
            "atlas_labels=%d, coef_labels=%d, missing_in_coef=%d, missing_in_atlas=%d",
            len(atlas_labels),
            len(coef_labels),
            len(missing_in_coef),
            len(missing_in_atlas),
        )

    tex_left = texture[:n_left]
    tex_right = texture[n_left:]

    vmax = float(np.max(np.abs(texture))) if texture.size else 1.0
    if vmax == 0:
        vmax = 1.0

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    plotting.plot_surf_stat_map(
        left_mesh,
        tex_left,
        hemi="left",
        view="lateral",
        cmap=cmap,
        threshold=threshold,
        symmetric_cbar=True,
        colorbar=False,
        vmin=-vmax,
        vmax=vmax,
        axes=ax1,
        title="Left",
    )
    plotting.plot_surf_stat_map(
        right_mesh,
        tex_right,
        hemi="right",
        view="lateral",
        cmap=cmap,
        threshold=threshold,
        symmetric_cbar=True,
        colorbar=True,
        vmin=-vmax,
        vmax=vmax,
        axes=ax2,
        title="Right",
    )
    fig.suptitle(title or f"Region coefficients — {subject_id}")

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=180, bbox_inches="tight")

    return fig


def filter_region_coefficient_records(
    records: Sequence[Mapping[str, Any]],
    *,
    subject_id: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Filter region coefficient records by subject and/or model."""
    out: list[dict[str, Any]] = []
    for rec in records:
        if subject_id is not None and str(rec.get("subject_id")) != str(subject_id):
            continue
        if model_name is not None and str(rec.get("model_name")) != str(model_name):
            continue
        out.append(dict(rec))
    return out


def average_region_coefficients(
    records: Sequence[Mapping[str, Any]],
    *,
    group_by: Sequence[str] = ("model_name", "subject_id"),
) -> list[dict[str, Any]]:
    """Average region coefficients across folds/runs for selected grouping."""
    if not records:
        return []

    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for rec in records:
        key = tuple(rec.get(k) for k in group_by)
        groups.setdefault(key, []).append(rec)

    averaged: list[dict[str, Any]] = []
    for key, recs in groups.items():
        all_region_keys = sorted(
            {rk for rec in recs for rk in (rec.get("region_coefficients", {}) or {}).keys()}
        )
        agg = {}
        for rk in all_region_keys:
            vals = [float(rec.get("region_coefficients", {}).get(rk, 0.0)) for rec in recs]
            agg[rk] = float(np.mean(vals))

        out = {k: v for k, v in zip(group_by, key)}
        out["model_name"] = str(out.get("model_name", "unknown"))
        out["subject_id"] = str(out.get("subject_id", "__group__"))
        out["region_coefficients"] = agg
        out["is_static"] = bool(all(bool(rec.get("is_static", False)) for rec in recs))
        out["n_records_averaged"] = len(recs)
        averaged.append(out)

    return averaged


def _to_label_value_map(
    coefficients: Mapping[Any, float],
    label_map: Mapping[Any, Any] | None = None,
) -> dict[int, float]:
    label_map = label_map or {}
    out: dict[int, float] = {}
    for key, value in coefficients.items():
        if key in label_map:
            label_id = int(label_map[key])
            out[label_id] = float(value)
            continue
        try:
            label_id = int(str(key).split(":")[-1])
            out[label_id] = float(value)
        except Exception:
            continue
    return out


def plot_subject_coefficients(
    subject_id: str,
    coefficients: Mapping[Any, float],
    atlas_img,
    *,
    label_map: Mapping[Any, Any] | None = None,
    title: str | None = None,
    output_file: str | Path | None = None,
    cmap: str = "coolwarm",
    threshold: float | None = None,
):
    """Plot region coefficients on a volumetric atlas using nilearn.

    Args:
        subject_id: Subject identifier used for title.
        coefficients: Region -> coefficient mapping.
        atlas_img: Path-like or niimg with integer atlas labels.
        label_map: Optional mapping from region key/name to atlas integer label.
        title: Optional figure title.
        output_file: Optional path where the figure is saved.
        cmap: Diverging colormap.
        threshold: Optional stat-map threshold.
    """
    from nilearn import image, plotting

    atlas_niimg = image.load_img(atlas_img)
    atlas_data = np.asarray(atlas_niimg.get_fdata())
    if atlas_data.ndim != 3:
        raise ValueError(f"Expected 3D atlas image, got shape {atlas_data.shape}.")

    label_values = _to_label_value_map(coefficients, label_map=label_map)
    atlas_labels = set(np.unique(atlas_data[atlas_data > 0]).astype(int).tolist())
    coef_labels = set(label_values.keys())
    if coef_labels and coef_labels != atlas_labels:
        missing_in_coef = sorted(list(atlas_labels - coef_labels))
        missing_in_atlas = sorted(list(coef_labels - atlas_labels))
        logger.warning(
            "Volumetric atlas/coefficient label mismatch: atlas_labels=%d, coef_labels=%d, "
            "missing_in_coef=%d, missing_in_atlas=%d",
            len(atlas_labels),
            len(coef_labels),
            len(missing_in_coef),
            len(missing_in_atlas),
        )

    stat_data = np.zeros_like(atlas_data, dtype=np.float32)
    for label, val in label_values.items():
        stat_data[atlas_data == label] = float(val)

    stat_img = image.new_img_like(atlas_niimg, stat_data)
    disp = plotting.plot_stat_map(
        stat_img,
        bg_img=atlas_niimg,
        title=title or f"Region coefficients — {subject_id}",
        cmap=cmap,
        threshold=threshold,
        symmetric_cbar=True,
    )

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        disp.savefig(str(output_file))

    return disp
