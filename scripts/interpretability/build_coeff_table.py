from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "exp_outputs" / "experiments"
SUMMARY_PATH = PROJECT_ROOT / "exp_outputs" / "summary" / "coefficients_long.parquet"

COEF_META_COLUMNS = {
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
	"coefficient_mode",
}


def _safe_get(cfg: Any, path: str, default: Any = None) -> Any:
	current = cfg
	for part in path.split("."):
		if current is None:
			return default
		try:
			if part in current:
				current = current[part]
			else:
				return default
		except Exception:
			return default
	return current if current is not None else default


def _target_measure(cfg: Any) -> str | None:
	target = _safe_get(cfg, "target.target_column")
	if isinstance(target, Sequence) and not isinstance(target, (str, bytes)):
		return ",".join(str(v) for v in target)
	if target is None:
		return None
	return str(target)


def _as_scalar(value: Any) -> Any:
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	return str(value)


def _build_exp_id(df: pd.DataFrame) -> pd.Series:
	cols = ["run_id"]
	if "fold" in df.columns:
		cols.append("fold")
	if "seed" in df.columns:
		cols.append("seed")
	return df[cols].astype(str).agg("_".join, axis=1)


def _score_table(metrics_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
	metric_rows = metrics_df[metrics_df["metric"].astype(str) == str(metric_name)].copy()
	if metric_rows.empty:
		for fallback in ("accuracy", "r2", "roc_auc", "f1", "mse"):
			candidate = metrics_df[metrics_df["metric"].astype(str) == fallback]
			if not candidate.empty:
				metric_rows = candidate.copy()
				break
	if metric_rows.empty:
		return pd.DataFrame(columns=["fold", "train_score", "test_score"])

	pivot = (
		metric_rows.pivot_table(
			index="fold",
			columns="split",
			values="value",
			aggfunc="mean",
		)
		.reset_index()
		.rename_axis(None, axis=1)
	)
	pivot = pivot.rename(columns={"train": "train_score", "test": "test_score"})
	if "train_score" not in pivot.columns:
		pivot["train_score"] = pd.NA
	if "test_score" not in pivot.columns:
		pivot["test_score"] = pd.NA
	return pivot[["fold", "train_score", "test_score"]]


def build_coeff_table(
	experiments_root: Path = EXPERIMENTS_ROOT,
	output_path: Path = SUMMARY_PATH,
) -> pd.DataFrame:
	rows: list[dict[str, Any]] = []

	for exp_dir in sorted(experiments_root.glob("exp_*")):
		coef_file = exp_dir / "coefficients" / "coefficients.parquet"
		metrics_file = exp_dir / "metrics" / "fold_metrics.parquet"
		config_file = exp_dir / "config.yaml"

		if not coef_file.exists() or not config_file.exists():
			continue

		df_coef = pd.read_parquet(coef_file)
		if df_coef.empty:
			continue

		cfg = OmegaConf.load(config_file)
		run_id = exp_dir.name.replace("exp_", "")

		metric_name = (
			_safe_get(cfg, "dataset.metric_to_compute")
			or (str(df_coef["primary_metric"].iloc[0]) if "primary_metric" in df_coef.columns else None)
		)

		if metrics_file.exists() and metric_name is not None:
			score_df = _score_table(pd.read_parquet(metrics_file), str(metric_name))
		else:
			score_df = pd.DataFrame(columns=["fold", "train_score", "test_score"])

		region_cols = [
			col for col in df_coef.columns if col not in COEF_META_COLUMNS and pd.api.types.is_numeric_dtype(df_coef[col])
		]
		if not region_cols:
			continue

		df_long = df_coef.melt(
			id_vars=[c for c in ["fold"] if c in df_coef.columns],
			value_vars=region_cols,
			var_name="region",
			value_name="coef",
		)

		if "fold" not in df_long.columns:
			df_long["fold"] = 0

		merged = df_long.merge(score_df, on="fold", how="left")

		for rec in merged.itertuples(index=False):
			rows.append(
				{
					"run_id": run_id,
					"fold": int(rec.fold),
					"seed": _as_scalar(_safe_get(cfg, "runtime.seed", _safe_get(cfg, "random_state"))),
					"dataset": _as_scalar(_safe_get(cfg, "dataset.name", None)),
					"task": _as_scalar(
						_safe_get(cfg, "target.prediction_task", _safe_get(cfg, "pred_head.prediction_task"))
					),
					"model": _as_scalar(_safe_get(cfg, "model.name", None)),
					"embedding": _as_scalar(_safe_get(cfg, "model.backbone", None)),
					"microstructure": _as_scalar(_safe_get(cfg, "dataset.metric_to_compute", None)),
					"target_measure": _target_measure(cfg),
					"train_score": rec.train_score,
					"test_score": rec.test_score,
					"region": str(rec.region),
					"coef": float(rec.coef),
				}
			)

	df_out = pd.DataFrame(rows)
	if df_out.empty:
		df_out["exp_id"] = pd.Series(dtype="string")
	else:
		df_out["exp_id"] = _build_exp_id(df_out)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df_out.to_parquet(output_path, index=False)
	return df_out


if __name__ == "__main__":
	df = build_coeff_table()
	print(f"Saved {len(df)} rows to {SUMMARY_PATH}")