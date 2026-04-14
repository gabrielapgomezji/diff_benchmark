"""
region_rank.py
--------------
Compute region-ranking consistency within each dataset-task combination.

Workflow per (dataset, target, prediction_task):
1) Identify pipelines from comprehensive results.
2) Load coefficient outputs for matching run_id values.
3) Compute mean coefficient per region per pipeline (across folds).
4) Rank regions by absolute coefficient for each pipeline.
5) Compute pipeline-by-pipeline Spearman rank correlation.
6) Cluster rows/columns and save a heatmap.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMBOS = [
	("hcp", "Sex", "binary_classification", "HCP (sex)"),
	("camcan", "Sex", "binary_classification", "CamCAN (sex)"),
	("camcan", "Age", "regression", "CamCAN (age)"),
	("abide", "DX_GROUP", "binary_classification", "ABIDE II (autism)"),
]

META_COLUMNS = {
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


def apply_miccai_style() -> None:
	sns.set_theme(style="white", context="paper")
	plt.rcParams.update(
		{
			"figure.dpi": 120,
			"savefig.dpi": 300,
			"savefig.bbox": "tight",
			"savefig.pad_inches": 0.03,
			"font.family": "serif",
			"font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
			"font.size": 9,
			"axes.titlesize": 10,
			"axes.labelsize": 9,
			"xtick.labelsize": 7,
			"ytick.labelsize": 7,
			"pdf.fonttype": 42,
			"ps.fonttype": 42,
		}
	)


def _clean_target(value: object) -> str:
	if value is None or (isinstance(value, float) and np.isnan(value)):
		return ""
	s = str(value).strip()
	s = s.replace("[", "").replace("]", "").replace("'", "")
	s = s.replace("Gender", "Sex")
	s = s.replace("Age_in_Yrs", "Age")
	s = s.replace("dx_group", "DX_GROUP")
	return s.strip()


def _normalize_comprehensive_columns(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	out["target_clean"] = out["target"].map(_clean_target)
	out["microstructure"] = out["primary_metric"].astype(str)
	out["embedding"] = out["config.model.backbone.region_representation"].fillna("none")
	out["model"] = out["model_name"].astype(str)
	if "config.random_state" in out.columns:
		out["seed"] = out["config.random_state"].astype(str)
	else:
		out["seed"] = out["run_id"].astype(str)
	out["pipeline_id"] = out.apply(
		lambda r: f"{r['microstructure']} | {r['embedding']} | {r['model']} | seed={r['seed']}",
		axis=1,
	)
	return out


def _choose_score_column(df_task: pd.DataFrame, prediction_task: str) -> str:
	if prediction_task == "binary_classification":
		if "accuracy_weighted_test_mean" in df_task.columns:
			return "accuracy_weighted_test_mean"
		if "accuracy_test_mean" in df_task.columns:
			return "accuracy_test_mean"
		raise ValueError("No classification score column found")

	if "r2_test_mean" in df_task.columns and df_task["r2_test_mean"].notna().any():
		return "r2_test_mean"

	if "mae_test_mean" in df_task.columns and df_task["mae_test_mean"].notna().any():
		warnings.warn("Using negative MAE because r2_test_mean is unavailable")
		df_task["_neg_mae"] = -df_task["mae_test_mean"]
		return "_neg_mae"

	raise ValueError("No regression score column found")


def _region_columns(df_coef: pd.DataFrame) -> list[str]:
	cols: list[str] = []
	for c in df_coef.columns:
		if c in META_COLUMNS:
			continue
		if pd.api.types.is_numeric_dtype(df_coef[c]):
			cols.append(c)
	return cols


def _load_coefficients_for_runs(coeff_root: Path, run_ids: set[str]) -> pd.DataFrame:
	parts: list[pd.DataFrame] = []
	for p in coeff_root.glob("**/coefficients/coefficients.parquet"):
		try:
			df = pd.read_parquet(p)
		except Exception as exc:  # pragma: no cover
			warnings.warn(f"Skipping unreadable coefficient file {p}: {exc}")
			continue

		if "run_id" not in df.columns:
			continue
		df = df[df["run_id"].astype(str).isin(run_ids)].copy()
		if df.empty:
			continue
		if "is_static" in df.columns:
			df = df[df["is_static"].astype(bool)].copy()
			if df.empty:
				continue
		parts.append(df)

	if not parts:
		return pd.DataFrame()
	return pd.concat(parts, ignore_index=True)


def _compute_pipeline_region_means(
	comp_df: pd.DataFrame,
	coef_df: pd.DataFrame,
) -> pd.DataFrame:
	if comp_df.empty or coef_df.empty:
		return pd.DataFrame()

	region_cols = _region_columns(coef_df)
	if not region_cols:
		return pd.DataFrame()

	coef_agg = coef_df.groupby("run_id", dropna=False)[region_cols].mean(numeric_only=True)
	coef_agg.index = coef_agg.index.astype(str)

	meta = comp_df[["run_id", "pipeline_id"]].copy()
	meta["run_id"] = meta["run_id"].astype(str)
	meta = meta.drop_duplicates(subset=["run_id", "pipeline_id"])

	merged = meta.merge(
		coef_agg.reset_index(),
		on="run_id",
		how="inner",
	)
	if merged.empty:
		return pd.DataFrame()

	# If several run_id map to the same textual pipeline_id, average them.
	pipeline_region = merged.groupby("pipeline_id", dropna=False)[region_cols].mean(numeric_only=True)
	return pipeline_region


def _spearman_pipeline_corr(pipeline_region: pd.DataFrame) -> pd.DataFrame:
	if pipeline_region.empty or pipeline_region.shape[0] < 2:
		return pd.DataFrame()

	abs_vals = pipeline_region.abs()
	rank_vals = abs_vals.rank(axis=1, method="average", ascending=False)
	# Correlation among pipeline rows computed across region columns.
	corr = rank_vals.T.corr(method="spearman")
	return corr


def _plot_cluster_heatmap(corr: pd.DataFrame, title: str, out_file: Path) -> None:
	if corr.empty:
		return

	n = corr.shape[0]
	fig_side = max(4.8, min(14.0, 2.2 + 0.45 * n))

	cg = sns.clustermap(
		corr,
		method="average",
		metric="euclidean",
		cmap="RdBu_r",
		vmin=-1.0,
		vmax=1.0,
		center=0.0,
		linewidths=0.20,
		linecolor="white",
		figsize=(fig_side, fig_side),
		cbar_kws={"label": "Spearman rank correlation"},
	)

	cg.ax_heatmap.set_xlabel("Pipelines")
	cg.ax_heatmap.set_ylabel("Pipelines")
	cg.ax_heatmap.tick_params(axis="x", labelrotation=55)
	for lbl in cg.ax_heatmap.get_xticklabels():
		lbl.set_horizontalalignment("right")

	cg.fig.suptitle(title, y=1.02)
	cg.fig.savefig(out_file)
	plt.close(cg.fig)


def plot_region_rank_correlations(
	comprehensive_path: str,
	coefficients_root: str = "exp_outputs/experiments",
	out_dir: str = "exp_outputs/summary/plots/region_viz",
	top_k: int | None = None,
) -> list[Path]:
	apply_miccai_style()

	comp = pd.read_parquet(comprehensive_path)
	required = {
		"run_id",
		"dataset",
		"target",
		"prediction_task",
		"primary_metric",
		"model_name",
		"config.model.backbone.region_representation",
	}
	missing = sorted(required - set(comp.columns))
	if missing:
		raise ValueError(f"Missing required columns in comprehensive table: {missing}")

	comp = _normalize_comprehensive_columns(comp)

	out_path = Path(out_dir)
	out_path.mkdir(parents=True, exist_ok=True)

	all_run_ids = set(comp["run_id"].astype(str).dropna().unique())
	coef = _load_coefficients_for_runs(Path(coefficients_root), all_run_ids)
	if coef.empty:
		raise RuntimeError("No matching coefficient rows found for comprehensive run_id values")
	coef["run_id"] = coef["run_id"].astype(str)

	out_files: list[Path] = []
	for dataset, target_clean, task, title in COMBOS:
		sub = comp[
			(comp["dataset"] == dataset)
			& (comp["target_clean"] == target_clean)
			& (comp["prediction_task"] == task)
		].copy()

		if sub.empty:
			warnings.warn(f"Skipping combo with no rows in comprehensive table: {title}")
			continue

		# Rank by task-appropriate score if top_k is requested.
		if top_k is not None and top_k > 0:
			score_col = _choose_score_column(sub, task)
			sub = sub[sub[score_col].notna()].copy()
			sub = sub.sort_values(score_col, ascending=False).head(top_k)

		run_ids_sub = set(sub["run_id"].astype(str).unique())
		coef_sub = coef[coef["run_id"].isin(run_ids_sub)].copy()

		pipeline_region = _compute_pipeline_region_means(sub, coef_sub)
		corr = _spearman_pipeline_corr(pipeline_region)
		if corr.empty:
			warnings.warn(f"Skipping combo with insufficient pipelines/coefficients: {title}")
			continue

		tag = f"{dataset}_{target_clean.lower()}_{task}".replace(" ", "_")
		if top_k is not None and top_k > 0:
			tag += f"_top{top_k}"
		out_file = out_path / f"region_rank_corr_{tag}.pdf"
		_plot_cluster_heatmap(
			corr,
			title=f"Region ranking correlation - {title}",
			out_file=out_file,
		)
		out_files.append(out_file)

	return out_files


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Region ranking Spearman correlation heatmaps per dataset-task"
	)
	parser.add_argument(
		"--comprehensive",
		default="exp_outputs/summary/comprehensive_results.parquet",
		help="Path to comprehensive parquet table",
	)
	parser.add_argument(
		"--coeff-root",
		default="exp_outputs/experiments",
		help="Root directory containing experiment coefficient outputs",
	)
	parser.add_argument(
		"--outdir",
		default="exp_outputs/summary/plots/region_viz",
		help="Output directory",
	)
	parser.add_argument(
		"--top-k",
		type=int,
		default=None,
		help="Optional: keep only top-K pipelines per dataset-task by score",
	)
	args = parser.parse_args()

	out_files = plot_region_rank_correlations(
		comprehensive_path=args.comprehensive,
		coefficients_root=args.coeff_root,
		out_dir=args.outdir,
		top_k=args.top_k,
	)

	if out_files:
		print("Saved rank-correlation heatmaps:")
		for p in out_files:
			print(" -", p)
	else:
		print("No heatmaps generated")


if __name__ == "__main__":
	main()
