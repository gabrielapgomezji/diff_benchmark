"""
stability_decomposition.py
--------------------------
Analyze the relationship between prediction performance and region-importance
stability across datasets.

Per dataset-task-pipeline group:
1) Mean prediction score across runs (seeded runs).
2) Global stability = average over regions of:
      abs(mean(coef)) / (std(coef) + 1e-6)
   where coef samples come from all available coefficient rows (folds/runs)
   for that group.

Output:
- Scatter plot: score vs stability
- Global and per-dataset regression lines
- Table with one row per point used in plot
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

DATASET_COLORS = {
	"hcp": "#4C78A8",
	"camcan": "#F58518",
	"abide": "#54A24B",
}

MODEL_MARKERS = {
	"region_group_lasso": "o",
	"region_elasticnet": "s",
	"region_pca": "^",
	"pointnet": "D",
}


def apply_miccai_style() -> None:
	sns.set_theme(style="whitegrid", context="paper")
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
			"xtick.labelsize": 8,
			"ytick.labelsize": 8,
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


def _normalize_comp(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	out["target_clean"] = out["target"].map(_clean_target)
	out["embedding"] = out["config.model.backbone.region_representation"].fillna("none")
	out["microstructure"] = out["primary_metric"].astype(str)
	out["model"] = out["model_name"].astype(str)
	out["pipeline"] = out["microstructure"] + "|" + out["embedding"] + "|" + out["model"]
	out["group_id"] = (
		out["dataset"].astype(str)
		+ "|"
		+ out["target_clean"].astype(str)
		+ "|"
		+ out["prediction_task"].astype(str)
		+ "|"
		+ out["pipeline"].astype(str)
	)
	return out


def _region_columns(df_coef: pd.DataFrame) -> list[str]:
	return [
		c
		for c in df_coef.columns
		if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(df_coef[c])
	]


def _load_all_coefficients(coeff_root: Path) -> pd.DataFrame:
	parts: list[pd.DataFrame] = []
	for p in coeff_root.glob("**/coefficients/coefficients.parquet"):
		try:
			df = pd.read_parquet(p)
		except Exception as exc:  # pragma: no cover
			warnings.warn(f"Skipping unreadable coefficient file {p}: {exc}")
			continue
		if "run_id" not in df.columns:
			continue
		if "is_static" in df.columns:
			df = df[df["is_static"].astype(bool)].copy()
			if df.empty:
				continue
		parts.append(df)

	if not parts:
		return pd.DataFrame()
	return pd.concat(parts, ignore_index=True)


def _compute_group_global_stability(coef_group: pd.DataFrame, region_cols: list[str]) -> tuple[float, int]:
	if coef_group.empty:
		return float("nan"), 0

	X = coef_group[region_cols].to_numpy(dtype=float)
	# Different experiment files can carry partially disjoint region columns.
	# Drop columns that are entirely NaN for this group before aggregation.
	valid_cols = ~np.all(np.isnan(X), axis=0)
	if not np.any(valid_cols):
		return float("nan"), 0
	X = X[:, valid_cols]

	mean_region = np.nanmean(X, axis=0)
	std_region = np.nanstd(X, axis=0)
	stability_region = np.abs(mean_region) / (std_region + 1e-6)
	global_stability = float(np.nanmean(stability_region))
	n_regions = int(mean_region.shape[0])
	return global_stability, n_regions


def _build_decomposition_table(comp: pd.DataFrame, coef_all: pd.DataFrame) -> pd.DataFrame:
	if comp.empty or coef_all.empty:
		return pd.DataFrame()

	coef_all = coef_all.copy()
	coef_all["run_id"] = coef_all["run_id"].astype(str)
	region_cols = _region_columns(coef_all)
	if not region_cols:
		return pd.DataFrame()

	rows: list[dict[str, object]] = []
	for (dataset, target_clean, task, pipeline), grp in comp.groupby(
		["dataset", "target_clean", "prediction_task", "pipeline"], dropna=False
	):
		g = grp.copy()
		try:
			score_col = _choose_score_column(g, str(task))
		except Exception:
			continue

		scores = g[score_col].dropna().to_numpy(dtype=float)
		if scores.size == 0:
			continue
		mean_score = float(np.mean(scores))

		run_ids = set(g["run_id"].astype(str).unique())
		coef_group = coef_all[coef_all["run_id"].isin(run_ids)].copy()
		if coef_group.empty:
			continue

		global_stability, n_regions = _compute_group_global_stability(coef_group, region_cols)
		if np.isnan(global_stability):
			continue

		model_name = str(g["model"].iloc[0])
		rows.append(
			{
				"dataset": str(dataset),
				"target_clean": str(target_clean),
				"prediction_task": str(task),
				"pipeline": str(pipeline),
				"model": model_name,
				"mean_prediction_score": mean_score,
				"global_stability": global_stability,
				"n_runs": int(g.shape[0]),
				"n_regions": int(n_regions),
			}
		)

	return pd.DataFrame(rows)


def _dataset_label(dataset: str) -> str:
	return {"hcp": "HCP", "camcan": "CamCAN", "abide": "ABIDE II"}.get(dataset, dataset)


def _plot_scatter(df: pd.DataFrame, out_file: Path) -> None:
	if df.empty:
		return

	d = df.copy()
	d["dataset_display"] = d["dataset"].map(_dataset_label)

	fig, ax = plt.subplots(figsize=(7.6, 4.2))

	# Scale optional size by number of regions.
	nr = d["n_regions"].to_numpy(dtype=float)
	if np.isclose(nr.min(), nr.max()):
		d["marker_size"] = 70.0
	else:
		d["marker_size"] = 40.0 + 100.0 * (nr - nr.min()) / (nr.max() - nr.min())

	# Draw points per model to enforce marker shape and per dataset for colors.
	for model_name, mdf in d.groupby("model", dropna=False):
		marker = MODEL_MARKERS.get(str(model_name), "o")
		for ds, sdf in mdf.groupby("dataset", dropna=False):
			ax.scatter(
				sdf["mean_prediction_score"],
				sdf["global_stability"],
				s=sdf["marker_size"],
				c=DATASET_COLORS.get(str(ds), "#888888"),
				marker=marker,
				alpha=0.82,
				edgecolors="white",
				linewidths=0.5,
				zorder=3,
			)

	# Global regression line.
	if d.shape[0] >= 2:
		sns.regplot(
			data=d,
			x="mean_prediction_score",
			y="global_stability",
			scatter=False,
			ci=None,
			line_kws={"color": "#222222", "linewidth": 1.5, "alpha": 0.9},
			ax=ax,
		)

	# Per-dataset regression lines.
	for ds, sdf in d.groupby("dataset", dropna=False):
		if sdf.shape[0] < 2:
			continue
		sns.regplot(
			data=sdf,
			x="mean_prediction_score",
			y="global_stability",
			scatter=False,
			ci=None,
			line_kws={
				"color": DATASET_COLORS.get(str(ds), "#777777"),
				"linewidth": 1.2,
				"alpha": 0.75,
				"linestyle": "--",
			},
			ax=ax,
		)

	# Custom legends.
	from matplotlib.lines import Line2D

	dataset_handles = [
		Line2D([0], [0], marker="o", linestyle="", color=DATASET_COLORS[k], markersize=6, label=_dataset_label(k))
		for k in ["hcp", "camcan", "abide"]
		if (d["dataset"] == k).any()
	]
	model_handles = [
		Line2D([0], [0], marker=MODEL_MARKERS.get(m, "o"), linestyle="", color="#555555", markersize=6, label=m)
		for m in sorted(d["model"].astype(str).unique())
	]

	leg1 = ax.legend(handles=dataset_handles, title="Dataset", loc="upper left", frameon=True)
	ax.add_artist(leg1)
	ax.legend(handles=model_handles, title="Model", loc="upper right", frameon=True)

	ax.set_title("Prediction Performance vs Region-Importance Stability")
	ax.set_xlabel("Mean prediction score")
	ax.set_ylabel("Global stability")
	ax.grid(axis="y", linestyle="--", alpha=0.3)
	ax.grid(axis="x", linestyle="--", alpha=0.25)
	sns.despine(ax=ax, top=True, right=True)

	fig.savefig(out_file)
	plt.close(fig)


def run_stability_decomposition(
	comprehensive_path: str,
	coeff_root: str = "exp_outputs/experiments",
	out_dir: str = "exp_outputs/summary/plots/region_viz",
) -> tuple[Path, Path]:
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

	comp = _normalize_comp(comp)
	coef_all = _load_all_coefficients(Path(coeff_root))
	if coef_all.empty:
		raise RuntimeError("No coefficient tables found under coeff_root")

	decomp = _build_decomposition_table(comp, coef_all)
	if decomp.empty:
		raise RuntimeError("No rows available for stability decomposition")

	out_path = Path(out_dir)
	out_path.mkdir(parents=True, exist_ok=True)
	table_file = out_path / "performance_vs_stability.parquet"
	plot_file = out_path / "performance_vs_stability_scatter.pdf"

	decomp.sort_values(["dataset", "prediction_task", "pipeline"], inplace=True)
	decomp.to_parquet(table_file, index=False)
	_plot_scatter(decomp, plot_file)

	return table_file, plot_file


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Performance vs stability decomposition across datasets"
	)
	parser.add_argument(
		"--comprehensive",
		default="exp_outputs/summary/comprehensive_results.parquet",
		help="Path to comprehensive parquet",
	)
	parser.add_argument(
		"--coeff-root",
		default="exp_outputs/experiments",
		help="Root directory containing coefficient parquets",
	)
	parser.add_argument(
		"--outdir",
		default="exp_outputs/summary/plots/region_viz",
		help="Output directory",
	)
	args = parser.parse_args()

	table_file, plot_file = run_stability_decomposition(
		comprehensive_path=args.comprehensive,
		coeff_root=args.coeff_root,
		out_dir=args.outdir,
	)
	print("Saved decomposition table to", table_file)
	print("Saved decomposition plot to", plot_file)


if __name__ == "__main__":
	main()
