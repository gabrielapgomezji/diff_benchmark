"""
region_viz.py
-------------
Create one performance heatmap per dataset-task combination from a results parquet.

For each (dataset, target, prediction_task):
  1) Filter rows.
  2) Group by (microstructure, embedding, model).
  3) Compute mean and std across seeds.
  4) Plot heatmap cells with annotation: mean +- std.

All heatmaps share a global color scale to support direct comparison.
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


# Requested panels
COMBOS = [
	("hcp", "Sex", "binary_classification", "HCP (sex)"),
	("camcan", "Sex", "binary_classification", "CamCAN (sex)"),
	("camcan", "Age", "regression", "CamCAN (age)"),
	("abide", "DX_GROUP", "binary_classification", "ABIDE II (autism)"),
]


def apply_miccai_style() -> None:
	"""Use the same publication-oriented style direction as other visualization scripts."""
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
			"axes.linewidth": 0.8,
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
	"""Select the task-appropriate score column used for heatmap values."""
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


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Create canonical fields for grouping and plotting."""
	out = df.copy()
	out["target_clean"] = out["target"].map(_clean_target)
	out["microstructure"] = out["primary_metric"].astype(str)
	out["embedding"] = out["config.model.backbone.region_representation"].fillna("none")
	out["model"] = out["model_name"].astype(str)

	if "config.random_state" in out.columns:
		out["seed"] = out["config.random_state"].astype(str)
	else:
		out["seed"] = out["run_id"].astype(str)

	out["col_label"] = out.apply(
		lambda r: f"{r['embedding']} | {r['model']}",
		axis=1,
	)
	return out


def _combo_key(dataset: str, target_clean: str, prediction_task: str) -> str:
	return f"{dataset}__{target_clean}__{prediction_task}"


def _aggregate_for_combo(
	df: pd.DataFrame,
	dataset: str,
	target_clean: str,
	prediction_task: str,
) -> tuple[pd.DataFrame, str]:
	df_task = df[
		(df["dataset"] == dataset)
		& (df["target_clean"] == target_clean)
		& (df["prediction_task"] == prediction_task)
	].copy()

	if df_task.empty:
		return pd.DataFrame(), ""

	score_col = _choose_score_column(df_task, prediction_task)
	df_task = df_task[df_task[score_col].notna()].copy()
	if df_task.empty:
		return pd.DataFrame(), ""

	grouped = (
		df_task.groupby(["microstructure", "embedding", "model", "col_label"], dropna=False)[
			score_col
		]
		.agg(["mean", "std", "count"])
		.reset_index()
		.rename(columns={"mean": "score_mean", "std": "score_std", "count": "n_seeds"})
	)
	grouped["score_std"] = grouped["score_std"].fillna(0.0)
	return grouped, score_col


def _plot_heatmap(
	agg_df: pd.DataFrame,
	title: str,
	out_file: Path,
	vmin: float,
	vmax: float,
) -> None:
	if agg_df.empty:
		return

	row_order = sorted(agg_df["microstructure"].astype(str).unique())
	col_order = sorted(agg_df["col_label"].astype(str).unique())

	mean_mat = agg_df.pivot(index="microstructure", columns="col_label", values="score_mean")
	std_mat = agg_df.pivot(index="microstructure", columns="col_label", values="score_std")

	mean_mat = mean_mat.reindex(index=row_order, columns=col_order)
	std_mat = std_mat.reindex(index=row_order, columns=col_order)

	annot = mean_mat.copy().astype(object)
	for r in mean_mat.index:
		for c in mean_mat.columns:
			m = mean_mat.loc[r, c]
			s = std_mat.loc[r, c]
			if pd.isna(m):
				annot.loc[r, c] = ""
			else:
				annot.loc[r, c] = f"{m:.3f} +- {s:.3f}"

	fig_w = max(6.6, 1.0 + 0.8 * len(col_order))
	fig_h = max(2.8, 1.1 + 0.7 * len(row_order))
	fig, ax = plt.subplots(figsize=(fig_w, fig_h))

	sns.heatmap(
		mean_mat,
		ax=ax,
		cmap="YlGnBu",
		vmin=vmin,
		vmax=vmax,
		annot=annot,
		fmt="",
		linewidths=0.6,
		linecolor="white",
		cbar_kws={"label": "Score"},
		annot_kws={"fontsize": 7},
	)

	ax.set_title(title)
	ax.set_xlabel("Embedding | Model")
	ax.set_ylabel("Microstructure")
	ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
	ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

	fig.savefig(out_file)
	plt.close(fig)


def plot_region_heatmaps(
	parquet_path: str,
	out_dir: str = "exp_outputs/summary/plots/region_viz",
) -> list[Path]:
	apply_miccai_style()

	df = pd.read_parquet(parquet_path)
	required = {
		"dataset",
		"target",
		"prediction_task",
		"primary_metric",
		"model_name",
		"config.model.backbone.region_representation",
	}
	missing = sorted(required - set(df.columns))
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	df = _normalize_columns(df)
	out_path = Path(out_dir)
	out_path.mkdir(parents=True, exist_ok=True)

	combo_tables: dict[str, pd.DataFrame] = {}
	for dataset, target_clean, task, _title in COMBOS:
		key = _combo_key(dataset, target_clean, task)
		agg_df, _score_col = _aggregate_for_combo(df, dataset, target_clean, task)
		combo_tables[key] = agg_df

	all_means = [
		t["score_mean"].to_numpy(dtype=float)
		for t in combo_tables.values()
		if not t.empty
	]
	if not all_means:
		raise RuntimeError("No data found for any requested dataset-task combination")

	merged_scores = np.concatenate(all_means)
	vmin = float(np.nanmin(merged_scores))
	vmax = float(np.nanmax(merged_scores))
	if np.isclose(vmin, vmax):
		vmin -= 1e-6
		vmax += 1e-6

	out_files: list[Path] = []
	for dataset, target_clean, task, title in COMBOS:
		key = _combo_key(dataset, target_clean, task)
		agg_df = combo_tables[key]
		if agg_df.empty:
			warnings.warn(f"Skipping empty combo: {title}")
			continue

		file_stub = f"heatmap_{dataset}_{target_clean.lower()}_{task}".replace(" ", "_")
		out_file = out_path / f"{file_stub}.pdf"
		_plot_heatmap(agg_df, title=title, out_file=out_file, vmin=vmin, vmax=vmax)
		out_files.append(out_file)

	return out_files


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Create per-dataset-task performance heatmaps with mean +- std annotations"
	)
	parser.add_argument(
		"--input",
		default="exp_outputs/summary/comprehensive_results.parquet",
		help="Input parquet path",
	)
	parser.add_argument(
		"--outdir",
		default="exp_outputs/summary/plots/region_viz",
		help="Output directory",
	)
	args = parser.parse_args()

	out_files = plot_region_heatmaps(args.input, args.outdir)
	print("Saved heatmaps:")
	for p in out_files:
		print(" -", p)


if __name__ == "__main__":
	main()
