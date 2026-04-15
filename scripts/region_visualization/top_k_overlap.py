"""
top_k_overlap.py
----------------
Compute top-k region overlap distributions across datasets.

Steps:
1) For each (dataset, task, pipeline), select top-k regions by abs(coefficient).
2) For each dataset, compute pairwise overlap among pipelines:
     overlap = |intersection(top-k_i, top-k_j)| / k
3) Summarize overlap distributions and plot per dataset.

Optional:
- Color pairwise points/violins by whether the pipeline pair shares microstructure.
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMBOS = [
	("hcp", "Sex", "binary_classification", "HCP"),
	("camcan", "Sex", "binary_classification", "CamCAN"),
	("camcan", "Age", "regression", "CamCAN"),
	("abide", "DX_GROUP", "binary_classification", "ABIDE II"),
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


def _normalize_comp(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	out["target_clean"] = out["target"].map(_clean_target)
	out["microstructure"] = out["primary_metric"].astype(str)
	out["embedding"] = out["config.model.backbone.region_representation"].fillna("none")
	out["model"] = out["model_name"].astype(str)
	if "config.random_state" in out.columns:
		out["seed"] = out["config.random_state"].astype(str)
	else:
		out["seed"] = out["run_id"].astype(str)
	out["task_id"] = out.apply(
		lambda r: f"{r['dataset']} | {r['target_clean']} | {r['prediction_task']}",
		axis=1,
	)
	out["pipeline_id"] = out.apply(
		lambda r: (
			f"{r['dataset']} | {r['target_clean']} | {r['prediction_task']}"
			f" | {r['microstructure']} | {r['embedding']} | {r['model']} | seed={r['seed']}"
		),
		axis=1,
	)
	return out


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


def _pipeline_region_vectors(comp: pd.DataFrame, coef: pd.DataFrame) -> pd.DataFrame:
	if comp.empty or coef.empty:
		return pd.DataFrame()

	region_cols = _region_columns(coef)
	if not region_cols:
		return pd.DataFrame()

	coef_agg = coef.groupby("run_id", dropna=False)[region_cols].mean(numeric_only=True)
	coef_agg.index = coef_agg.index.astype(str)

	meta_cols = [
		"run_id",
		"dataset",
		"target_clean",
		"prediction_task",
		"task_id",
		"pipeline_id",
		"microstructure",
		"embedding",
		"model",
		"seed",
	]
	meta = comp[meta_cols].copy()
	meta["run_id"] = meta["run_id"].astype(str)
	meta = meta.drop_duplicates(subset=["run_id", "pipeline_id"])

	merged = meta.merge(coef_agg.reset_index(), on="run_id", how="inner")
	if merged.empty:
		return pd.DataFrame()

	group_keys = [
		"dataset",
		"target_clean",
		"prediction_task",
		"task_id",
		"pipeline_id",
		"microstructure",
		"embedding",
		"model",
		"seed",
	]
	# Average if duplicated (defensive).
	pipe_vec = merged.groupby(group_keys, dropna=False)[region_cols].mean(numeric_only=True).reset_index()
	return pipe_vec


def _top_k_regions(values: pd.Series, k: int) -> set[str]:
	abs_vals = values.abs().sort_values(ascending=False)
	return set(abs_vals.head(k).index.astype(str).tolist())


def _pairwise_overlaps_for_dataset(pipe_regions_df: pd.DataFrame, k: int) -> pd.DataFrame:
	if pipe_regions_df.empty:
		return pd.DataFrame()

	region_cols = [c for c in pipe_regions_df.columns if c not in {
		"dataset",
		"target_clean",
		"prediction_task",
		"task_id",
		"pipeline_id",
		"microstructure",
		"embedding",
		"model",
		"seed",
	}]

	if not region_cols:
		return pd.DataFrame()

	records: list[dict[str, object]] = []
	for dataset_name, ds_df in pipe_regions_df.groupby("dataset", dropna=False):
		pipelines = []
		for _, row in ds_df.iterrows():
			region_vals = row[region_cols]
			top_set = _top_k_regions(region_vals, k)
			if not top_set:
				continue
			pipelines.append(
				{
					"pipeline_id": str(row["pipeline_id"]),
					"microstructure": str(row["microstructure"]),
					"task_id": str(row["task_id"]),
					"top_set": top_set,
				}
			)

		for a, b in itertools.combinations(pipelines, 2):
			inter = len(a["top_set"].intersection(b["top_set"]))
			overlap = inter / float(k)
			records.append(
				{
					"dataset": str(dataset_name),
					"pipeline_a": a["pipeline_id"],
					"pipeline_b": b["pipeline_id"],
					"task_a": a["task_id"],
					"task_b": b["task_id"],
					"same_microstructure": bool(a["microstructure"] == b["microstructure"]),
					"overlap": float(overlap),
				}
			)

	return pd.DataFrame(records)


def _dataset_display_name(dataset: str) -> str:
	mapping = {
		"hcp": "HCP",
		"camcan": "CamCAN",
		"abide": "ABIDE II",
	}
	return mapping.get(str(dataset), str(dataset))


def _plot_overlap_distribution(
	overlap_df: pd.DataFrame,
	out_file: Path,
	color_by_microstructure: bool,
) -> None:
	if overlap_df.empty:
		return

	plot_df = overlap_df.copy()
	plot_df["dataset_display"] = plot_df["dataset"].map(_dataset_display_name)
	order = [d for d in ["HCP", "CamCAN", "ABIDE II"] if d in set(plot_df["dataset_display"])]

	fig, ax = plt.subplots(figsize=(7.2, 3.2))

	if color_by_microstructure:
		sns.violinplot(
			data=plot_df,
			x="dataset_display",
			y="overlap",
			hue="same_microstructure",
			order=order,
			inner="quartile",
			density_norm="width",
			cut=0,
			linewidth=0.8,
			ax=ax,
		)
		ax.legend(title="Same microstructure", loc="upper right", frameon=True)
	else:
		sns.violinplot(
			data=plot_df,
			x="dataset_display",
			y="overlap",
			order=order,
			inner="quartile",
			density_norm="width",
			cut=0,
			linewidth=0.8,
			color="#6BAED6",
			ax=ax,
		)

	ax.set_xlabel("")
	ax.set_ylabel("Top-k overlap")
	ax.set_ylim(0.0, 1.0)
	ax.set_title("Top-k region overlap across datasets")
	ax.grid(axis="y", linestyle="--", alpha=0.3)
	ax.grid(axis="x", visible=False)
	sns.despine(ax=ax, top=True, right=True)

	fig.savefig(out_file)
	plt.close(fig)


def compute_top_k_overlap(
	comprehensive_path: str,
	coefficients_root: str = "exp_outputs/experiments",
	out_dir: str = "exp_outputs/summary/plots/region_viz",
	k: int = 10,
	color_by_microstructure: bool = False,
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

	# Keep requested combos only.
	combo_mask = pd.Series(False, index=comp.index)
	for ds, target, task, _ in COMBOS:
		combo_mask |= (
			(comp["dataset"] == ds)
			& (comp["target_clean"] == target)
			& (comp["prediction_task"] == task)
		)
	comp = comp[combo_mask].copy()
	if comp.empty:
		raise RuntimeError("No rows left after filtering requested dataset-task combos")

	run_ids = set(comp["run_id"].astype(str).dropna().unique())
	coef = _load_coefficients_for_runs(Path(coefficients_root), run_ids)
	if coef.empty:
		raise RuntimeError("No matching coefficient rows found for filtered run_id values")
	coef["run_id"] = coef["run_id"].astype(str)

	pipe_regions = _pipeline_region_vectors(comp, coef)
	if pipe_regions.empty:
		raise RuntimeError("Could not build pipeline-level region coefficient vectors")

	overlap_df = _pairwise_overlaps_for_dataset(pipe_regions, k=k)
	if overlap_df.empty:
		raise RuntimeError("Insufficient pipeline pairs to compute overlap distributions")

	out_path = Path(out_dir)
	out_path.mkdir(parents=True, exist_ok=True)

	color_tag = "same_micro" if color_by_microstructure else "all_pairs"
	summary_file = out_path / f"top_{k}_overlap_pairs_{color_tag}.parquet"
	plot_file = out_path / f"top_{k}_overlap_distribution_{color_tag}.pdf"

	overlap_df.to_parquet(summary_file, index=False)
	_plot_overlap_distribution(
		overlap_df=overlap_df,
		out_file=plot_file,
		color_by_microstructure=color_by_microstructure,
	)

	return summary_file, plot_file


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Top-k region overlap across datasets"
	)
	parser.add_argument(
		"--comprehensive",
		default="exp_outputs/summary/comprehensive_results.parquet",
		help="Path to comprehensive parquet table",
	)
	parser.add_argument(
		"--coeff-root",
		default="exp_outputs/experiments",
		help="Root containing experiment coefficient outputs",
	)
	parser.add_argument(
		"--outdir",
		default="exp_outputs/summary/plots/region_viz",
		help="Output directory",
	)
	parser.add_argument(
		"--k",
		type=int,
		default=10,
		help="Top-k regions for overlap computation",
	)
	parser.add_argument(
		"--color-same-microstructure",
		action="store_true",
		help="Color overlap distribution by same vs different microstructure",
	)
	args = parser.parse_args()

	if args.k <= 0:
		raise ValueError("--k must be > 0")

	summary_file, plot_file = compute_top_k_overlap(
		comprehensive_path=args.comprehensive,
		coefficients_root=args.coeff_root,
		out_dir=args.outdir,
		k=args.k,
		color_by_microstructure=args.color_same_microstructure,
	)
	print("Saved overlap pair table to", summary_file)
	print("Saved overlap plot to", plot_file)


if __name__ == "__main__":
	main()
