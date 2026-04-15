"""
stability_map.py
----------------
Compute region stability scores per dataset-task combination.

For each combo:
1) Collect region coefficients across pipelines (seed/model/embedding).
2) Compute region mean and std.
3) Stability = abs(mean) / (std + 1e-6).
4) Min-max normalize stability to [0, 1].
5) Save:
   - top-15 stable regions bar plot
   - stability histogram
6) Optional brain projection using run atlas metadata.

Cross-dataset summary:
- Identify regions stable in multiple datasets from top-k lists.
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

from diff_benchmark.analysis.region_coefficients import load_atlas_from_run


COMBOS = [
	("hcp", "Sex", "binary_classification", "HCP (sex)", "hcp_sex"),
	("camcan", "Sex", "binary_classification", "CamCAN (sex)", "camcan_sex"),
	("camcan", "Age", "regression", "CamCAN (age)", "camcan_age"),
	("abide", "DX_GROUP", "binary_classification", "ABIDE II (autism)", "abide_autism"),
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
			"xtick.labelsize": 7,
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


def _normalize_comprehensive(df: pd.DataFrame) -> pd.DataFrame:
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
		lambda r: (
			f"{r['dataset']} | {r['target_clean']} | {r['prediction_task']}"
			f" | {r['microstructure']} | {r['embedding']} | {r['model']} | seed={r['seed']}"
		),
		axis=1,
	)
	return out


def _region_columns(df_coef: pd.DataFrame) -> list[str]:
	return [
		c
		for c in df_coef.columns
		if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(df_coef[c])
	]


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


def _pipeline_region_matrix(comp_sub: pd.DataFrame, coef_sub: pd.DataFrame) -> pd.DataFrame:
	if comp_sub.empty or coef_sub.empty:
		return pd.DataFrame()

	region_cols = _region_columns(coef_sub)
	if not region_cols:
		return pd.DataFrame()

	coef_agg = coef_sub.groupby("run_id", dropna=False)[region_cols].mean(numeric_only=True)
	coef_agg.index = coef_agg.index.astype(str)

	meta = comp_sub[["run_id", "pipeline_id"]].copy()
	meta["run_id"] = meta["run_id"].astype(str)
	meta = meta.drop_duplicates(subset=["run_id", "pipeline_id"])

	merged = meta.merge(coef_agg.reset_index(), on="run_id", how="inner")
	if merged.empty:
		return pd.DataFrame()

	return merged.groupby("pipeline_id", dropna=False)[region_cols].mean(numeric_only=True)


def _compute_stability_df(pipeline_region: pd.DataFrame) -> pd.DataFrame:
	mean_coef = pipeline_region.mean(axis=0)
	std_coef = pipeline_region.std(axis=0, ddof=0)
	stability = np.abs(mean_coef.to_numpy(dtype=float)) / (std_coef.to_numpy(dtype=float) + 1e-6)

	df = pd.DataFrame(
		{
			"region_id": mean_coef.index.astype(str),
			"mean_coef": mean_coef.to_numpy(dtype=float),
			"std_coef": std_coef.to_numpy(dtype=float),
			"stability": stability.astype(float),
		}
	)

	vmin = float(df["stability"].min())
	vmax = float(df["stability"].max())
	if np.isclose(vmin, vmax):
		df["stability_norm"] = 0.0
	else:
		df["stability_norm"] = (df["stability"] - vmin) / (vmax - vmin)

	return df.sort_values("stability", ascending=False).reset_index(drop=True)


def _plot_top_regions(stability_df: pd.DataFrame, title: str, out_file: Path, top_n: int = 15) -> None:
	top = stability_df.head(top_n).copy()
	fig_h = max(3.6, 0.28 * len(top) + 1.4)
	fig, ax = plt.subplots(figsize=(7.2, fig_h))
	sns.barplot(data=top, y="region_id", x="stability_norm", color="#2C7FB8", ax=ax)
	ax.set_title(title)
	ax.set_xlabel("Normalized stability")
	ax.set_ylabel("Region")
	ax.set_xlim(0.0, 1.0)
	ax.grid(axis="x", linestyle="--", alpha=0.3)
	ax.grid(axis="y", visible=False)
	sns.despine(ax=ax, top=True, right=True)
	fig.savefig(out_file)
	plt.close(fig)


def _plot_stability_hist(stability_df: pd.DataFrame, title: str, out_file: Path) -> None:
	fig, ax = plt.subplots(figsize=(5.8, 3.2))
	sns.histplot(stability_df["stability_norm"], bins=20, kde=True, color="#74A9CF", ax=ax)
	ax.set_title(title)
	ax.set_xlabel("Normalized stability")
	ax.set_ylabel("Count")
	ax.set_xlim(0.0, 1.0)
	ax.grid(axis="y", linestyle="--", alpha=0.3)
	ax.grid(axis="x", visible=False)
	sns.despine(ax=ax, top=True, right=True)
	fig.savefig(out_file)
	plt.close(fig)


def _label_value_map(series: pd.Series) -> dict[int, float]:
	out: dict[int, float] = {}
	for k, v in series.items():
		try:
			label = int(str(k).split(":")[-1])
			out[label] = float(v)
		except Exception:
			continue
	return out


def _plot_brain_stability(
	atlas_info: dict,
	stability_series: pd.Series,
	title: str,
	out_file: Path,
) -> None:
	label_values = _label_value_map(stability_series)
	if not label_values:
		warnings.warn(f"No plottable label values for {out_file.name}")
		return

	atlas_type = atlas_info.get("atlas_type", "surface_schaefer")
	if atlas_type == "surface_schaefer":
		from nilearn import plotting

		left_mesh = atlas_info["left_mesh"]
		right_mesh = atlas_info["right_mesh"]
		parcel_labels = np.asarray(atlas_info["parcel_labels"]).astype(np.int32)
		n_left = int(atlas_info["n_left_vertices"])

		texture = np.zeros(parcel_labels.shape[0], dtype=np.float32)
		for label, value in label_values.items():
			texture[parcel_labels == int(label)] = float(value)

		tex_left = texture[:n_left]
		tex_right = texture[n_left:]

		fig = plt.figure(figsize=(12.8, 4.8))
		ax1 = fig.add_subplot(1, 2, 1, projection="3d")
		ax2 = fig.add_subplot(1, 2, 2, projection="3d")

		plotting.plot_surf_stat_map(
			left_mesh,
			tex_left,
			hemi="left",
			view="lateral",
			cmap="viridis",
			darkness=None,
			symmetric_cbar=False,
			colorbar=False,
			vmin=0.0,
			vmax=1.0,
			axes=ax1,
			title="Left",
		)
		plotting.plot_surf_stat_map(
			right_mesh,
			tex_right,
			hemi="right",
			view="lateral",
			cmap="viridis",
			darkness=None,
			symmetric_cbar=False,
			colorbar=True,
			vmin=0.0,
			vmax=1.0,
			axes=ax2,
			title="Right",
		)

		fig.suptitle(title)
		fig.savefig(out_file, dpi=180, bbox_inches="tight")
		plt.close(fig)
		return

	if atlas_type == "volume":
		from nilearn import image, plotting

		atlas_img = image.load_img(str(atlas_info["atlas_path"]))
		atlas_data = np.asarray(atlas_img.get_fdata())
		stat_data = np.zeros_like(atlas_data, dtype=np.float32)
		for label, value in label_values.items():
			stat_data[atlas_data == int(label)] = float(value)

		stat_img = image.new_img_like(atlas_img, stat_data)
		disp = plotting.plot_stat_map(
			stat_img,
			bg_img=atlas_img,
			title=title,
			cmap="viridis",
			symmetric_cbar=False,
			vmin=0.0,
			vmax=1.0,
		)
		disp.savefig(str(out_file))
		disp.close()
		return

	raise ValueError(f"Unsupported atlas type: {atlas_type}")


def compute_stability_maps(
	comprehensive_path: str,
	coeff_root: str = "exp_outputs/experiments",
	out_dir: str = "exp_outputs/summary/plots/region_viz",
	project_brain: bool = True,
	top_n: int = 15,
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

	comp = _normalize_comprehensive(comp)
	all_runs = set(comp["run_id"].astype(str).dropna().unique())
	coef_all = _load_coefficients_for_runs(Path(coeff_root), all_runs)
	if coef_all.empty:
		raise RuntimeError("No coefficients matched run_id values from comprehensive table")

	out_path = Path(out_dir)
	out_path.mkdir(parents=True, exist_ok=True)

	produced: list[Path] = []
	stable_registry: list[pd.DataFrame] = []

	for ds, target, task, label, tag in COMBOS:
		sub = comp[
			(comp["dataset"] == ds)
			& (comp["target_clean"] == target)
			& (comp["prediction_task"] == task)
		].copy()

		if sub.empty:
			warnings.warn(f"Skipping combo with no rows: {label}")
			continue

		run_ids = set(sub["run_id"].astype(str).unique())
		coef_sub = coef_all[coef_all["run_id"].astype(str).isin(run_ids)].copy()
		pipeline_region = _pipeline_region_matrix(sub, coef_sub)
		if pipeline_region.empty or pipeline_region.shape[0] < 2:
			warnings.warn(f"Skipping combo with insufficient pipeline coefficients: {label}")
			continue

		stability_df = _compute_stability_df(pipeline_region)
		stability_df["dataset"] = ds
		stability_df["target_clean"] = target
		stability_df["prediction_task"] = task
		stability_df["combo_tag"] = tag

		metrics_file = out_path / f"stability_scores_{tag}.parquet"
		stability_df.to_parquet(metrics_file, index=False)
		produced.append(metrics_file)

		top_file = out_path / f"stability_top{top_n}_{tag}.pdf"
		hist_file = out_path / f"stability_hist_{tag}.pdf"
		_plot_top_regions(
			stability_df=stability_df,
			title=f"Top {top_n} Stable Regions - {label}",
			out_file=top_file,
			top_n=top_n,
		)
		_plot_stability_hist(
			stability_df=stability_df,
			title=f"Stability Distribution - {label}",
			out_file=hist_file,
		)
		produced.extend([top_file, hist_file])

		if project_brain:
			anchor_run = str(sub["run_id"].astype(str).iloc[0])
			try:
				atlas_info = load_atlas_from_run(anchor_run, experiments_root=coeff_root)
				brain_file = out_path / f"stability_brain_{tag}.png"
				_plot_brain_stability(
					atlas_info=atlas_info,
					stability_series=stability_df.set_index("region_id")["stability_norm"],
					title=f"Region Stability Across Pipelines - {label}",
					out_file=brain_file,
				)
				produced.append(brain_file)
			except Exception as exc:
				warnings.warn(f"Could not render brain stability map for {label}: {exc}")

		stable_top = stability_df.head(top_n)[["region_id"]].copy()
		stable_top["dataset"] = ds
		stable_top["combo_tag"] = tag
		stable_registry.append(stable_top)

	if stable_registry:
		stable_all = pd.concat(stable_registry, ignore_index=True)
		summary = (
			stable_all.groupby("region_id", dropna=False)
			.agg(
				n_dataset_tasks=("combo_tag", "nunique"),
				n_datasets=("dataset", "nunique"),
				datasets=("dataset", lambda s: ",".join(sorted(set(map(str, s))))),
				combo_tags=("combo_tag", lambda s: ",".join(sorted(set(map(str, s))))),
			)
			.reset_index()
			.sort_values(["n_datasets", "n_dataset_tasks", "region_id"], ascending=[False, False, True])
		)

		summary_file = out_path / f"stable_regions_shared_top{top_n}.parquet"
		summary.to_parquet(summary_file, index=False)
		produced.append(summary_file)

	return produced


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Compute and visualize region stability scores per dataset-task"
	)
	parser.add_argument(
		"--comprehensive",
		default="exp_outputs/summary/comprehensive_results.parquet",
		help="Path to comprehensive parquet",
	)
	parser.add_argument(
		"--coeff-root",
		default="exp_outputs/experiments",
		help="Root folder containing coefficients.parquet files",
	)
	parser.add_argument(
		"--outdir",
		default="exp_outputs/summary/plots/region_viz",
		help="Output directory",
	)
	parser.add_argument(
		"--top-n",
		type=int,
		default=15,
		help="Top-N regions for bar plot and cross-dataset stable-region summary",
	)
	parser.add_argument(
		"--no-brain",
		action="store_true",
		help="Disable brain atlas projection",
	)
	args = parser.parse_args()

	if args.top_n <= 0:
		raise ValueError("--top-n must be > 0")

	outputs = compute_stability_maps(
		comprehensive_path=args.comprehensive,
		coeff_root=args.coeff_root,
		out_dir=args.outdir,
		project_brain=not args.no_brain,
		top_n=args.top_n,
	)

	if outputs:
		print("Saved stability outputs:")
		for p in outputs:
			print(" -", p)
	else:
		print("No outputs generated")


if __name__ == "__main__":
	main()
