"""
brain_consistency_map.py
------------------------
Brain visualizations for consistency of region importance across pipelines.

For each dataset-task combination, this script computes per-region metrics over
all pipelines (microstructure, embedding, model, seed) and renders maps:

- Stability score: abs(mean(coef)) / (std(coef) + 1e-6)
- Sign agreement: proportion matching the sign of mean(coef)
- Mean importance: mean(coef) with diverging colormap

Optional:
- Selection frequency: fraction of pipelines where region is in top 20% by
  absolute coefficient.
- Difference maps for stability between selected dataset-task pairs.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
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


@dataclass
class ComboResult:
	combo_label: str
	combo_tag: str
	metrics_df: pd.DataFrame
	atlas_info: dict


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
	out["pipeline_id"] = out.apply(
		lambda r: (
			f"{r['dataset']} | {r['target_clean']} | {r['prediction_task']}"
			f" | {r['microstructure']} | {r['embedding']} | {r['model']} | seed={r['seed']}"
		),
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

	# Average possible duplicate run entries by textual pipeline key.
	mat = merged.groupby("pipeline_id", dropna=False)[region_cols].mean(numeric_only=True)
	return mat


def _selection_frequency(pipeline_region: pd.DataFrame, top_frac: float = 0.2) -> pd.Series:
	if pipeline_region.empty:
		return pd.Series(dtype=float)

	n_regions = pipeline_region.shape[1]
	top_k = max(1, int(np.ceil(top_frac * n_regions)))
	counts = pd.Series(0.0, index=pipeline_region.columns, dtype=float)

	for _, row in pipeline_region.iterrows():
		chosen = row.abs().sort_values(ascending=False).head(top_k).index
		counts.loc[chosen] += 1.0

	counts = counts / float(pipeline_region.shape[0])
	return counts


def _minmax_norm(values: pd.Series, vmin: float, vmax: float) -> pd.Series:
	if np.isclose(vmin, vmax):
		return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
	return (values - vmin) / (vmax - vmin)


def _label_value_map(series: pd.Series) -> dict[int, float]:
	out: dict[int, float] = {}
	for k, v in series.items():
		try:
			label = int(str(k).split(":")[-1])
			out[label] = float(v)
		except Exception:
			continue
	return out


def _surface_texture_from_label_map(surface_atlas: dict, label_values: dict[int, float]) -> tuple[np.ndarray, np.ndarray, float]:
	parcel_labels = np.asarray(surface_atlas["parcel_labels"]).astype(np.int32)
	n_left = int(surface_atlas["n_left_vertices"])
	texture = np.zeros(parcel_labels.shape[0], dtype=np.float32)
	for label, value in label_values.items():
		texture[parcel_labels == int(label)] = float(value)
	return texture[:n_left], texture[n_left:], float(np.nanmax(np.abs(texture)) if texture.size else 1.0)


def _plot_surface_metric(
	surface_atlas: dict,
	label_values: dict[int, float],
	title: str,
	out_file: Path,
	cmap: str,
	vmin: float,
	vmax: float,
	symmetric: bool,
) -> None:
	from nilearn import plotting

	left_mesh = surface_atlas["left_mesh"]
	right_mesh = surface_atlas["right_mesh"]
	tex_left, tex_right, _ = _surface_texture_from_label_map(surface_atlas, label_values)

	fig = plt.figure(figsize=(12.8, 4.8))
	ax1 = fig.add_subplot(1, 2, 1, projection="3d")
	ax2 = fig.add_subplot(1, 2, 2, projection="3d")

	plotting.plot_surf_stat_map(
		left_mesh,
		tex_left,
		hemi="left",
		view="lateral",
		cmap=cmap,
		darkness=None,
		symmetric_cbar=symmetric,
		colorbar=False,
		vmin=vmin,
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
		darkness=None,
		symmetric_cbar=symmetric,
		colorbar=True,
		vmin=vmin,
		vmax=vmax,
		axes=ax2,
		title="Right",
	)

	fig.suptitle(title)
	out_file.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_file, dpi=180, bbox_inches="tight")
	plt.close(fig)


def _plot_volume_metric(
	atlas_path: str,
	label_values: dict[int, float],
	title: str,
	out_file: Path,
	cmap: str,
	vmin: float,
	vmax: float,
	symmetric: bool,
) -> None:
	from nilearn import image, plotting

	atlas_img = image.load_img(atlas_path)
	atlas_data = np.asarray(atlas_img.get_fdata())
	stat_data = np.zeros_like(atlas_data, dtype=np.float32)
	for label, value in label_values.items():
		stat_data[atlas_data == int(label)] = float(value)

	stat_img = image.new_img_like(atlas_img, stat_data)
	display = plotting.plot_stat_map(
		stat_img,
		bg_img=atlas_img,
		title=title,
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		symmetric_cbar=symmetric,
	)
	out_file.parent.mkdir(parents=True, exist_ok=True)
	display.savefig(str(out_file))
	display.close()


def _plot_metric_on_brain(
	atlas_info: dict,
	values_by_region: pd.Series,
	title: str,
	out_file: Path,
	cmap: str,
	vmin: float,
	vmax: float,
	symmetric: bool,
) -> None:
	label_values = _label_value_map(values_by_region)
	if not label_values:
		warnings.warn(f"No valid region labels to plot for {out_file.name}")
		return

	atlas_type = atlas_info.get("atlas_type", "surface_schaefer")
	if atlas_type == "surface_schaefer":
		_plot_surface_metric(
			surface_atlas=atlas_info,
			label_values=label_values,
			title=title,
			out_file=out_file,
			cmap=cmap,
			vmin=vmin,
			vmax=vmax,
			symmetric=symmetric,
		)
		return

	if atlas_type == "volume":
		_plot_volume_metric(
			atlas_path=str(atlas_info["atlas_path"]),
			label_values=label_values,
			title=title,
			out_file=out_file,
			cmap=cmap,
			vmin=vmin,
			vmax=vmax,
			symmetric=symmetric,
		)
		return

	raise ValueError(f"Unsupported atlas_type: {atlas_type}")


def _compute_combo_metrics(
	comp: pd.DataFrame,
	coef_all: pd.DataFrame,
	dataset: str,
	target_clean: str,
	task: str,
	combo_label: str,
	combo_tag: str,
	experiments_root: str,
	top_pipelines: int | None,
) -> ComboResult | None:
	sub = comp[
		(comp["dataset"] == dataset)
		& (comp["target_clean"] == target_clean)
		& (comp["prediction_task"] == task)
	].copy()
	if sub.empty:
		warnings.warn(f"Skipping combo with no comprehensive rows: {combo_label}")
		return None

	if top_pipelines is not None and top_pipelines > 0:
		score_col = _choose_score_column(sub, task)
		sub = sub[sub[score_col].notna()].copy()
		sub = sub.sort_values(score_col, ascending=False).head(top_pipelines)

	run_ids = set(sub["run_id"].astype(str).unique())
	coef_sub = coef_all[coef_all["run_id"].astype(str).isin(run_ids)].copy()
	mat = _pipeline_region_matrix(sub, coef_sub)
	if mat.empty or mat.shape[0] < 2:
		warnings.warn(f"Skipping combo with insufficient pipeline coefficients: {combo_label}")
		return None

	mean_w = mat.mean(axis=0)
	std_w = mat.std(axis=0, ddof=0)
	sign_ref = np.where(mean_w.to_numpy(dtype=float) >= 0.0, 1.0, -1.0)
	sign_vals = np.where(mat.to_numpy(dtype=float) >= 0.0, 1.0, -1.0)
	sign_agreement = (sign_vals == sign_ref.reshape(1, -1)).mean(axis=0)
	stability = np.abs(mean_w.to_numpy(dtype=float)) / (std_w.to_numpy(dtype=float) + 1e-6)
	sel_freq = _selection_frequency(mat, top_frac=0.2)

	metrics_df = pd.DataFrame(
		{
			"region_id": mean_w.index.astype(str),
			"mean_w": mean_w.to_numpy(dtype=float),
			"sign_agreement": sign_agreement.astype(float),
			"stability": stability.astype(float),
			"selection_frequency": sel_freq.reindex(mean_w.index).fillna(0.0).to_numpy(dtype=float),
		}
	)

	anchor_run = str(sub["run_id"].astype(str).iloc[0])
	try:
		atlas_info = load_atlas_from_run(anchor_run, experiments_root=experiments_root)
	except Exception as exc:
		warnings.warn(f"Could not load atlas for {combo_label} (run_id={anchor_run}): {exc}")
		return None

	return ComboResult(
		combo_label=combo_label,
		combo_tag=combo_tag,
		metrics_df=metrics_df,
		atlas_info=atlas_info,
	)


def create_brain_consistency_maps(
	comprehensive_path: str,
	coefficients_root: str = "exp_outputs/experiments",
	out_dir: str = "exp_outputs/summary/plots/region_viz",
	top_pipelines: int | None = None,
	diff_a: str | None = None,
	diff_b: str | None = None,
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

	comp = _normalize_comp(comp)
	all_run_ids = set(comp["run_id"].astype(str).dropna().unique())
	coef_all = _load_coefficients_for_runs(Path(coefficients_root), all_run_ids)
	if coef_all.empty:
		raise RuntimeError("No matching coefficient rows found for comprehensive run_id values")

	results: list[ComboResult] = []
	for ds, target, task, label, tag in COMBOS:
		res = _compute_combo_metrics(
			comp=comp,
			coef_all=coef_all,
			dataset=ds,
			target_clean=target,
			task=task,
			combo_label=label,
			combo_tag=tag,
			experiments_root=coefficients_root,
			top_pipelines=top_pipelines,
		)
		if res is not None:
			results.append(res)

	if not results:
		raise RuntimeError("No dataset-task result could be computed")

	all_metrics = pd.concat([r.metrics_df.assign(combo=r.combo_tag) for r in results], ignore_index=True)

	stab_min = float(all_metrics["stability"].min())
	stab_max = float(all_metrics["stability"].max())
	sign_min = float(all_metrics["sign_agreement"].min())
	sign_max = float(all_metrics["sign_agreement"].max())
	mean_abs_max = float(np.max(np.abs(all_metrics["mean_w"].to_numpy(dtype=float))))
	if mean_abs_max <= 0 or np.isnan(mean_abs_max):
		mean_abs_max = 1.0

	out_path = Path(out_dir)
	out_path.mkdir(parents=True, exist_ok=True)

	generated: list[Path] = []
	combo_tables: dict[str, pd.DataFrame] = {}
	combo_atlas: dict[str, dict] = {}

	for res in results:
		df = res.metrics_df.copy()
		df["stability_norm"] = _minmax_norm(df["stability"], stab_min, stab_max)
		df["sign_agreement_norm"] = _minmax_norm(df["sign_agreement"], sign_min, sign_max)
		df["selection_frequency_norm"] = df["selection_frequency"]  # already in [0,1]
		combo_tables[res.combo_tag] = df
		combo_atlas[res.combo_tag] = res.atlas_info

		table_file = out_path / f"brain_consistency_metrics_{res.combo_tag}.parquet"
		df.to_parquet(table_file, index=False)
		generated.append(table_file)

		stab_file = out_path / f"brain_stability_{res.combo_tag}.png"
		sign_file = out_path / f"brain_sign_consistency_{res.combo_tag}.png"
		mean_file = out_path / f"brain_mean_importance_{res.combo_tag}.png"

		_plot_metric_on_brain(
			atlas_info=res.atlas_info,
			values_by_region=df.set_index("region_id")["stability_norm"],
			title=f"Region Stability Across Pipelines - {res.combo_label}",
			out_file=stab_file,
			cmap="viridis",
			vmin=0.0,
			vmax=1.0,
			symmetric=False,
		)
		_plot_metric_on_brain(
			atlas_info=res.atlas_info,
			values_by_region=df.set_index("region_id")["sign_agreement_norm"],
			title=f"Sign Consistency Across Pipelines - {res.combo_label}",
			out_file=sign_file,
			cmap="magma",
			vmin=0.0,
			vmax=1.0,
			symmetric=False,
		)
		_plot_metric_on_brain(
			atlas_info=res.atlas_info,
			values_by_region=df.set_index("region_id")["mean_w"],
			title=f"Average Region Contribution - {res.combo_label}",
			out_file=mean_file,
			cmap="RdBu_r",
			vmin=-mean_abs_max,
			vmax=mean_abs_max,
			symmetric=True,
		)

		generated.extend([stab_file, sign_file, mean_file])

	if diff_a and diff_b:
		if diff_a not in combo_tables or diff_b not in combo_tables:
			warnings.warn("Skipping difference map: diff tags not found among available combos")
		else:
			a = combo_tables[diff_a].set_index("region_id")
			b = combo_tables[diff_b].set_index("region_id")
			common = a.index.intersection(b.index)
			if len(common) == 0:
				warnings.warn("Skipping difference map: no common region ids")
			else:
				diff = a.loc[common, "stability_norm"] - b.loc[common, "stability_norm"]
				atlas_for_diff = combo_atlas[diff_a]
				diff_file = out_path / f"brain_stability_diff_{diff_a}_minus_{diff_b}.png"
				_plot_metric_on_brain(
					atlas_info=atlas_for_diff,
					values_by_region=diff,
					title=f"Stability Difference: {diff_a} - {diff_b}",
					out_file=diff_file,
					cmap="RdBu_r",
					vmin=-1.0,
					vmax=1.0,
					symmetric=True,
				)
				generated.append(diff_file)

	return generated


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Brain consistency maps of region importance across pipelines"
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
		"--top-pipelines",
		type=int,
		default=None,
		help="Optional: keep top-N pipelines per dataset-task by performance before metrics",
	)
	parser.add_argument(
		"--diff-a",
		default=None,
		help="Optional combo tag for stability difference map minuend (e.g., camcan_sex)",
	)
	parser.add_argument(
		"--diff-b",
		default=None,
		help="Optional combo tag for stability difference map subtrahend (e.g., hcp_sex)",
	)
	args = parser.parse_args()

	outputs = create_brain_consistency_maps(
		comprehensive_path=args.comprehensive,
		coefficients_root=args.coeff_root,
		out_dir=args.outdir,
		top_pipelines=args.top_pipelines,
		diff_a=args.diff_a,
		diff_b=args.diff_b,
	)
	print("Saved consistency outputs:")
	for p in outputs:
		print(" -", p)


if __name__ == "__main__":
	main()
