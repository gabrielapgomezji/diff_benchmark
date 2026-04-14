from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diff_benchmark.analysis.region_coefficients import load_atlas_from_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "coefficients_long.parquet"
BEST_RUNS_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "best_runs_by_config.parquet"
OUTPUT_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "coefficients_selected_normalized.parquet"
STABILITY_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "region_stability_metrics.parquet"
MAPS_DIR = PROJECT_ROOT / "exp_outputs" / "summary" / "brain_maps"


def _pick_group_columns(df: pd.DataFrame) -> list[str]:
	candidates = ["microstructure", "model_type", "embedding", "task", "dataset"]
	if "model_type" not in df.columns and "model" in df.columns:
		df["model_type"] = df["model"]
	return [c for c in candidates if c in df.columns]


def _select_best_runs(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
	score_df = (
		df[["run_id", "test_score"] + group_cols]
		.dropna(subset=["test_score"])
		.drop_duplicates()
	)

	run_scores = (
		score_df.groupby(["run_id"] + group_cols, dropna=False, as_index=False)["test_score"]
		.mean()
		.rename(columns={"test_score": "mean_test_score"})
	)

	best = (
		run_scores.sort_values("mean_test_score", ascending=False)
		.groupby(group_cols, dropna=False, as_index=False)
		.head(1)
		.reset_index(drop=True)
	)
	return best


def _normalize_selected(df_selected: pd.DataFrame) -> pd.DataFrame:
	out = df_selected.copy()
	l2_norm = out.groupby("run_id")["coef"].transform(lambda s: (s.pow(2).sum()) ** 0.5)
	out["coef_norm"] = out["coef"].where(l2_norm == 0, out["coef"] / l2_norm)
	return out


def _topk_selection(group: pd.DataFrame, k: float = 0.1) -> pd.Series:
	abs_coef = group["coef_norm"].abs()
	thresh = abs_coef.quantile(1.0 - float(k))
	return (abs_coef >= thresh).astype(int)


def _add_topk_selection(df: pd.DataFrame, k: float = 0.1) -> pd.DataFrame:
	out = df.copy()
	out["selected"] = out.groupby("run_id")["coef_norm"].transform(
		lambda s: (s.abs() >= s.abs().quantile(1.0 - float(k))).astype(int)
	)
	return out


def _compute_metrics(group: pd.DataFrame) -> pd.Series:
	selected = group[group["selected"] == 1]

	if len(selected) == 0:
		return pd.Series(
			{
				"selection_freq": 0.0,
				"sign_consistency": 0.0,
				"effect_median": 0.0,
				"effect_iqr": 0.0,
			}
		)

	signs = selected["coef_norm"].apply(lambda x: 1 if x > 0 else -1)
	return pd.Series(
		{
			"selection_freq": float(group["selected"].mean()),
			"sign_consistency": float(signs.value_counts(normalize=True).max()),
			"effect_median": float(selected["coef_norm"].median()),
			"effect_iqr": float(
				selected["coef_norm"].quantile(0.75) - selected["coef_norm"].quantile(0.25)
			),
		}
	)


def _label_value_map(values_by_region: pd.Series) -> dict[int, float]:
	out: dict[int, float] = {}
	for region, value in values_by_region.items():
		try:
			label = int(str(region).split(":")[-1])
			out[label] = float(value)
		except Exception:
			continue
	return out


def _surface_texture_from_label_map(surface_atlas: dict, label_values: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
	parcel_labels = np.asarray(surface_atlas["parcel_labels"]).astype(np.int32)
	n_left = int(surface_atlas["n_left_vertices"])
	texture = np.zeros(parcel_labels.shape[0], dtype=np.float32)
	for label, value in label_values.items():
		texture[parcel_labels == int(label)] = float(value)
	return texture[:n_left], texture[n_left:]


def _plot_surface_metric(
	surface_atlas: dict,
	label_values: dict[int, float],
	title: str,
	out_file: Path,
	vmin: float,
	vmax: float,
	symmetric: bool,
	cmap: str,
) -> None:
	from nilearn import plotting

	left_mesh = surface_atlas["left_mesh"]
	right_mesh = surface_atlas["right_mesh"]
	tex_left, tex_right = _surface_texture_from_label_map(surface_atlas, label_values)

	fig = plt.figure(figsize=(12, 4.8))
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


def _plot_dataset_task_maps(metrics_main: pd.DataFrame, normalized_df: pd.DataFrame, maps_dir: Path) -> None:
	if metrics_main.empty:
		return

	for (dataset, task), combo in metrics_main.groupby(["dataset", "task"], dropna=False):
		run_candidates = normalized_df[
			(normalized_df["dataset"].astype(str) == str(dataset))
			& (normalized_df["task"].astype(str) == str(task))
		]["run_id"].astype(str)
		if run_candidates.empty:
			continue

		anchor_run = run_candidates.iloc[0]
		try:
			atlas_info = load_atlas_from_run(anchor_run, experiments_root=PROJECT_ROOT / "exp_outputs" / "experiments")
		except Exception as exc:
			warnings.warn(f"Skipping brain map for {dataset}/{task}: could not load atlas ({exc})")
			continue

		label_values = _label_value_map(combo.set_index("region")["selection_freq"])
		if not label_values:
			warnings.warn(f"Skipping brain map for {dataset}/{task}: no plottable numeric region labels")
			continue

		tag = f"{str(dataset).lower()}_{str(task).lower().replace(' ', '_')}"
		out_file = maps_dir / f"selection_frequency_{tag}.png"
		vals = combo["selection_freq"].astype(float)
		vmin, vmax = float(vals.min()), float(vals.max())

		atlas_type = atlas_info.get("atlas_type", "surface_schaefer")
		if atlas_type != "surface_schaefer":
			warnings.warn(
				f"Atlas type {atlas_type} for {dataset}/{task} is not handled in this script; skipping map"
			)
			continue

		try:
			_plot_surface_metric(
				surface_atlas=atlas_info,
				label_values=label_values,
				title=f"Selection Frequency | {dataset} | {task}",
				out_file=out_file,
				vmin=vmin,
				vmax=vmax,
				symmetric=False,
				cmap="viridis",
			)
		except Exception as exc:
			warnings.warn(f"Failed plotting brain map for {dataset}/{task}: {exc}")


def build_selected_normalized_coefficients(
	input_table: Path = INPUT_TABLE,
	best_runs_table: Path = BEST_RUNS_TABLE,
	output_table: Path = OUTPUT_TABLE,
	stability_table: Path = STABILITY_TABLE,
	maps_dir: Path = MAPS_DIR,
	top_k: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	df = pd.read_parquet(input_table)
	required = {"run_id", "coef", "test_score"}
	missing = required - set(df.columns)
	if missing:
		missing_txt = ", ".join(sorted(missing))
		raise ValueError(f"Missing required columns in input table: {missing_txt}")

	group_cols = _pick_group_columns(df)
	if not group_cols:
		raise ValueError(
			"No grouping columns found. Need at least one of: "
			"microstructure, model_type/model, embedding, task, dataset"
		)

	best_runs = _select_best_runs(df, group_cols)
	selected = df[df["run_id"].isin(best_runs["run_id"])].copy()
	normalized = _normalize_selected(selected)
	normalized = _add_topk_selection(normalized, k=top_k)
	metrics_main = (
		normalized.groupby(["dataset", "task", "region"], dropna=False)[["selected", "coef_norm"]]
		.apply(_compute_metrics)
		.reset_index()
	)

	best_runs_table.parent.mkdir(parents=True, exist_ok=True)
	output_table.parent.mkdir(parents=True, exist_ok=True)
	stability_table.parent.mkdir(parents=True, exist_ok=True)
	best_runs.to_parquet(best_runs_table, index=False)
	normalized.to_parquet(output_table, index=False)
	metrics_main.to_parquet(stability_table, index=False)
	_plot_dataset_task_maps(metrics_main, normalized, maps_dir)

	return best_runs, normalized, metrics_main


if __name__ == "__main__":
	best_df, norm_df, metrics_df = build_selected_normalized_coefficients()
	print(f"Saved best runs: {len(best_df)} rows -> {BEST_RUNS_TABLE}")
	print(f"Saved normalized coefficients: {len(norm_df)} rows -> {OUTPUT_TABLE}")
	print(f"Saved stability metrics: {len(metrics_df)} rows -> {STABILITY_TABLE}")
	print(f"Saved brain maps under: {MAPS_DIR}")
