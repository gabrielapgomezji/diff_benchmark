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
RANK_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "region_rank_metrics.parquet"
MAPS_DIR = PROJECT_ROOT / "exp_outputs" / "summary" / "brain_maps"


def _coefficient_mode_settings(coefficient_mode: str) -> tuple[str, bool]:
	mode = str(coefficient_mode).strip().lower()
	if mode in {"abs", "absolute"}:
		return "absolute", False
	if mode in {"sign", "signed"}:
		return "signed", True
	raise ValueError("coefficient_mode must be one of: absolute, abs, signed, sign")


def _coef_importance(series: pd.Series, use_sign: bool) -> pd.Series:
	if use_sign:
		return series
	return series.abs()


def _pick_group_columns(df: pd.DataFrame) -> list[str]:
	candidates = ["microstructure", "model_type", "embedding", "task", "dataset"]
	if "model_type" not in df.columns and "model" in df.columns:
		df["model_type"] = df["model"]
	return [c for c in candidates if c in df.columns]


def _build_exp_id(df: pd.DataFrame) -> pd.Series:
	cols = ["run_id"]
	if "fold" in df.columns:
		cols.append("fold")
	if "seed" in df.columns:
		cols.append("seed")
	return df[cols].astype(str).agg("_".join, axis=1)


def _select_best_runs(df: pd.DataFrame, group_cols: list[str], outlier_iqr_mult: float = 1.5) -> pd.DataFrame:
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
	if run_scores.empty:
		return run_scores

	def _filter_group(group: pd.DataFrame) -> pd.DataFrame:
		out = group.copy()
		scores = out["mean_test_score"].astype(float)
		best_score = float(scores.max())
		gaps = best_score - scores

		q1_gap = float(gaps.quantile(0.25))
		q3_gap = float(gaps.quantile(0.75))
		iqr_gap = q3_gap - q1_gap
		gap_cutoff = q3_gap + float(outlier_iqr_mult) * iqr_gap if iqr_gap > 0 else q3_gap

		q1_score = float(scores.quantile(0.25))
		q3_score = float(scores.quantile(0.75))
		iqr_score = q3_score - q1_score
		score_floor = q1_score - float(outlier_iqr_mult) * iqr_score if iqr_score > 0 else q1_score

		keep_mask = (gaps <= gap_cutoff) & (scores >= score_floor)
		keep_mask = keep_mask | (scores == best_score)

		if int(keep_mask.sum()) == 0:
			keep_mask = scores == best_score

		out["best_test_score"] = best_score
		out["score_gap_from_best"] = gaps
		return out.loc[keep_mask.values]

	grouped = run_scores.groupby(group_cols, dropna=False, as_index=False)
	try:
		selected_runs = grouped.apply(_filter_group, include_groups=False).reset_index(drop=True)
	except TypeError:
		selected_runs = grouped.apply(_filter_group).reset_index(drop=True)
	return selected_runs


def _normalize_selected(df_selected: pd.DataFrame) -> pd.DataFrame:
	out = df_selected.copy()
	l2_norm = out.groupby("exp_id")["coef"].transform(lambda s: (s.pow(2).sum()) ** 0.5)
	out["coef_norm"] = out["coef"].where(l2_norm == 0, out["coef"] / l2_norm)
	return out


def _selection_key_columns(df: pd.DataFrame) -> list[str]:
	keys = [c for c in ["run_id", "fold", "seed"] if c in df.columns]
	if keys:
		return keys
	if "exp_id" in df.columns:
		return ["exp_id"]
	raise ValueError("Cannot build selection groups: expected run_id/fold/seed or exp_id")


def _add_percentile_selection(df: pd.DataFrame, percentile: float = 0.90, use_sign: bool = False) -> pd.DataFrame:
	if not (0.0 < float(percentile) < 1.0):
		raise ValueError("percentile must be in (0, 1)")

	group_cols = _selection_key_columns(df)
	out = df.copy()
	out["selected"] = out.groupby(group_cols, dropna=False)["coef"].transform(
		lambda s: (_coef_importance(s, use_sign=use_sign) >= _coef_importance(s, use_sign=use_sign).quantile(float(percentile))).astype(int)
	)
	return out


def _add_rank_percentile(df: pd.DataFrame, use_sign: bool = False) -> pd.DataFrame:
	group_cols = _selection_key_columns(df)
	out = df.copy()
	out["rank_percentile"] = out.groupby(group_cols, dropna=False)["coef"].transform(
		lambda s: _coef_importance(s, use_sign=use_sign).rank(method="average", pct=True)
	)
	return out


def _add_score_weight(df: pd.DataFrame, score_col: str = "test_score", weight_col: str = "score_weight") -> pd.DataFrame:
	out = df.copy()
	weights = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0).astype(float)
	weights = weights.clip(lower=0.0)
	if float(weights.sum()) <= 0.0:
		weights = pd.Series(np.ones(len(out), dtype=float), index=out.index)
	out[weight_col] = weights
	return out


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
	v = pd.to_numeric(values, errors="coerce").astype(float)
	w = pd.to_numeric(weights, errors="coerce").astype(float)
	mask = v.notna() & w.notna()
	if int(mask.sum()) == 0:
		return 0.0
	vv = v.loc[mask].to_numpy()
	ww = w.loc[mask].to_numpy()
	ww = np.clip(ww, 0.0, None)
	if float(ww.sum()) <= 0.0:
		ww = np.ones_like(ww, dtype=float)
	return float(np.average(vv, weights=ww))


def _weighted_variance(values: pd.Series, weights: pd.Series) -> float:
	v = pd.to_numeric(values, errors="coerce").astype(float)
	w = pd.to_numeric(weights, errors="coerce").astype(float)
	mask = v.notna() & w.notna()
	if int(mask.sum()) == 0:
		return 0.0
	vv = v.loc[mask].to_numpy()
	ww = w.loc[mask].to_numpy()
	ww = np.clip(ww, 0.0, None)
	if float(ww.sum()) <= 0.0:
		ww = np.ones_like(ww, dtype=float)
	mean = float(np.average(vv, weights=ww))
	return float(np.average((vv - mean) ** 2, weights=ww))


def _weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
	v = pd.to_numeric(values, errors="coerce").astype(float)
	w = pd.to_numeric(weights, errors="coerce").astype(float)
	mask = v.notna() & w.notna()
	if int(mask.sum()) == 0:
		return 0.0
	vv = v.loc[mask].to_numpy()
	ww = w.loc[mask].to_numpy()
	ww = np.clip(ww, 0.0, None)
	if float(ww.sum()) <= 0.0:
		ww = np.ones_like(ww, dtype=float)
	order = np.argsort(vv)
	vv_sorted = vv[order]
	ww_sorted = ww[order]
	cum_w = np.cumsum(ww_sorted)
	cutoff = float(q) * float(ww_sorted.sum())
	idx = int(np.searchsorted(cum_w, cutoff, side="left"))
	idx = min(max(idx, 0), len(vv_sorted) - 1)
	return float(vv_sorted[idx])


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

	signs = selected["coef"].apply(lambda x: 1 if x > 0 else -1)
	return pd.Series(
		{
			"selection_freq": float(group["selected"].mean()),
			"sign_consistency": float(signs.value_counts(normalize=True).max()),
			"effect_median": float(selected["coef"].median()),
			"effect_iqr": float(
				selected["coef"].quantile(0.75) - selected["coef"].quantile(0.25)
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


def _plot_dataset_task_maps(
	metrics_main: pd.DataFrame,
	normalized_df: pd.DataFrame,
	maps_dir: Path,
	use_sign: bool = False,
) -> None:
	if normalized_df.empty:
		return

	def _selection_stats(group: pd.DataFrame) -> pd.Series:
		mean_val = _weighted_mean(group["selected"], group["score_weight"])
		var_val = _weighted_variance(group["selected"], group["score_weight"])
		return pd.Series(
			{
				"selection_freq_mean": mean_val,
				"selection_freq_std": float(np.sqrt(max(var_val, 0.0))),
			}
		)

	grouped = normalized_df.groupby(["dataset", "task", "region"], dropna=False, as_index=False)
	try:
		selection_stats = grouped.apply(_selection_stats, include_groups=False).reset_index(drop=True)
	except TypeError:
		selection_stats = grouped.apply(_selection_stats).reset_index(drop=True)
	if selection_stats.empty:
		return

	for (dataset, task), combo in selection_stats.groupby(["dataset", "task"], dropna=False):
		combo_metrics = metrics_main[
			(metrics_main["dataset"].astype(str) == str(dataset))
			& (metrics_main["task"].astype(str) == str(task))
		].copy()

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

		mean_label_values = _label_value_map(combo.set_index("region")["selection_freq_mean"])
		std_label_values = _label_value_map(combo.set_index("region")["selection_freq_std"].fillna(0.0))
		sign_label_values: dict[int, float] = {}
		if use_sign and not combo_metrics.empty:
			sign_label_values = _label_value_map(combo_metrics.set_index("region")["sign_consistency"].fillna(0.0))
		if not mean_label_values and not std_label_values and not sign_label_values:
			warnings.warn(f"Skipping brain map for {dataset}/{task}: no plottable numeric region labels")
			continue

		tag = f"{str(dataset).lower()}_{str(task).lower().replace(' ', '_')}"

		mean_vals = combo["selection_freq_mean"].astype(float)
		mean_vmin = 0.0
		mean_vmax = float(mean_vals.max()) if not mean_vals.empty else 0.0
		if mean_vmax <= 0.0:
			mean_vmax = 1e-12

		std_vals = combo["selection_freq_std"].fillna(0.0).astype(float)
		std_vmin = 0.0
		std_vmax = float(std_vals.max()) if not std_vals.empty else 0.0
		if std_vmax <= 0.0:
			std_vmax = 1e-12

		sign_vmin, sign_vmax = 0.0, 1.0

		atlas_type = atlas_info.get("atlas_type", "surface_schaefer")
		if atlas_type != "surface_schaefer":
			warnings.warn(
				f"Atlas type {atlas_type} for {dataset}/{task} is not handled in this script; skipping map"
			)
			continue

		try:
			if mean_label_values:
				_plot_surface_metric(
					surface_atlas=atlas_info,
					label_values=mean_label_values,
					title=f"Selection Frequency Mean | {dataset} | {task}",
					out_file=maps_dir / f"selection_frequency_{tag}.png",
					vmin=mean_vmin,
					vmax=mean_vmax,
					symmetric=False,
					cmap="Reds",
				)
			if std_label_values:
				_plot_surface_metric(
					surface_atlas=atlas_info,
					label_values=std_label_values,
					title=f"Selection Frequency Std | {dataset} | {task}",
					out_file=maps_dir / f"selection_frequency_std_{tag}.png",
					vmin=std_vmin,
					vmax=std_vmax,
					symmetric=False,
					cmap="Reds",
				)
			if sign_label_values:
				_plot_surface_metric(
					surface_atlas=atlas_info,
					label_values=sign_label_values,
					title=f"Sign Consistency | {dataset} | {task}",
					out_file=maps_dir / f"sign_consistency_{tag}.png",
					vmin=sign_vmin,
					vmax=sign_vmax,
					symmetric=False,
					cmap="Reds",
				)
		except Exception as exc:
			warnings.warn(f"Failed plotting brain map for {dataset}/{task}: {exc}")


def _plot_rank_maps(rank_metrics: pd.DataFrame, normalized_df: pd.DataFrame, maps_dir: Path) -> None:
	if rank_metrics.empty or normalized_df.empty:
		return

	for (dataset, task), combo in rank_metrics.groupby(["dataset", "task"], dropna=False):
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
			warnings.warn(f"Skipping rank map for {dataset}/{task}: could not load atlas ({exc})")
			continue

		rank_mean_values = _label_value_map(combo.set_index("region")["rank_mean"])
		rank_median_values = _label_value_map(combo.set_index("region")["rank_median"])
		rank_var_values = _label_value_map(combo.set_index("region")["rank_variance"].fillna(0.0))
		if not rank_mean_values and not rank_median_values and not rank_var_values:
			warnings.warn(f"Skipping rank map for {dataset}/{task}: no plottable numeric region labels")
			continue

		tag = f"{str(dataset).lower()}_{str(task).lower().replace(' ', '_')}"

		rank_mean_vmin, rank_mean_vmax = 0.0, 1.0
		rank_median_vmin, rank_median_vmax = 0.0, 1.0

		var_vals = combo["rank_variance"].fillna(0.0).astype(float)
		rank_var_vmin = 0.0
		rank_var_vmax = float(var_vals.max()) if not var_vals.empty else 0.0
		if rank_var_vmax <= 0.0:
			rank_var_vmax = 1e-12

		atlas_type = atlas_info.get("atlas_type", "surface_schaefer")
		if atlas_type != "surface_schaefer":
			warnings.warn(
				f"Atlas type {atlas_type} for {dataset}/{task} is not handled in this script; skipping rank map"
			)
			continue

		try:
			if rank_mean_values:
				_plot_surface_metric(
					surface_atlas=atlas_info,
					label_values=rank_mean_values,
					title=f"Rank Mean | {dataset} | {task}",
					out_file=maps_dir / f"rank_mean_{tag}.png",
					vmin=rank_mean_vmin,
					vmax=rank_mean_vmax,
					symmetric=False,
					cmap="Reds",
				)
			if rank_median_values:
				_plot_surface_metric(
					surface_atlas=atlas_info,
					label_values=rank_median_values,
					title=f"Rank Median | {dataset} | {task}",
					out_file=maps_dir / f"rank_median_{tag}.png",
					vmin=rank_median_vmin,
					vmax=rank_median_vmax,
					symmetric=False,
					cmap="Reds",
				)
			if rank_var_values:
				_plot_surface_metric(
					surface_atlas=atlas_info,
					label_values=rank_var_values,
					title=f"Rank Variance | {dataset} | {task}",
					out_file=maps_dir / f"rank_variance_{tag}.png",
					vmin=0, #rank_var_vmin,
					vmax=0.25, #rank_var_vmax,
					symmetric=False,
					cmap="Reds",
				)
		except Exception as exc:
			warnings.warn(f"Failed plotting rank map for {dataset}/{task}: {exc}")


def build_selected_normalized_coefficients(
	input_table: Path = INPUT_TABLE,
	best_runs_table: Path = BEST_RUNS_TABLE,
	output_table: Path = OUTPUT_TABLE,
	stability_table: Path = STABILITY_TABLE,
	rank_table: Path = RANK_TABLE,
	maps_dir: Path = MAPS_DIR,
	selection_percentile: float = 0.90,
	coefficient_mode: str = "absolute",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if not (0.90 <= float(selection_percentile) <= 0.95):
		raise ValueError("selection_percentile must be between 0.90 and 0.95")
	mode_name, use_sign = _coefficient_mode_settings(coefficient_mode)
	effective_maps_dir = maps_dir
	if use_sign:
		effective_maps_dir = maps_dir.parent / f"{maps_dir.name}_signs"

	df = pd.read_parquet(input_table)
	required = {"run_id", "coef", "test_score"}
	missing = required - set(df.columns)
	if missing:
		missing_txt = ", ".join(sorted(missing))
		raise ValueError(f"Missing required columns in input table: {missing_txt}")
	df = df.copy()
	df["exp_id"] = _build_exp_id(df)

	group_cols = _pick_group_columns(df)
	if not group_cols:
		raise ValueError(
			"No grouping columns found. Need at least one of: "
			"microstructure, model_type/model, embedding, task, dataset"
		)

	best_runs = _select_best_runs(df, group_cols)
	selected = df[df["run_id"].isin(best_runs["run_id"])].copy()
	if selected.empty:
		raise ValueError("No rows left after run selection; check score distributions and selection thresholds")
	normalized = _add_percentile_selection(selected, percentile=selection_percentile, use_sign=use_sign)
	normalized = _add_rank_percentile(normalized, use_sign=use_sign)
	normalized = _add_score_weight(normalized)
	normalized["coefficient_mode"] = mode_name
	metrics_main = (
		normalized.groupby(["dataset", "task", "region"], dropna=False)[["selected", "coef"]]
		.apply(_compute_metrics)
		.reset_index()
	)
	metrics_main["coefficient_mode"] = mode_name

	def _rank_stats(group: pd.DataFrame) -> pd.Series:
		return pd.Series(
			{
				"rank_mean": _weighted_mean(group["rank_percentile"], group["score_weight"]),
				"rank_median": _weighted_quantile(group["rank_percentile"], group["score_weight"], 0.5),
				"rank_variance": _weighted_variance(group["rank_percentile"], group["score_weight"]),
			}
		)

	grouped_rank = normalized.groupby(["dataset", "task", "region"], dropna=False, as_index=False)
	try:
		rank_metrics = grouped_rank.apply(_rank_stats, include_groups=False).reset_index(drop=True)
	except TypeError:
		rank_metrics = grouped_rank.apply(_rank_stats).reset_index(drop=True)
	rank_metrics["coefficient_mode"] = mode_name

	best_runs_table.parent.mkdir(parents=True, exist_ok=True)
	output_table.parent.mkdir(parents=True, exist_ok=True)
	stability_table.parent.mkdir(parents=True, exist_ok=True)
	rank_table.parent.mkdir(parents=True, exist_ok=True)
	best_runs.to_parquet(best_runs_table, index=False)
	normalized.to_parquet(output_table, index=False)
	metrics_main.to_parquet(stability_table, index=False)
	rank_metrics.to_parquet(rank_table, index=False)
	_plot_dataset_task_maps(metrics_main, normalized, effective_maps_dir, use_sign=use_sign)
	_plot_rank_maps(rank_metrics, normalized, effective_maps_dir)

	return best_runs, normalized, metrics_main, rank_metrics


if __name__ == "__main__":
	best_df, norm_df, metrics_df, rank_df = build_selected_normalized_coefficients()
	# best_df, norm_df, metrics_df, rank_df = build_selected_normalized_coefficients(coefficient_mode="signed")
	print(f"Saved best runs: {len(best_df)} rows -> {BEST_RUNS_TABLE}")
	print(f"Saved normalized coefficients: {len(norm_df)} rows -> {OUTPUT_TABLE}")
	print(f"Saved stability metrics: {len(metrics_df)} rows -> {STABILITY_TABLE}")
	print(f"Saved rank metrics: {len(rank_df)} rows -> {RANK_TABLE}")
	print(f"Saved brain maps under: {MAPS_DIR}")
