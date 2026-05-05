from __future__ import annotations

from pathlib import Path
import ast

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diff_benchmark.analysis.region_coefficients import load_atlas_from_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "coefficients_long.parquet"
OUTPUT_DIR = PROJECT_ROOT / "exp_outputs" / "summary" / "MD_embedding_correlation_maps"

REGION_REPRESENTATIONS = ["flatten", "mean_std", "summary_stats", "percentiles", "pca"]
PERCENTILE = 0.90
MICROSTRUCTURE_SELECTION = "md"


def _build_exp_id(df: pd.DataFrame) -> pd.Series:
	cols = ["run_id"]
	if "fold" in df.columns:
		cols.append("fold")
	if "seed" in df.columns:
		cols.append("seed")
	return df[cols].astype(str).agg("_".join, axis=1)


def _pick_group_columns(df: pd.DataFrame) -> list[str]:
	candidates = ["task", "dataset"]
	if "model_type" not in df.columns and "model" in df.columns:
		df["model_type"] = df["model"]
	return [c for c in candidates if c in df.columns]


def _safe_parse_embedding(s: str) -> dict:
	try:
		return ast.literal_eval(s)
	except Exception:
		return {}


def _extract_model_embedding(row: pd.Series) -> tuple[str, str]:
	model = row.get("model_type", row.get("model", "unknown"))
	emb_raw = row.get("embedding", "")
	emb_dict = _safe_parse_embedding(emb_raw)

	if "region_encoder" in emb_dict:
		encoder = emb_dict.get("region_encoder", {})
		enc_type = encoder.get("type", "unknown")
		include_size = encoder.get("include_size", False)
		if include_size:
			embedding = "pointnet_size"
		else:
			embedding = f"pointnet_{enc_type}"
		model = "region_group_lasso"
		return model, embedding

	if model == "region_pca":
		return "region_group_lasso", "pca"

	if "region_representation" in emb_dict:
		embedding = emb_dict["region_representation"]
	else:
		embedding = "unknown"

	return model, embedding


def _add_model_embedding_columns(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	parsed = out.apply(_extract_model_embedding, axis=1)
	out["model_name"] = [p[0] for p in parsed]
	out["embedding_name"] = [p[1] for p in parsed]
	return out


def _exclude_region_permutation_models(df: pd.DataFrame) -> pd.DataFrame:
	cols = [c for c in ["model", "model_type", "model_name"] if c in df.columns]
	if not cols:
		return df
	mask = np.ones(len(df), dtype=bool)
	for col in cols:
		mask &= ~df[col].astype(str).str.contains("region_permutation", case=False, na=False)
	return df.loc[mask].copy()


def _selection_key_columns(df: pd.DataFrame) -> list[str]:
	keys = [c for c in ["run_id", "fold", "seed"] if c in df.columns]
	if keys:
		return keys
	if "exp_id" in df.columns:
		return ["exp_id"]
	raise ValueError("Cannot build selection groups")


def _add_percentile_selection(df: pd.DataFrame, percentile: float = 0.90) -> pd.DataFrame:
	group_cols = _selection_key_columns(df)
	out = df.copy()
	out["selected"] = out.groupby(group_cols)["coef"].transform(
		lambda s: (s.abs() >= s.abs().quantile(percentile)).astype(int)
	)
	return out


def _select_best_experiments(df: pd.DataFrame, group_cols: list[str], outlier_iqr_mult: float = 1.5) -> pd.DataFrame:
	score_df = (
		df[["exp_id", "test_score"] + group_cols]
		.dropna(subset=["test_score"])
		.drop_duplicates()
	)

	exp_scores = score_df.rename(columns={"test_score": "score"})
	if exp_scores.empty:
		return exp_scores

	def _filter_group(group: pd.DataFrame) -> pd.DataFrame:
		out = group.copy()
		scores = out["score"].astype(float)
		best_score = float(scores.max())
		gaps = best_score - scores

		q1_gap = float(gaps.quantile(0.25))
		q3_gap = float(gaps.quantile(0.75))
		iqr_gap = q3_gap - q1_gap
		gap_cutoff = q3_gap + outlier_iqr_mult * iqr_gap if iqr_gap > 0 else q3_gap

		q1_score = float(scores.quantile(0.25))
		q3_score = float(scores.quantile(0.75))
		iqr_score = q3_score - q1_score
		score_floor = q1_score

		keep_mask = (gaps <= gap_cutoff) & (scores >= score_floor)
		keep_mask = keep_mask | (scores == best_score)

		if keep_mask.sum() == 0:
			keep_mask = scores == best_score

		out["best_test_score"] = best_score
		out["score_gap_from_best"] = gaps
		return out.loc[keep_mask.values]

	grouped = exp_scores.groupby(group_cols, dropna=False, as_index=False)
	try:
		selected = grouped.apply(_filter_group, include_groups=False).reset_index(drop=True)
	except TypeError:
		selected = grouped.apply(_filter_group).reset_index(drop=True)
	return selected


def _selection_stats_binomial(group: pd.DataFrame) -> pd.Series:
	s = group["selected"].astype(int)
	N = len(s)
	if N == 0:
		return pd.Series({"selection_freq": 0.0, "n_runs": 0})
	p = float(s.mean())
	return pd.Series({"selection_freq": p, "n_runs": N})


def _compute_selection_maps(df: pd.DataFrame, extra_group_cols: list[str]) -> pd.DataFrame:
	group_cols = ["dataset", "task"] + extra_group_cols + ["region"]
	return df.groupby(group_cols).apply(_selection_stats_binomial).reset_index()


def _filter_microstructure(df: pd.DataFrame) -> pd.DataFrame:
	if MICROSTRUCTURE_SELECTION is None:
		return df
	if "microstructure" not in df.columns:
		return df
	return df[df["microstructure"].astype(str) == str(MICROSTRUCTURE_SELECTION)].copy()


def _label_value_map(values_by_region: pd.Series) -> dict[int, float]:
	out: dict[int, float] = {}
	for region, value in values_by_region.items():
		try:
			label = int(str(region).split(":")[-1])
			out[label] = float(value)
		except Exception:
			continue
	return out


def _surface_texture_from_label_map(surface_atlas, label_values):
	labels = np.asarray(surface_atlas["parcel_labels"]).astype(int)
	n_left = int(surface_atlas["n_left_vertices"])

	texture = np.zeros(len(labels))
	for label, value in label_values.items():
		texture[labels == label] = value

	return texture[:n_left], texture[n_left:]


def _plot_surface_row(surface_atlas, label_values, ax_left, ax_right, vmin, vmax, title: str, cmap: str) -> None:
	from nilearn import plotting

	left_mesh = surface_atlas["left_mesh"]
	right_mesh = surface_atlas["right_mesh"]
	tex_left, tex_right = _surface_texture_from_label_map(surface_atlas, label_values)

	plotting.plot_surf_stat_map(
		left_mesh,
		tex_left,
		hemi="left",
		axes=ax_left,
		vmin=vmin,
		vmax=vmax,
		cmap=cmap,
		colorbar=False,
	)
	plotting.plot_surf_stat_map(
		right_mesh,
		tex_right,
		hemi="right",
		axes=ax_right,
		vmin=vmin,
		vmax=vmax,
		cmap=cmap,
		colorbar=False,
	)
	if title:
		ax_left.set_title(title, fontsize=10)


def _pearson_corr(a: pd.Series, b: pd.Series) -> float:
	if a.empty or b.empty:
		return float("nan")
	joined = pd.concat([a, b], axis=1, join="inner").dropna()
	if len(joined) < 2:
		return float("nan")
	if float(joined.iloc[:, 0].std()) == 0.0 or float(joined.iloc[:, 1].std()) == 0.0:
		return float("nan")
	return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def _align_series(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
	joined = pd.concat([a, b], axis=1, join="inner").dropna()
	if joined.empty:
		return a.iloc[0:0], b.iloc[0:0]
	return joined.iloc[:, 0], joined.iloc[:, 1]


def _correlation_matrix(series_by_key: dict[str, pd.Series]) -> pd.DataFrame:
	keys = list(series_by_key.keys())
	matrix = pd.DataFrame(index=keys, columns=keys, dtype=float)
	for i in keys:
		for j in keys:
			matrix.loc[i, j] = _pearson_corr(series_by_key[i], series_by_key[j])
	return matrix


def _build_embedding_matrix(
	stats_embedding: pd.DataFrame,
	dataset: str,
	task: str,
	embeddings: list[str],
) -> pd.DataFrame:
	filtered = stats_embedding[
		(stats_embedding["dataset"] == dataset)
		& (stats_embedding["task"] == task)
		& (stats_embedding["embedding_name"].isin(embeddings))
	]
	if filtered.empty:
		return pd.DataFrame(index=pd.Index([], name="region"), columns=embeddings)

	df = (
		filtered.pivot_table(
			index="region",
			columns="embedding_name",
			values="selection_freq",
			aggfunc="mean",
		)
		.reindex(columns=embeddings)
		.fillna(0.0)
	)
	return df


def _normalize_per_region(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return df
	mean = df.mean(axis=1)
	std = df.std(axis=1) + 1e-8
	sum = df.sum(axis=1) + 1e-8
	# return df.sub(mean, axis=0).div(std, axis=0)
	return df.div(sum, axis=0)


def _correlation_matrices(df_norm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	pearson = df_norm.corr(method="pearson") if not df_norm.empty else pd.DataFrame()
	rank_df = df_norm.rank(axis=0)
	spearman = rank_df.corr(method="pearson") if not rank_df.empty else pd.DataFrame()
	return pearson, spearman


def _save_single_map(surface_atlas, values: dict[int, float], title: str, out_file: Path, vmin: float, vmax: float, cmap: str) -> None:
	fig = plt.figure(figsize=(10, 4))
	gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.02)
	ax_left = fig.add_subplot(gs[0, 0], projection="3d")
	ax_right = fig.add_subplot(gs[0, 1], projection="3d")
	cax = fig.add_subplot(gs[0, 2])
	_plot_surface_row(surface_atlas, values, ax_left, ax_right, vmin, vmax, title, cmap)
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	sm.set_array([])
	fig.colorbar(sm, cax=cax)
	fig.tight_layout()
	fig.savefig(out_file, dpi=150)
	plt.close(fig)


def main() -> None:
	df = pd.read_parquet(INPUT_TABLE)
	df = _exclude_region_permutation_models(df)
	df["exp_id"] = _build_exp_id(df)

	df = _add_model_embedding_columns(df)
	group_cols = _pick_group_columns(df)
	best_runs = _select_best_experiments(df, group_cols)
	df = df[df["exp_id"].isin(best_runs["exp_id"])].copy()

	df = _add_percentile_selection(df, percentile=PERCENTILE)
	df = _filter_microstructure(df)

	stats_global = _compute_selection_maps(df, [])
	stats_embedding = _compute_selection_maps(df, ["embedding_name"])

	for (dataset, task), global_map in stats_global.groupby(["dataset", "task"], dropna=False):
		run_id = df[(df["dataset"] == dataset) & (df["task"] == task)]["run_id"].iloc[0]
		surface_atlas = load_atlas_from_run(run_id)

		global_series = global_map.set_index("region")["selection_freq"]
		embedding_df = _build_embedding_matrix(stats_embedding, dataset, task, REGION_REPRESENTATIONS)
		if embedding_df.empty:
			continue

		embedding_df = embedding_df.reindex(embedding_df.index.union(global_series.index)).fillna(0.0)
		global_series = global_series.reindex(embedding_df.index).fillna(0.0)
		breakpoint()
		norm_df = _normalize_per_region(embedding_df)
		mean_by_region = embedding_df.mean(axis=1)
		std_by_region = embedding_df.std(axis=1) + 1e-8
		global_norm = (global_series - mean_by_region) / std_by_region
		rows = ["global"] + REGION_REPRESENTATIONS
		series_by_key_raw: dict[str, pd.Series] = {"global": global_series}
		series_by_key_norm: dict[str, pd.Series] = {"global": global_norm}
		n_rows = len(rows)
		fig = plt.figure(figsize=(12, 4 * n_rows))
		gs = fig.add_gridspec(n_rows, 3, width_ratios=[1, 1, 0.06], wspace=0.02, hspace=0.12)
		axes = []
		for row_idx in range(n_rows):
			ax_left = fig.add_subplot(gs[row_idx, 0], projection="3d")
			ax_right = fig.add_subplot(gs[row_idx, 1], projection="3d")
			axes.append((ax_left, ax_right))
		cax = fig.add_subplot(gs[:, 2])
		if n_rows == 1:
			axes = [axes[0]]

		for idx, emb in enumerate(rows):
			ax_left, ax_right = axes[idx]
			if emb == "global":
				values = _label_value_map(global_series)
				title = f"global | corr=1.00"
			else:
				if emb not in embedding_df.columns:
					ax_left.set_axis_off()
					ax_right.set_axis_off()
					continue
				series_raw = embedding_df[emb]
				series_norm = norm_df[emb]
				series_by_key_raw[emb] = series_raw
				series_by_key_norm[emb] = series_norm
				corr_val = _pearson_corr(series_norm, global_norm)
				values = _label_value_map(series_raw)
				title = f"{emb} | corr={corr_val:.2f}"

			_plot_surface_row(surface_atlas, values, ax_left, ax_right, 0, 1, title, "Reds")

		sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=0, vmax=1))
		sm.set_array([])
		fig.colorbar(sm, cax=cax)

		fig.suptitle(f"Embedding correlation maps | {dataset} | {task}")
		fig.subplots_adjust(left=0.03, right=0.94, top=0.94, bottom=0.03)
		OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
		fig.savefig(OUTPUT_DIR / f"embedding_corr_{dataset}_{task}.png", dpi=150)
		plt.close(fig)

		pearson_mat, spearman_mat = _correlation_matrices(norm_df)
		pearson_mat.to_csv(OUTPUT_DIR / f"embedding_corr_matrix_pearson_{dataset}_{task}.csv")
		spearman_mat.to_csv(OUTPUT_DIR / f"embedding_corr_matrix_spearman_{dataset}_{task}.csv")

		if not pearson_mat.empty:
			plt.figure(figsize=(6, 5))
			plt.imshow(pearson_mat.values.astype(float), vmin=-1, vmax=1, cmap="coolwarm")
			plt.colorbar(label="Pearson r")
			plt.xticks(range(len(pearson_mat.columns)), pearson_mat.columns, rotation=45, ha="right")
			plt.yticks(range(len(pearson_mat.index)), pearson_mat.index)
			plt.title(f"Embedding correlations (Pearson) | {dataset} | {task}")
			plt.tight_layout()
			plt.savefig(OUTPUT_DIR / f"embedding_corr_matrix_pearson_{dataset}_{task}.png", dpi=150)
			plt.close()

		if not spearman_mat.empty:
			plt.figure(figsize=(6, 5))
			plt.imshow(spearman_mat.values.astype(float), vmin=-1, vmax=1, cmap="coolwarm")
			plt.colorbar(label="Spearman r")
			plt.xticks(range(len(spearman_mat.columns)), spearman_mat.columns, rotation=45, ha="right")
			plt.yticks(range(len(spearman_mat.index)), spearman_mat.index)
			plt.title(f"Embedding correlations (Spearman) | {dataset} | {task}")
			plt.tight_layout()
			plt.savefig(OUTPUT_DIR / f"embedding_corr_matrix_spearman_{dataset}_{task}.png", dpi=150)
			plt.close()

		region_agreement_df = pd.DataFrame({
			"mean_freq": embedding_df.mean(axis=1),
			"std_freq": embedding_df.std(axis=1),
		})
		region_agreement_df["cv"] = region_agreement_df["std_freq"] / (region_agreement_df["mean_freq"] + 1e-8)

		consensus_values = _label_value_map(region_agreement_df["mean_freq"])
		disagreement_values = _label_value_map(region_agreement_df["std_freq"])
		cv_values = _label_value_map(region_agreement_df["cv"])

		_save_single_map(
			surface_atlas,
			consensus_values,
			f"Embedding consensus | {dataset} | {task}",
			OUTPUT_DIR / f"embedding_consensus_map_{dataset}_{task}.png",
			vmin=0,
			vmax=float(region_agreement_df["mean_freq"].max()) if not region_agreement_df.empty else 1.0,
			cmap="Reds",
		)
		_save_single_map(
			surface_atlas,
			disagreement_values,
			f"Embedding disagreement | {dataset} | {task}",
			OUTPUT_DIR / f"embedding_disagreement_map_{dataset}_{task}.png",
			vmin=0,
			vmax=float(region_agreement_df["std_freq"].max()) if not region_agreement_df.empty else 1.0,
			cmap="Blues",
		)
		_save_single_map(
			surface_atlas,
			cv_values,
			f"Embedding disagreement (CV) | {dataset} | {task}",
			OUTPUT_DIR / f"embedding_disagreement_cv_map_{dataset}_{task}.png",
			vmin=0,
			vmax=float(region_agreement_df["cv"].max()) if not region_agreement_df.empty else 1.0,
			cmap="coolwarm",
		)

		# Grid of brain maps: rows=map, columns=reference map (each ref has left/right)
		keys = list(series_by_key_norm.keys())
		max_abs = 0.0
		for row_key in keys:
			for col_key in keys:
				left, right = _align_series(series_by_key_norm[row_key], series_by_key_norm[col_key])
				if left.empty or right.empty:
					continue
				max_abs = max(max_abs, float(np.abs((left - right).to_numpy()).max()))
		if max_abs <= 0.0:
			max_abs = 1e-6
		max_abs_abs = max_abs
		n_rows = len(keys)
		n_cols = len(keys) * 2
		fig = plt.figure(figsize=(2.2 * n_cols + 1.4, 2.4 * n_rows))
		gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.06], wspace=0.02, hspace=0.12)
		for row_idx, row_key in enumerate(keys):
			row_series = series_by_key_norm[row_key]
			for col_idx, ref_key in enumerate(keys):
				ref_series = series_by_key_norm[ref_key]
				corr_val = _pearson_corr(row_series, ref_series)
				left, right = _align_series(row_series, ref_series)
				if left.empty or right.empty:
					ax_left = fig.add_subplot(gs[row_idx, col_idx * 2], projection="3d")
					ax_right = fig.add_subplot(gs[row_idx, col_idx * 2 + 1], projection="3d")
					ax_left.set_axis_off()
					ax_right.set_axis_off()
					continue
				diff_series = left - right
				row_values = _label_value_map(diff_series)
				ax_left = fig.add_subplot(gs[row_idx, col_idx * 2], projection="3d")
				ax_right = fig.add_subplot(gs[row_idx, col_idx * 2 + 1], projection="3d")
				title = f"{row_key} - {ref_key} | r={corr_val:.2f}"
				_plot_surface_row(surface_atlas, row_values, ax_left, ax_right, -max_abs, max_abs, title, "coolwarm")

		cax = fig.add_subplot(gs[:, -1])
		sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-max_abs, vmax=max_abs))
		sm.set_array([])
		fig.colorbar(sm, cax=cax)

		fig.suptitle(f"Embedding map comparisons | {dataset} | {task}")
		fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.02)
		fig.savefig(OUTPUT_DIR / f"embedding_corr_grid_{dataset}_{task}.png", dpi=150)
		plt.close(fig)

		fig = plt.figure(figsize=(2.2 * n_cols + 1.4, 2.4 * n_rows))
		gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.06], wspace=0.02, hspace=0.12)
		for row_idx, row_key in enumerate(keys):
			row_series = series_by_key_norm[row_key]
			for col_idx, ref_key in enumerate(keys):
				ref_series = series_by_key_norm[ref_key]
				corr_val = _pearson_corr(row_series, ref_series)
				left, right = _align_series(row_series, ref_series)
				if left.empty or right.empty:
					ax_left = fig.add_subplot(gs[row_idx, col_idx * 2], projection="3d")
					ax_right = fig.add_subplot(gs[row_idx, col_idx * 2 + 1], projection="3d")
					ax_left.set_axis_off()
					ax_right.set_axis_off()
					continue
				abs_diff_series = (left - right).abs()
				row_values = _label_value_map(abs_diff_series)
				ax_left = fig.add_subplot(gs[row_idx, col_idx * 2], projection="3d")
				ax_right = fig.add_subplot(gs[row_idx, col_idx * 2 + 1], projection="3d")
				title = f"|{row_key} - {ref_key}| | r={corr_val:.2f}"
				_plot_surface_row(surface_atlas, row_values, ax_left, ax_right, 0, max_abs_abs, title, "magma")

		cax = fig.add_subplot(gs[:, -1])
		sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(vmin=0, vmax=max_abs_abs))
		sm.set_array([])
		fig.colorbar(sm, cax=cax)
		fig.suptitle(f"Embedding map comparisons (abs diff) | {dataset} | {task}")
		fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.02)
		fig.savefig(OUTPUT_DIR / f"embedding_corr_grid_abs_{dataset}_{task}.png", dpi=150)
		plt.close(fig)


if __name__ == "__main__":
	main()
