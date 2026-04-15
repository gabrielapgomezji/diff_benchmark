"""
variance_decoomposition.py
--------------------------
Variance decomposition of prediction scores per dataset-task combination.

For each requested (dataset, target, prediction_task):
  1) Fit linear model: score ~ microstructure + embedding + model
  2) Compute percent variance explained by each factor (ANOVA sum of squares)
  3) Plot bar chart per panel

Optional: include pairwise interaction terms.
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

BASE_FACTORS = ["microstructure", "embedding", "model"]


def apply_miccai_style() -> None:
	"""Apply publication-style defaults aligned with other project plots."""
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


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	out["target_clean"] = out["target"].map(_clean_target)
	out["microstructure"] = out["primary_metric"].astype(str)
	out["embedding"] = out["config.model.backbone.region_representation"].fillna("none")
	out["model"] = out["model_name"].astype(str)
	return out


def _choose_score_column(df_task: pd.DataFrame, prediction_task: str) -> tuple[pd.DataFrame, str]:
	"""Choose score column and return a dataframe ready for model fitting."""
	if prediction_task == "binary_classification":
		if "accuracy_weighted_test_mean" in df_task.columns:
			return df_task, "accuracy_weighted_test_mean"
		if "accuracy_test_mean" in df_task.columns:
			return df_task, "accuracy_test_mean"
		raise ValueError("No classification score column found")

	if "r2_test_mean" in df_task.columns and df_task["r2_test_mean"].notna().any():
		return df_task, "r2_test_mean"

	if "mae_test_mean" in df_task.columns and df_task["mae_test_mean"].notna().any():
		warnings.warn("Using negative MAE because r2_test_mean is unavailable")
		tmp = df_task.copy()
		tmp["_neg_mae"] = -tmp["mae_test_mean"]
		return tmp, "_neg_mae"

	raise ValueError("No regression score column found")


def _factor_display_name(factor: str) -> str:
	mapping = {
		"microstructure": "Microstructure",
		"embedding": "Embedding",
		"model": "Model",
		"microstructure:embedding": "Microstructure x Embedding",
		"microstructure:model": "Microstructure x Model",
		"embedding:model": "Embedding x Model",
	}
	return mapping.get(factor, factor)


def _dummy_block(df: pd.DataFrame, col: str) -> np.ndarray:
	"""Return a one-hot design block for a categorical factor."""
	d = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=False)
	arr = d.to_numpy(dtype=float)
	if arr.ndim == 1:
		arr = arr.reshape(-1, 1)
	return arr


def _interaction_block(df: pd.DataFrame, left: str, right: str) -> np.ndarray:
	"""Return a one-hot design block for pairwise categorical interaction."""
	interaction = df[left].astype(str) + "::" + df[right].astype(str)
	d = pd.get_dummies(interaction, prefix=f"{left}x{right}", drop_first=False)
	arr = d.to_numpy(dtype=float)
	if arr.ndim == 1:
		arr = arr.reshape(-1, 1)
	return arr


def _sse_from_design(y: np.ndarray, X: np.ndarray) -> float:
	"""Fit OLS via least squares and return residual sum of squares."""
	if X.size == 0:
		y_hat = np.repeat(np.mean(y), len(y))
		resid = y - y_hat
		return float(np.sum(resid * resid))
	beta, *_ = np.linalg.lstsq(X, y, rcond=None)
	resid = y - X @ beta
	return float(np.sum(resid * resid))


def _fit_anova_explained_percent(
	df_task: pd.DataFrame,
	score_col: str,
	include_interactions: bool,
) -> pd.DataFrame:
	"""Fit OLS and return percent variance explained per factor block.

	Contribution is computed as reduction in SSE when adding a factor block to
	the full model context (full-vs-reduced), normalized by total sum of squares.
	"""
	work = df_task[["microstructure", "embedding", "model", score_col]].copy()
	work = work.rename(columns={score_col: "score"})
	work = work.dropna(subset=["microstructure", "embedding", "model", "score"]).copy()

	for c in ["microstructure", "embedding", "model"]:
		work[c] = work[c].astype(str)

	if work.empty:
		return pd.DataFrame(columns=["factor", "pct_explained"])

	y = work["score"].to_numpy(dtype=float)
	total_ss = float(np.sum((y - np.mean(y)) ** 2))
	if total_ss <= 0 or np.isnan(total_ss):
		return pd.DataFrame(columns=["factor", "pct_explained"])

	blocks: dict[str, np.ndarray] = {
		"microstructure": _dummy_block(work, "microstructure"),
		"embedding": _dummy_block(work, "embedding"),
		"model": _dummy_block(work, "model"),
	}
	if include_interactions:
		blocks["microstructure:embedding"] = _interaction_block(
			work, "microstructure", "embedding"
		)
		blocks["microstructure:model"] = _interaction_block(
			work, "microstructure", "model"
		)
		blocks["embedding:model"] = _interaction_block(work, "embedding", "model")

	intercept = np.ones((len(work), 1), dtype=float)
	X_full = np.concatenate([intercept] + [blocks[name] for name in blocks], axis=1)
	full_sse = _sse_from_design(y, X_full)

	rows: list[dict[str, float | str]] = []
	for factor in blocks:
		reduced_parts = [intercept] + [blk for name, blk in blocks.items() if name != factor]
		X_reduced = np.concatenate(reduced_parts, axis=1)
		reduced_sse = _sse_from_design(y, X_reduced)
		contribution = max(0.0, reduced_sse - full_sse)
		rows.append(
			{
				"factor": factor,
				"pct_explained": 100.0 * contribution / total_ss,
			}
		)

	return pd.DataFrame(rows)


def _combo_df(
	df: pd.DataFrame,
	dataset: str,
	target_clean: str,
	prediction_task: str,
) -> pd.DataFrame:
	return df[
		(df["dataset"] == dataset)
		& (df["target_clean"] == target_clean)
		& (df["prediction_task"] == prediction_task)
	].copy()


def plot_variance_decomposition(
	parquet_path: str,
	out_dir: str = "exp_outputs/summary/plots/region_viz",
	include_interactions: bool = False,
) -> Path:
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

	fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6), sharey=True)
	axes_flat = axes.flatten()

	for i, (dataset, target_clean, task, title) in enumerate(COMBOS):
		ax = axes_flat[i]
		df_task = _combo_df(df, dataset, target_clean, task)

		if df_task.empty:
			ax.set_title(f"{title}\n(no data)")
			ax.set_xlabel("")
			ax.set_ylabel("% variance explained")
			ax.set_ylim(0, 100)
			ax.grid(axis="y", linestyle="--", alpha=0.3)
			continue

		try:
			df_task_scored, score_col = _choose_score_column(df_task, task)
			explained = _fit_anova_explained_percent(
				df_task_scored,
				score_col=score_col,
				include_interactions=include_interactions,
			)
		except Exception as exc:  # pragma: no cover - defensive path for data issues
			explained = pd.DataFrame(columns=["factor", "pct_explained"])
			warnings.warn(f"Variance decomposition failed for {title}: {exc}")

		if explained.empty:
			ax.set_title(f"{title}\n(no valid model)")
			ax.set_xlabel("")
			ax.set_ylabel("% variance explained")
			ax.set_ylim(0, 100)
			ax.grid(axis="y", linestyle="--", alpha=0.3)
			continue

		order = ["microstructure", "embedding", "model"]
		if include_interactions:
			order += ["microstructure:embedding", "microstructure:model", "embedding:model"]

		explained = explained.set_index("factor").reindex(order).reset_index()
		explained["pct_explained"] = explained["pct_explained"].fillna(0.0)
		explained["factor_display"] = explained["factor"].map(_factor_display_name)

		sns.barplot(
			data=explained,
			x="factor_display",
			y="pct_explained",
			ax=ax,
			color="#4C78A8",
		)

		ax.set_title(title)
		ax.set_xlabel("")
		ax.set_ylabel("% variance explained")
		ax.set_ylim(0, 100)
		ax.tick_params(axis="x", labelrotation=24)
		for lbl in ax.get_xticklabels():
			lbl.set_horizontalalignment("right")
		ax.grid(axis="y", linestyle="--", alpha=0.3)

		for p in ax.patches:
			h = p.get_height()
			if h <= 0:
				continue
			ax.text(
				p.get_x() + p.get_width() / 2.0,
				h + 1.0,
				f"{h:.1f}%",
				ha="center",
				va="bottom",
				fontsize=7,
			)

	for ax in axes_flat:
		sns.despine(ax=ax, top=True, right=True)

	fig.suptitle(
		"Variance decomposition of prediction score across dataset-task combinations",
		y=1.02,
	)
	fig.tight_layout()

	out_path = Path(out_dir)
	out_path.mkdir(parents=True, exist_ok=True)
	suffix = "with_interactions" if include_interactions else "main_effects"
	out_file = out_path / f"variance_decomposition_{suffix}.pdf"
	fig.savefig(out_file)
	plt.close(fig)
	return out_file


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Variance decomposition per dataset-task: score ~ microstructure + embedding + model"
		)
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
	parser.add_argument(
		"--include-interactions",
		action="store_true",
		help="Include pairwise interaction terms in the linear model",
	)
	args = parser.parse_args()

	out_file = plot_variance_decomposition(
		parquet_path=args.input,
		out_dir=args.outdir,
		include_interactions=args.include_interactions,
	)
	print("Saved variance decomposition figure to", out_file)


if __name__ == "__main__":
	main()
