from __future__ import annotations

from pathlib import Path
import ast

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "coefficients_long.parquet"
OUTPUT_DIR = PROJECT_ROOT / "exp_outputs" / "summary" / "feature_importance_plots"

MICROSTRUCTURES = ["md", "mk", "sh", "b0"]
REGION_REPRESENTATIONS = ["flatten", "mean_std", "summary_stats", "percentiles", "pca"]
USE_ALL_FOLDS = True
FIRST_FOLD_ID = 0
EMBEDDING_SELECTION = None #"flatten"


def _build_exp_id(df: pd.DataFrame) -> pd.Series:
	cols = ["run_id"]
	if "fold" in df.columns:
		cols.append("fold")
	if "seed" in df.columns:
		cols.append("seed")
	return df[cols].astype(str).agg("_".join, axis=1)


def _safe_parse_embedding(value: str) -> dict:
	try:
		return ast.literal_eval(value)
	except Exception:
		return {}


def _extract_permutation_region(embedding_raw: str | dict) -> int | None:
	if isinstance(embedding_raw, dict):
		data = embedding_raw
	else:
		data = _safe_parse_embedding(str(embedding_raw))
	if not isinstance(data, dict):
		return None
	region = data.get("permutation_region", None)
	if region is None:
		return None
	try:
		return int(region)
	except Exception:
		return None


def _prepare_score_table(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	if "exp_id" not in out.columns:
		out["exp_id"] = _build_exp_id(out)
	cols = [
		"exp_id",
		"run_id",
		"fold",
		"seed",
		"dataset",
		"task",
		"model",
		"microstructure",
		"embedding",
		"test_score",
	]
	keep = [c for c in cols if c in out.columns]
	return out[keep].drop_duplicates()


def _filter_md_fold0(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	if "microstructure" in out.columns:
		out = out[out["microstructure"].astype(str).isin(MICROSTRUCTURES)]
	if not USE_ALL_FOLDS and "fold" in out.columns:
		out = out[out["fold"] == FIRST_FOLD_ID]
	return out


def _select_representations() -> list[str]:
	if EMBEDDING_SELECTION is None:
		return REGION_REPRESENTATIONS
	selection = str(EMBEDDING_SELECTION)
	return [selection] if selection in REGION_REPRESENTATIONS else REGION_REPRESENTATIONS


def _plot_region_deltas(df: pd.DataFrame, title: str, out_file: Path) -> None:
	if df.empty:
		return
	std_vals = df["delta_std"].fillna(0.0).astype(float)
	max_std = float(std_vals.max()) if np.isfinite(std_vals.max()) else 0.0
	if max_std <= 0.0:
		sizes = np.full(len(std_vals), 40.0)
	else:
		sizes = 30.0 + (std_vals / max_std) * 170.0
	plt.figure(figsize=(8, 4))
	plt.scatter(df["permutation_region"], df["delta_mean"], alpha=0.75, s=sizes)
	plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
	plt.xlabel("Permutation region")
	plt.ylabel("Test score delta (mean)")
	plt.title(title)
	plt.tight_layout()
	out_file.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out_file, dpi=150)
	plt.close()


def _compute_region_summary(perm_df: pd.DataFrame) -> pd.DataFrame:
	return (
		perm_df.groupby(["dataset", "task", "fold", "permutation_region"], dropna=False)["test_score"]
		.mean()
		.reset_index()
	)


def _extract_region_representation(embedding_raw: str | dict) -> str | None:
	if isinstance(embedding_raw, dict):
		data = embedding_raw
	else:
		data = _safe_parse_embedding(str(embedding_raw))
	if not isinstance(data, dict):
		return None
	value = data.get("region_representation", None)
	return str(value) if value is not None else None


def _align_folds_across_regions(sub: pd.DataFrame) -> pd.DataFrame:
	if not USE_ALL_FOLDS:
		return sub
	if "fold" not in sub.columns:
		return sub
	region_folds = (
		sub.groupby("permutation_region", dropna=False)["fold"]
		.apply(lambda s: set(s.dropna().unique()))
	)
	if region_folds.empty:
		return sub
	common_folds = set.intersection(*region_folds.tolist()) if len(region_folds) > 1 else next(iter(region_folds), set())
	if not common_folds:
		return sub
	min_count = min(len(folds) for folds in region_folds.tolist() if folds)
	ordered = sorted(common_folds)
	selected_folds = ordered[:min_count] if len(ordered) > min_count else ordered
	return sub[sub["fold"].isin(selected_folds)]


def main() -> None:
	df = pd.read_parquet(INPUT_TABLE)
	breakpoint()
	score_df = _prepare_score_table(df)
	score_df = _filter_md_fold0(score_df)

	score_df["embedding_dict"] = score_df["embedding"].apply(_safe_parse_embedding)
	score_df["permutation_region"] = score_df["embedding_dict"].apply(_extract_permutation_region)
	score_df["region_representation"] = score_df["embedding_dict"].apply(_extract_region_representation)

	perm_mask = score_df["model"].astype(str).str.contains("region_permutation", case=False, na=False)
	base_perm_df = score_df[perm_mask & score_df["permutation_region"].notna()].copy()

	if base_perm_df.empty:
		print("No permutation-region models found after filtering.")
		return

	representations = _select_representations()

	for microstructure in MICROSTRUCTURES:
		micro_df = base_perm_df[base_perm_df["microstructure"].astype(str) == microstructure]
		if micro_df.empty:
			continue

		for representation in representations:
			perm_df = micro_df[micro_df["region_representation"].astype(str) == representation]
			if perm_df.empty:
				continue

			perm_summary = _compute_region_summary(perm_df)

			# Baseline: region 0 within permutation runs
			for (dataset, task), sub in perm_summary.groupby(["dataset", "task"], dropna=False):
				sub = _align_folds_across_regions(sub)
				baseline_region0 = sub[sub["permutation_region"] == 0]
				if baseline_region0.empty:
					continue
				baseline_region0 = baseline_region0.rename(columns={"test_score": "baseline_score"})
				merged = sub.merge(
					baseline_region0[["dataset", "task", "fold", "baseline_score"]],
					on=["dataset", "task", "fold"],
					how="inner",
				)
				merged["delta"] = merged["baseline_score"] - merged["test_score"]
				stats = (
					merged.groupby(["dataset", "task", "permutation_region"], dropna=False)["delta"]
					.agg(delta_mean="mean", delta_std="std")
					.reset_index()
				)
				_plot_region_deltas(
					stats,
					(
						"Delta = test_score_region0 - test_score_region | "
						f"{microstructure} | {representation} | {dataset} | {task}"
					),
					OUTPUT_DIR
					/ f"perm_region_delta_vs_region0_{microstructure}_{representation}_{dataset}_{task}.png",
				)


if __name__ == "__main__":
	main()
