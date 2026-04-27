from __future__ import annotations

import hashlib
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def _safe_tag(value: object, max_len: int = 40) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.-")
    if not text:
        text = "na"
    if len(text) <= max_len:
        return text

    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10]
    keep = max(8, max_len - 11)
    return f"{text[:keep]}_{digest}"

def _extract_region_repr(embedding: str) -> str:
    match = re.search(r"region_repr_([A-Za-z0-9._-]+)", str(embedding))
    if not match:
        return "na"
    if match:
        return match.group(0)


def _hist_output_path(output_dir: Path, dataset: object, task: object, embedding: object) -> Path:
    ds = _safe_tag(dataset, max_len=32)
    tk = _safe_tag(task, max_len=7)
    # em = re.search(r"region_representation", str(embedding)).group(0)
    # em = _safe_tag(embedding, max_len=40)
    em = _extract_region_repr(embedding)
    key = f"{dataset}|{task}|{embedding}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    filename = f"{ds}_{tk}_{em}_{digest}_hist.png"
    return output_dir / filename


def bootstrap_coeff_distributions(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    group_cols: list[str] = ["dataset", "task", "embedding"],
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Bootstrap coefficient distributions per region.
    Assumes df already contains:
        - coef_norm
        - region
        - exp_id (IMPORTANT, one per run/fold/seed)
    """
    rng = np.random.default_rng(random_state)

    required = {"exp_id", "region", "coef_norm"}
    missing = required - set(df.columns)
    if missing:
        missing_txt = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns for bootstrap: {missing_txt}")

    missing_groups = [c for c in group_cols if c not in df.columns]
    if missing_groups:
        missing_txt = ", ".join(missing_groups)
        raise ValueError(f"Missing grouping columns for bootstrap: {missing_txt}")

    results = []

    for keys, group in df.groupby(group_cols):
        # Keep one coefficient per (exp_id, region) to ensure each fold/seed experiment
        # contributes independently in resampling.
        exp_region = (
            group.groupby(["exp_id", "region"], as_index=False)["coef_norm"]
            .mean()
        )
        exp_ids = exp_region["exp_id"].unique()

        if len(exp_ids) == 0:
            continue

        for _ in range(n_bootstrap):
            sampled_ids = rng.choice(exp_ids, size=len(exp_ids), replace=True)
            sampled_parts = []
            for exp_id in sampled_ids:
                sampled_parts.append(exp_region[exp_region["exp_id"] == exp_id])
            sample = pd.concat(sampled_parts, ignore_index=True)

            agg = sample.groupby("region", as_index=False)["coef_norm"].mean()

            for _, row in agg.iterrows():
                results.append({
                    **dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))),
                    "region": row["region"],
                    "coef": row["coef_norm"],
                })

    return pd.DataFrame(results)


def plot_histogram_grid(
    df: pd.DataFrame,
    dataset: str,
    task: str,
    embedding: str,
    out_file: Path,
    max_regions: int = 50,
):
    """
    Plot histogram grid of coefficient distributions.
    """

    subset = df[
        (df["dataset"] == dataset) &
        (df["task"] == task) &
        (df["embedding"] == embedding)
    ]

    if subset.empty:
        return

    regions = subset["region"].unique()[:max_regions]
    if len(regions) == 0:
        return

    n = len(regions)
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows))
    axes = axes.flatten()

    for i, region in enumerate(regions):
        ax = axes[i]
        data = subset[subset["region"] == region]["coef"]

        ax.hist(data, bins=30)
        ax.set_title(f"Region {region}", fontsize=8)

    # Remove empty plots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{dataset} | {task} | {embedding}")
    fig.tight_layout()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def run_analysis(
    input_path: Path,
    output_dir: Path,
):
    df = pd.read_parquet(input_path)

    if "exp_id" not in df.columns:
        raise ValueError("exp_id is required. Did you update normalization script?")

    if "coef_norm" not in df.columns:
        raise ValueError("coef_norm is required. Input must be selected-normalized output.")

    boot_df = bootstrap_coeff_distributions(df)

    if boot_df.empty:
        raise ValueError("No bootstrap results were generated. Check input filters/group columns.")

    for (dataset, task, embedding), _ in boot_df.groupby(
        ["dataset", "task", "embedding"]
    ):
        out_file = _hist_output_path(output_dir, dataset, task, embedding)

        plot_histogram_grid(
            boot_df,
            dataset,
            task,
            embedding,
            out_file,
        )


if __name__ == "__main__":
    INPUT = Path("exp_outputs/summary/coefficients_selected_normalized.parquet")
    OUTPUT = Path("exp_outputs/summary/bootstrap_hists")

    run_analysis(INPUT, OUTPUT)