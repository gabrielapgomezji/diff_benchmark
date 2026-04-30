from __future__ import annotations

from pathlib import Path
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_TABLE = PROJECT_ROOT / "exp_outputs" / "summary" / "coefficients_long.parquet"
OUTPUT_DIR = PROJECT_ROOT / "exp_outputs" / "summary" / "model_agreement_plots"


# ---------------------------------------------------------------------
# HELPERS (same logic as your pipeline)
# ---------------------------------------------------------------------

def _build_exp_id(df: pd.DataFrame) -> pd.Series:
    cols = ["run_id"]
    if "fold" in df.columns:
        cols.append("fold")
    if "seed" in df.columns:
        cols.append("seed")
    return df[cols].astype(str).agg("_".join, axis=1)


def _pick_group_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in ["task", "dataset"] if c in df.columns]


def _safe_parse_embedding(s: str) -> dict:
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


# ---------------------------------------------------------------------
# MODEL / EMBEDDING NORMALIZATION (same as before)
# ---------------------------------------------------------------------

def _extract_model_embedding(row: pd.Series) -> tuple[str, str]:
    model = row.get("model_type", row.get("model", "unknown"))
    emb_dict = _safe_parse_embedding(row.get("embedding", ""))

    if "region_encoder" in emb_dict:
        enc = emb_dict["region_encoder"]
        if enc.get("include_size", False):
            return "region_group_lasso", "pointnet_size"
        return "region_group_lasso", f"pointnet_{enc.get('type', 'unknown')}"

    if model == "region_pca":
        return "region_group_lasso", "pca"

    if "region_representation" in emb_dict:
        return model, emb_dict["region_representation"]

    return model, "unknown"


def _add_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df.apply(_extract_model_embedding, axis=1)
    out = df.copy()
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


# ---------------------------------------------------------------------
# FILTERING (same logic)
# ---------------------------------------------------------------------

def _select_best_experiments(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    score_df = df[["exp_id", "test_score"] + group_cols].dropna().drop_duplicates()

    def _filter_group(group):
        scores = group["test_score"]
        q1 = scores.quantile(0.25)
        q3 = scores.quantile(0.75)
        iqr = q3 - q1
        keep = scores >= (q1 - 1.5 * iqr)
        return group[keep]

    # return score_df.groupby(group_cols).apply(_filter_group).reset_index(drop=True)
    return (
        score_df
        .groupby(group_cols, group_keys=False)
        .apply(_filter_group, include_groups=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# SELECTION
# ---------------------------------------------------------------------

def _add_percentile_selection(df: pd.DataFrame, percentile: float = 0.9):
    keys = ["run_id", "fold"] if "fold" in df.columns else ["run_id"]

    df["selected"] = df.groupby(keys)["coef"].transform(
        lambda s: (s.abs() >= s.abs().quantile(percentile)).astype(int)
    )
    return df


# ---------------------------------------------------------------------
# CORE COMPUTATION
# ---------------------------------------------------------------------

# def compute_model_agreement(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Returns dataframe with:
#     dataset, task, region, mean_freq, std_freq
#     """

#     # selection frequency per model
#     freq = (
#         df.groupby(["dataset", "task", "model_name", "region"])["selected"]
#         .mean()
#         .reset_index(name="freq")
#     )

#     # aggregate across models
#     agg = (
#         freq.groupby(["dataset", "task", "region"])["freq"]
#         .agg(mean_freq="mean", std_freq="std")
#         .reset_index()
#     )

#     return agg
def compute_model_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns dataframe with:
    dataset, task, model_name, region, mean_freq, std_freq
    where std is computed across runs within each model.
    """

    # define what a "run" is
    run_keys = ["dataset", "task", "model_name", "region", "exp_id"]

    # average selection per run (in case duplicates exist)
    per_run = (
        df.groupby(run_keys)["selected"]
        .mean()
        .reset_index()
    )

    # now compute mean + std across runs (within each model)
    agg = (
        per_run.groupby(["dataset", "task", "model_name", "region"])["selected"]
        .agg(mean_freq="mean", std_freq="std")
        .reset_index()
    )

    return agg

def compute_model_agreement_bernoulli(df: pd.DataFrame) -> pd.DataFrame:
    run_keys = ["dataset", "task", "model_name", "region", "exp_id"]

    # one value per run
    per_run = (
        df.groupby(run_keys)["selected"]
        .mean()
        .reset_index()
    )

    grouped = per_run.groupby(["dataset", "task", "model_name", "region"])

    agg = grouped["selected"].agg(
        mean_freq="mean",
        n_runs="count"
    ).reset_index()

    # compute Bernoulli standard error
    agg["se_freq"] = np.sqrt(
        agg["mean_freq"] * (1 - agg["mean_freq"]) / agg["n_runs"]
    )

    return agg


def compute_embedding_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns dataframe with:
    dataset, task, region, mean_freq, std_freq
    computed across embeddings (within each dataset/task/region).
    """

    freq = (
        df.groupby(["dataset", "task", "embedding_name", "region"])["selected"]
        .mean()
        .reset_index(name="freq")
    )

    # agg = (
    #     freq.groupby(["dataset", "task", "region"])["freq"]
    #     .agg(mean_freq="mean", std_freq="std")
    #     .reset_index()
    # )
    agg = (
        freq.groupby(["dataset", "task", "embedding_name", "region"])["freq"]
        .agg(mean_freq="mean", std_freq="std")
        .reset_index()
    )

    return agg

def compute_embedding_agreement_bernoulli(df: pd.DataFrame) -> pd.DataFrame:
    run_keys = ["dataset", "task", "embedding_name", "region", "exp_id"]

    # one value per run
    per_run = (
        df.groupby(run_keys)["selected"]
        .mean()
        .reset_index()
    )

    grouped = per_run.groupby(["dataset", "task", "embedding_name", "region"])

    agg = grouped["selected"].agg(
        mean_freq="mean",
        n_runs="count"
    ).reset_index()

    # Bernoulli SE
    agg["se_freq"] = np.sqrt(
        agg["mean_freq"] * (1 - agg["mean_freq"]) / agg["n_runs"]
    )

    return agg

# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

# def plot_agreement_scatter(df: pd.DataFrame):
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#     for (dataset, task), sub in df.groupby(["dataset", "task"]):
#         sub = sub.dropna(subset=["mean_freq", "std_freq"])
#         x = sub["mean_freq"]
#         y = sub["std_freq"]

#         plt.figure(figsize=(5, 5))

#         plt.scatter(x, y, alpha=0.7)

#         plt.xlabel("Average selection frequency (across models)")
#         plt.ylabel("Std across models (disagreement)")
#         plt.title(f"{dataset} | {task}")

#         # quadrant lines
#         plt.axvline(0.5, linestyle="--", linewidth=1)
#         plt.axhline(y.mean(), linestyle="--", linewidth=1)

#         plt.xlim(0, 1)
#         # plt.ylim(0, y.max() * 1.1)
#         y_max = y.max()
#         if not np.isfinite(y_max) or y_max == 0:
#             y_max = 1e-6
#         plt.ylim(0, y_max * 1.1)

#         plt.tight_layout()

#         plt.savefig(
#             OUTPUT_DIR / f"agreement_{dataset}_{task}.png",
#             dpi=150
#         )
#         plt.close()
def plot_agreement_scatter(df: pd.DataFrame, bernoulli: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for (dataset, task), sub in df.groupby(["dataset", "task"]):
        if bernoulli:
            sub = sub.dropna(subset=["mean_freq", "se_freq"])
        else:
            sub = sub.dropna(subset=["mean_freq", "std_freq"])

        plt.figure(figsize=(6, 5))

        for model, msub in sub.groupby("model_name"):
            plt.scatter(
                msub["mean_freq"],
                msub["std_freq"] if not bernoulli else msub["se_freq"],
                alpha=0.7,
                label=model
            )

        plt.xlabel("Average selection frequency")
        if bernoulli:
            plt.ylabel("Standard error of p̂")
        else:
            plt.ylabel("Std across runs (within model)")
        plt.title(f"{dataset} | {task}")

        plt.axvline(0.5, linestyle="--", linewidth=1)

        y_col = "se_freq" if bernoulli else "std_freq"
        y_max = sub[y_col].max()
        if not np.isfinite(y_max) or y_max == 0:
            y_max = 1e-6
        plt.ylim(0, y_max * 1.1)

        plt.legend()
        plt.tight_layout()
        if bernoulli:
            plt.savefig(
                OUTPUT_DIR / f"agreement_bernoulli_{dataset}_{task}.png",
                dpi=150
            )
        else:
            plt.savefig(
                OUTPUT_DIR / f"agreement_{dataset}_{task}.png",
                dpi=150
            )
        plt.close()


def plot_embedding_agreement_scatter(df: pd.DataFrame, bernoulli: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for (dataset, task), sub in df.groupby(["dataset", "task"]):

        y_col = "se_freq" if bernoulli else "std_freq"

        sub = sub.dropna(subset=["mean_freq", y_col])

        plt.figure(figsize=(6, 5))

        for emb, esub in sub.groupby("embedding_name"):
            plt.scatter(
                esub["mean_freq"],
                esub[y_col],
                alpha=0.7,
                label=emb
            )

        plt.xlabel("Average selection frequency")

        plt.ylabel(
            "Standard error of p̂"
            if bernoulli else
            "Std across embeddings (disagreement)"
        )

        title_suffix = "(Bernoulli SE)" if bernoulli else "(Embedding disagreement)"
        plt.title(f"{dataset} | {task} {title_suffix}")

        plt.axvline(0.5, linestyle="--", linewidth=1)
        plt.axhline(sub[y_col].mean(), linestyle="--", linewidth=1)

        plt.xlim(0, 1)

        y_max = sub[y_col].max()
        if not np.isfinite(y_max) or y_max == 0:
            y_max = 1e-6
        plt.ylim(0, y_max * 1.1)

        plt.legend()
        plt.tight_layout()

        filename = (
            f"agreement_embedding_bernoulli_{dataset}_{task}.png"
            if bernoulli else
            f"agreement_embedding_{dataset}_{task}.png"
        )

        plt.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    df = pd.read_parquet(INPUT_TABLE)

    df = _exclude_region_permutation_models(df)

    df["exp_id"] = _build_exp_id(df)
    df = _add_model_columns(df)

    group_cols = _pick_group_columns(df)
    best = _select_best_experiments(df, group_cols)

    df = df[df["exp_id"].isin(best["exp_id"])].copy()

    df = _add_percentile_selection(df)

    agg_model = compute_model_agreement(df)
    agg_model_bernoulli = compute_model_agreement_bernoulli(df)
    agg_embedding = compute_embedding_agreement(df)
    agg_embedding_bernoulli = compute_embedding_agreement_bernoulli(df)

    plot_agreement_scatter(agg_model)
    plot_embedding_agreement_scatter(agg_embedding)
    plot_agreement_scatter(agg_model_bernoulli, bernoulli=True)
    plot_embedding_agreement_scatter(agg_embedding_bernoulli, bernoulli=True)

if __name__ == "__main__":
    main()