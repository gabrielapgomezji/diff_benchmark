import ast
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).parent.parent / "df.parquet"
PLOTS_DIR = Path(__file__).parent.parent / "outputs/plots"
CSV_DIR = Path(__file__).parent.parent / "outputs/csv"

REPRESENTATION_ORDER = ["summary_stats", "mean_std", "percentiles", "pca", "flatten"]
REGION_MODELS = ["region_elasticnet", "region_group_lasso"]

PERFORMANCE_QUANTILE = 0.10
SELECTION_QUANTILE = 0.90
NETWORK_SCORE_METHOD = "at_least_one"


def ensure_output_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)


def parse_region_representation(embedding_config: str) -> str:
    """Return the region representation encoded in an embedding config."""
    try:
        embedding = ast.literal_eval(embedding_config)
    except (ValueError, SyntaxError):
        return "unknown"

    if not isinstance(embedding, dict):
        return "unknown"

    return embedding.get("region_representation", "unknown")


def model_display_name(model: str) -> str:
    return model.removeprefix("region_").replace("_", " ").title()


def parse_network_name(region_name: str) -> str:
    """Return the 17-network label encoded in an atlas region name."""
    parts = str(region_name).split("_")
    if len(parts) >= 3 and parts[1] in {"LH", "RH"}:
        return parts[2]

    return "unknown"

def parse_global_network_name(region_name: str) -> str:
    parts = str(region_name).split("_")

    if len(parts) < 3 or parts[1] not in {"LH", "RH"}:
        return "unknown"

    net17 = parts[2]

    if net17.startswith("Vis"):
        return "Visual"
    if net17.startswith("SomMot"):
        return "Somatomotor"
    if net17.startswith("DorsAttn"):
        return "DorsalAttention"
    if net17.startswith("SalVentAttn"):
        return "VentralAttention"
    if net17.startswith("Limbic"):
        return "Limbic"
    if net17.startswith("Cont"):
        return "Control"
    if net17.startswith("Default") or net17 == "TempPar":
        return "Default"

    return "unknown"


def task_performance_thresholds(
    df: pd.DataFrame,
    tasks: list[str],
    performance_quantile: float = PERFORMANCE_QUANTILE,
) -> pd.Series:
    runs = df[df["task"].isin(tasks)].drop_duplicates("exp_id")
    return runs.groupby("task")["test_score"].quantile(performance_quantile)


def prepare_selection_data(
    df: pd.DataFrame,
    datasets: list[str],
    tasks: list[str],
    microstructures: list[str],
    models: list[str] | None = None,
    representation_order: list[str] | None = None,
    performance_quantile: float = PERFORMANCE_QUANTILE,
    selection_quantile: float = SELECTION_QUANTILE,
    selection_group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return one row per eligible run/region with a boolean selection marker."""
    if models is None:
        models = REGION_MODELS
    if representation_order is None:
        representation_order = REPRESENTATION_ORDER
    if selection_group_columns is None:
        selection_group_columns = ["model", "embedding"]

    thresholds = task_performance_thresholds(df, tasks, performance_quantile)
    data = df[
        df["dataset"].isin(datasets)
        & df["task"].isin(tasks)
        & df["microstructure"].isin(microstructures)
        & df["model"].isin(models)
        & df["region_name"].notna()
    ].copy()

    data["performance_threshold"] = data["task"].map(thresholds)
    data = data[data["test_score"] >= data["performance_threshold"]].copy()

    data["region_representation"] = data["embedding"].map(parse_region_representation)
    data = data[data["region_representation"].isin(representation_order)].copy()

    data["selection_threshold"] = data.groupby(selection_group_columns)["coef"].transform(
        lambda coef: coef.quantile(selection_quantile)
    )
    data["selected"] = data["coef"] >= data["selection_threshold"]
    return data


def region_selection_proportions(selection_data: pd.DataFrame) -> pd.DataFrame:
    """Return the proportion of eligible runs where each region was selected."""
    group_columns = [
        "dataset",
        "task",
        "microstructure",
        "model",
        "region_representation",
        "region_id",
        "region_name",
    ]
    return (
        selection_data.groupby(group_columns, as_index=False)["selected"]
        .mean()
        .rename(columns={"selected": "selection_proportion"})
    )


def network_selection_proportions(
    region_proportions: pd.DataFrame,
    score_method: str = NETWORK_SCORE_METHOD,
) -> pd.DataFrame:
    """Return network scores from region scores.

    score_method="average" uses the mean region score in each network.
    score_method="at_least_one" uses 1 - prod(1 - p_i).
    """
    data = region_proportions.copy()
    data["network_name"] = data["region_name"].map(parse_network_name)
    group_columns = [
        "dataset",
        "task",
        "microstructure",
        "model",
        "region_representation",
        "network_name",
    ]

    score_method = score_method.lower()
    if score_method in {"a", "option_a"}:
        score_method = "average"
    elif score_method in {"b", "option_b"}:
        score_method = "at_least_one"

    if score_method == "average":
        network_scores = data.groupby(group_columns, as_index=False)[
            "selection_proportion"
        ].mean()
    elif score_method == "at_least_one":
        network_scores = (
            data.groupby(group_columns)["selection_proportion"]
            .apply(lambda scores: 1 - (1 - scores).prod())
            .reset_index(name="selection_proportion")
        )
    else:
        raise ValueError(
            "score_method must be one of: 'average', 'at_least_one', 'A', 'B'"
        )

    return network_scores.sort_values(group_columns).reset_index(drop=True)

def main_network_selection_proportions(
    region_proportions: pd.DataFrame,
    score_method: str = NETWORK_SCORE_METHOD,
) -> pd.DataFrame:
    """Return network scores from region scores.

    score_method="average" uses the mean region score in each network.
    score_method="at_least_one" uses 1 - prod(1 - p_i).
    """
    data = region_proportions.copy()
    data["network_name"] = data["region_name"].map(parse_global_network_name)
    group_columns = [
        "dataset",
        "task",
        "microstructure",
        "model",
        "region_representation",
        "network_name",
    ]

    score_method = score_method.lower()
    if score_method in {"a", "option_a"}:
        score_method = "average"
    elif score_method in {"b", "option_b"}:
        score_method = "at_least_one"

    if score_method == "average":
        network_scores = data.groupby(group_columns, as_index=False)[
            "selection_proportion"
        ].mean()
    elif score_method == "at_least_one":
        network_scores = (
            data.groupby(group_columns)["selection_proportion"]
            .apply(lambda scores: 1 - (1 - scores).prod())
            .reset_index(name="selection_proportion")
        )
    else:
        raise ValueError(
            "score_method must be one of: 'average', 'at_least_one', 'A', 'B'"
        )

    return network_scores.sort_values(group_columns).reset_index(drop=True)

def top_items_by_selection(
    proportions: pd.DataFrame,
    group_columns: list[str],
    top_n: int,
    item_id_column: str,
) -> pd.DataFrame:
    return (
        proportions.sort_values(
            group_columns + ["selection_proportion", item_id_column],
            ascending=[True] * len(group_columns) + [False, True],
        )
        .groupby(group_columns, group_keys=False)
        .head(top_n)
    )


def top_regions_by_selection(
    proportions: pd.DataFrame,
    group_columns: list[str],
    top_n: int,
) -> pd.DataFrame:
    return top_items_by_selection(
        proportions,
        group_columns,
        top_n,
        item_id_column="region_id",
    )


def jaccard_matrix(region_scores: dict[str, pd.Series], labels: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for left in labels:
        for right in labels:
            scores = pd.concat(
                [region_scores[left], region_scores[right]],
                axis=1,
            ).fillna(0)
            numerator = scores.min(axis=1).sum()
            denominator = scores.max(axis=1).sum()
            matrix.loc[left, right] = numerator / denominator if denominator else 1.0

    return matrix


def cosine_matrix(region_scores: dict[str, pd.Series], labels: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for left in labels:
        for right in labels:
            scores = pd.concat(
                [region_scores[left], region_scores[right]],
                axis=1,
            ).fillna(0)
            values = scores.to_numpy(dtype=float)
            numerator = float(np.dot(values[:, 0], values[:, 1]))
            denominator = float(np.linalg.norm(values[:, 0]) * np.linalg.norm(values[:, 1]))
            if denominator == 0.0:
                left_empty = float(np.linalg.norm(values[:, 0])) == 0.0
                right_empty = float(np.linalg.norm(values[:, 1])) == 0.0
                matrix.loc[left, right] = 1.0 if left_empty and right_empty else 0.0
            else:
                matrix.loc[left, right] = numerator / denominator

    return matrix

def concordance_correlation_coefficient(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.size == 0 or y.size == 0:
        return np.nan
    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    if x.size < 2:
        return 1.0 if np.allclose(x, y) else np.nan

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    var_x = np.var(x)
    var_y = np.var(y)

    cov_xy = np.cov(x, y, ddof=0)[0, 1]

    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    if denominator == 0.0:
        return 1.0 if np.allclose(x, y) else np.nan

    return 2 * cov_xy / denominator

# def network_ccc(network_proportions: pd.DataFrame) -> pd.DataFrame:
#     """
#     Compute per-network CCC across embedding representations.

#     Parameters
#     ----------
#     network_proportions : DataFrame
#         Output of ``network_selection_proportions()``.
#         Must contain: dataset, model, network_name,
#                       region_representation, selection_proportion.
#         Do NOT pass the consensus ``scores`` DataFrame here — that one has
#         already averaged out ``region_representation``.
#     """
#     results = []

#     for network in sorted(network_proportions["network_name"].unique()):
#         subset = network_proportions[network_proportions["network_name"] == network]

#         def get_vec(dataset, model):
#             return (
#                 subset[
#                     (subset["dataset"] == dataset)
#                     & (subset["model"] == model)
#                 ]
#                 .groupby("region_representation")["selection_proportion"]
#                 .mean()
#             )

#         def paired_ccc(ld, lm, rd, rm):
#             left  = get_vec(ld, lm)
#             right = get_vec(rd, rm)
#             paired = pd.concat([left, right], axis=1, join="inner").dropna()
#             if paired.empty:
#                 return np.nan
#             return concordance_correlation_coefficient(
#                 paired.iloc[:, 0].to_numpy(),
#                 paired.iloc[:, 1].to_numpy(),
#             )

#         results.append({
#             "network":           network,
#             "ccc_hcp_models":    paired_ccc("hcp",    "region_group_lasso", "hcp",    "region_elasticnet"),
#             "ccc_camcan_models": paired_ccc("camcan", "region_group_lasso", "camcan", "region_elasticnet"),
#             "ccc_gl_datasets":   paired_ccc("hcp",    "region_group_lasso", "camcan", "region_group_lasso"),
#             "ccc_en_datasets":   paired_ccc("hcp",    "region_elasticnet",  "camcan", "region_elasticnet"),
#         })

#     return pd.DataFrame(results)


# # ── 2. Helper: pivot consensus scores for scatter panels ─────────────────────

# def _pivot_network_scores(network_scores: pd.DataFrame):
#     """
#     Returns two pivoted DataFrames built from consensus ``network_scores``.

#     pivot_models : index=(dataset, network_name), columns=(region_group_lasso, region_elasticnet)
#     pivot_datasets: index=(model,   network_name), columns=(hcp, camcan)
#     """
#     pivot_models = (
#         network_scores
#         .pivot_table(index=["dataset",  "network_name"],
#                      columns="model",   values="average_score", aggfunc="mean")
#         .reset_index()
#     )
#     pivot_models.columns.name = None

#     pivot_datasets = (
#         network_scores
#         .pivot_table(index=["model",    "network_name"],
#                      columns="dataset", values="average_score", aggfunc="mean")
#         .reset_index()
#     )
#     pivot_datasets.columns.name = None

#     return pivot_models, pivot_datasets