import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import (
    CSV_DIR,
    DATA_PATH,
    PLOTS_DIR,
    REGION_MODELS,
    REPRESENTATION_ORDER,
    ensure_output_dirs,
    model_display_name,
    network_selection_proportions,
    prepare_selection_data,
    region_selection_proportions,
)


DATASETS = ["camcan", "hcp"]
TASKS = ["binary_classification"]
MICROSTRUCTURES = ["md"]
TOP_N_REGIONS = 20
HISTOGRAM_BINS = 20


def consensus_scores(
    proportions: pd.DataFrame,
    item_id_column: str,
    item_name_column: str,
) -> pd.DataFrame:
    """Average each item selection score across embedding representations."""
    item_columns = list(dict.fromkeys([item_id_column, item_name_column]))
    return (
        proportions.groupby(
            ["dataset", "model"] + item_columns,
            as_index=False,
        )
        .agg(
            average_score=("selection_proportion", "mean"),
            embedding_count=("region_representation", "nunique"),
        )
        .sort_values(["dataset", "model", "average_score", item_id_column])
    )


def plot_score_histogram(scores: pd.DataFrame, item_label: str) -> None:
    sns.set_theme(context="notebook", style="whitegrid")

    fig, axes = plt.subplots(
        len(DATASETS),
        len(REGION_MODELS),
        figsize=(13, 8),
        sharex=True,
        sharey=True,
    )

    for row, dataset in enumerate(DATASETS):
        for col, model in enumerate(REGION_MODELS):
            ax = axes[row, col]
            subset = scores[(scores["dataset"] == dataset) & (scores["model"] == model)]
            ax.hist(
                subset["average_score"],
                bins=HISTOGRAM_BINS,
                range=(0, 1),
                color="#3a7ca5",
                edgecolor="white",
                linewidth=0.8,
            )
            ax.set_title(f"{dataset} | {model_display_name(model)}")
            ax.set_xlabel("Average score across embeddings")
            ax.set_ylabel(f"Number of {item_label}")
            ax.set_xlim(0, 1)

    fig.suptitle(
        f"Consensus {item_label} scores across embedding representations\n"
        f"{', '.join(TASKS)} | {', '.join(MICROSTRUCTURES)}"
    )
    fig.tight_layout()
    if item_label == "parcels":
        output_path = PLOTS_DIR / "consensus_average_score_histogram.png"
    else:
        output_path = PLOTS_DIR / f"consensus_{item_label}_average_score_histogram.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved consensus score histogram to {output_path}")


def plot_top_items(
    scores: pd.DataFrame,
    item_label: str,
    item_id_column: str,
    item_name_column: str,
) -> None:
    sns.set_theme(context="notebook", style="whitegrid")

    fig, axes = plt.subplots(
        len(DATASETS),
        len(REGION_MODELS),
        figsize=(15, 12),
        sharex=True,
    )

    for row, dataset in enumerate(DATASETS):
        for col, model in enumerate(REGION_MODELS):
            ax = axes[row, col]
            subset = scores[(scores["dataset"] == dataset) & (scores["model"] == model)]
            top_regions = (
                subset.sort_values(
                    ["average_score", item_id_column],
                    ascending=[False, True],
                )
                .head(TOP_N_REGIONS)
                .sort_values("average_score")
            )

            ax.barh(
                top_regions[item_name_column],
                top_regions["average_score"],
                color="#7b8c3a",
            )
            ax.set_title(f"{dataset} | {model_display_name(model)}")
            ax.set_xlabel("Average score across embeddings")
            ax.set_ylabel("")
            ax.set_xlim(0, max(0.01, scores["average_score"].max() * 1.08))
            ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        f"Top {TOP_N_REGIONS} consensus {item_label} across embedding representations\n"
        f"{', '.join(TASKS)} | {', '.join(MICROSTRUCTURES)}"
    )
    fig.tight_layout()
    if item_label == "parcels":
        output_path = PLOTS_DIR / "consensus_top20_parcels.png"
    else:
        output_path = PLOTS_DIR / f"consensus_{item_label}_top20.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved top consensus {item_label} plot to {output_path}")


def main() -> None:
    ensure_output_dirs()
    df = pd.read_parquet(DATA_PATH)
    selection_data = prepare_selection_data(
        df,
        datasets=DATASETS,
        tasks=TASKS,
        microstructures=MICROSTRUCTURES,
        models=REGION_MODELS,
    )
    region_proportions = region_selection_proportions(selection_data)
    item_configs = [
        ("parcels", region_proportions, "region_id", "region_name"),
        (
            "networks",
            network_selection_proportions(region_proportions),
            "network_name",
            "network_name",
        ),
    ]

    for item_label, proportions, item_id_column, item_name_column in item_configs:
        scores = consensus_scores(proportions, item_id_column, item_name_column)
        if item_label == "parcels":
            scores_path = CSV_DIR / "consensus_average_scores.csv"
            top_scores_path = CSV_DIR / "consensus_top20_parcels.csv"
        else:
            scores_path = CSV_DIR / f"consensus_{item_label}_average_scores.csv"
            top_scores_path = CSV_DIR / f"consensus_{item_label}_top20.csv"
        scores.to_csv(scores_path, index=False)
        (
            scores.sort_values(["dataset", "model", "average_score", item_id_column])
            .groupby(["dataset", "model"], group_keys=False)
            .tail(TOP_N_REGIONS)
            .sort_values(
                ["dataset", "model", "average_score"],
                ascending=[True, True, False],
            )
            .to_csv(top_scores_path, index=False)
        )

        plot_score_histogram(scores, item_label)
        plot_top_items(scores, item_label, item_id_column, item_name_column)

        print(f"Saved consensus {item_label} average scores to {scores_path}")
        print(f"Saved top consensus {item_label} to {top_scores_path}")
    print(f"Rows after performance filtering: {len(selection_data):,}")
    print(f"Selected rows: {selection_data['selected'].sum():,}")
    print(f"Embedding representations averaged: {', '.join(REPRESENTATION_ORDER)}")


if __name__ == "__main__":
    main()
