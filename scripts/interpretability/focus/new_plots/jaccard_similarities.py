from pathlib import Path

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
    cosine_matrix,
    jaccard_matrix,
    model_display_name,
    network_selection_proportions,
    prepare_selection_data,
    region_selection_proportions,
    top_items_by_selection,
    top_regions_by_selection,
)


DATASETS = ["camcan", "hcp"]
TASKS = ["binary_classification"]
MICROSTRUCTURES = ["md"]
TOP_N_REGIONS = 10
TOP_N_NETWORKS = 5


def stable_item_scores(regions: pd.DataFrame, item_id_column: str) -> dict[str, pd.Series]:
    scores = {
        representation: representation_regions.groupby(item_id_column)[
            "selection_proportion"
        ].mean()
        for representation, representation_regions in regions.groupby("region_representation")
    }
    for representation in REPRESENTATION_ORDER:
        scores.setdefault(representation, pd.Series(dtype=float))

    return scores


def plot_similarity_heatmap(
    top_matrix: pd.DataFrame,
    full_matrix: pd.DataFrame,
    dataset: str,
    model: str,
    output_path: Path,
    item_label: str,
    top_n: int,
    metric_label: str,
    title_prefix: str,
) -> None:
    sns.set_theme(context="notebook", style="white")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    matrices = [
        (top_matrix, f"Top {top_n} {item_label}"),
        (full_matrix, f"All {item_label}"),
    ]
    for ax, (matrix, title) in zip(axes, matrices):
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="mako",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.4,
            linecolor="white",
            cbar=ax is axes[-1],
            cbar_kws={"label": metric_label},
        )
        ax.set_title(title)
        ax.set_xlabel("Region representation")
        ax.set_ylabel("Region representation")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(
        f"Embedding {title_prefix} - {model_display_name(model)}\n"
        f"{dataset} | {', '.join(TASKS)} | {', '.join(MICROSTRUCTURES)}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_jaccard_heatmap(
    top_matrix: pd.DataFrame,
    full_matrix: pd.DataFrame,
    dataset: str,
    model: str,
    output_path: Path,
    item_label: str,
    top_n: int,
) -> None:
    plot_similarity_heatmap(
        top_matrix,
        full_matrix,
        dataset,
        model,
        output_path,
        item_label,
        top_n,
        metric_label="Weighted Jaccard similarity",
        title_prefix="weighted Jaccard similarity",
    )


def plot_cosine_heatmap(
    top_matrix: pd.DataFrame,
    full_matrix: pd.DataFrame,
    dataset: str,
    model: str,
    output_path: Path,
    item_label: str,
    top_n: int,
) -> None:
    plot_similarity_heatmap(
        top_matrix,
        full_matrix,
        dataset,
        model,
        output_path,
        item_label,
        top_n,
        metric_label="Cosine similarity",
        title_prefix="cosine similarity",
    )


def output_stem(metric: str, dataset: str, model: str, item_kind: str) -> str:
    base = f"{metric}_{dataset}_{model}_{'_'.join(MICROSTRUCTURES)}_{'_'.join(TASKS)}"
    if item_kind == "regions":
        return base

    return f"{base}_{item_kind}"


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
    network_proportions = network_selection_proportions(region_proportions)
    item_configs = [
        (
            "regions",
            region_proportions,
            top_regions_by_selection(
                region_proportions,
                group_columns=["dataset", "model", "region_representation"],
                top_n=TOP_N_REGIONS,
            ),
            "region_id",
            TOP_N_REGIONS,
        ),
        (
            "networks",
            network_proportions,
            top_items_by_selection(
                network_proportions,
                group_columns=["dataset", "model", "region_representation"],
                top_n=TOP_N_NETWORKS,
                item_id_column="network_name",
            ),
            "network_name",
            TOP_N_NETWORKS,
        ),
    ]

    metric_configs = [
        ("jaccard", jaccard_matrix, plot_jaccard_heatmap),
        ("cosine", cosine_matrix, plot_cosine_heatmap),
    ]

    for item_kind, proportions, top_items, item_id_column, top_n in item_configs:
        for dataset in DATASETS:
            for model in REGION_MODELS:
                full_subset = proportions[
                    (proportions["dataset"] == dataset) & (proportions["model"] == model)
                ]
                top_subset = top_items[
                    (top_items["dataset"] == dataset) & (top_items["model"] == model)
                ]
                top_scores = stable_item_scores(top_subset, item_id_column)
                full_scores = stable_item_scores(full_subset, item_id_column)

                for metric_name, matrix_fn, plot_fn in metric_configs:
                    top_metric_matrix = matrix_fn(top_scores, REPRESENTATION_ORDER)
                    full_metric_matrix = matrix_fn(full_scores, REPRESENTATION_ORDER)

                    stem = output_stem(metric_name, dataset, model, item_kind)
                    matrix_path = CSV_DIR / f"{stem}.csv"
                    full_matrix_path = CSV_DIR / f"{stem}_full_{item_kind}.csv"
                    top_items_path = CSV_DIR / f"{stem}_top_{item_kind}.csv"
                    png_path = PLOTS_DIR / f"{stem}.pdf"

                    top_metric_matrix.to_csv(matrix_path)
                    full_metric_matrix.to_csv(full_matrix_path)
                    top_subset.to_csv(top_items_path, index=False)
                    plot_fn(
                        top_metric_matrix,
                        full_metric_matrix,
                        dataset,
                        model,
                        png_path,
                        item_kind,
                        top_n,
                    )

                    print(
                        f"Saved {dataset} {model} {item_kind} {metric_name} heatmap to {png_path}"
                    )
                    print(
                        f"Saved {dataset} {model} top-{item_kind} {metric_name} matrix "
                        f"to {matrix_path}"
                    )
                    print(
                        f"Saved {dataset} {model} full-{item_kind} {metric_name} matrix "
                        f"to {full_matrix_path}"
                    )
                    print(f"Saved {dataset} {model} top-{item_kind} sets to {top_items_path}")

    print(f"Rows after performance filtering: {len(selection_data):,}")
    print(f"Selected rows: {selection_data['selected'].sum():,}")


if __name__ == "__main__":
    main()
