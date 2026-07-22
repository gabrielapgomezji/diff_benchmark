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


def plot_model_similarity_grid(
    plot_data: list[tuple[str, int, pd.DataFrame, pd.DataFrame]],
    dataset: str,
    model: str,
    output_path: Path,
    metric_label: str,
    title_prefix: str,
) -> None:
    sns.set_theme(context="notebook", style="white")

    fig, axes = plt.subplots(len(plot_data), 2, figsize=(13, 6 * len(plot_data)))
    if len(plot_data) == 1:
        axes = [axes]

    for row_axes, (item_label, top_n, top_matrix, full_matrix) in zip(axes, plot_data):
        panels = [
            (row_axes[0], top_matrix, f"Top {top_n} {item_label}"),
            (row_axes[1], full_matrix, f"All {item_label}"),
        ]
        for ax, matrix, title in panels:
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
                cbar=ax is row_axes[-1],
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


def output_stem(metric: str, dataset: str, model: str, item_kind: str) -> str:
    base = f"{metric}_{dataset}_{model}_{'_'.join(MICROSTRUCTURES)}_{'_'.join(TASKS)}"
    if item_kind == "regions":
        return base

    return f"{base}_{item_kind}"

def plot_dataset_full_similarity_grid(
    data: dict[tuple[str, str], pd.DataFrame],
    dataset: str,
    metric_label: str,
    title_prefix: str,
    output_path: Path,
) -> None:
    """
    2×2 similarity grid for one dataset, formatted for thesis readability.

    The figure is intentionally compact but uses larger fonts, especially for
    heatmap annotations and tick labels, so the PDF remains readable after it is
    scaled down inside a thesis document.
    """
    sns.set_theme(context="paper", style="white")
    plt.rcParams.update({
        "pdf.fonttype": 42,      # keep text editable/searchable in the PDF
        "ps.fonttype": 42,
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    # Row order: ElasticNet on top, Group Lasso on bottom.
    row_models = ["region_elasticnet", "region_group_lasso"]
    # Column order: regions on left, networks on right.
    col_items = ["regions", "networks"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        constrained_layout=True,
    )

    for row_idx, model in enumerate(row_models):
        for col_idx, item_kind in enumerate(col_items):
            ax = axes[row_idx, col_idx]
            matrix = data.get((model, item_kind))

            if matrix is None or matrix.empty:
                ax.set_visible(False)
                continue

            # Only draw the colorbar on the rightmost column.
            draw_cbar = col_idx == len(col_items) - 1

            heatmap = sns.heatmap(
                matrix,
                ax=ax,
                cmap="mako",
                vmin=0,
                vmax=1,
                annot=True,
                annot_kws={"fontsize": 14, "fontweight": "bold"},
                fmt=".2f",
                square=True,
                linewidths=0.6,
                linecolor="white",
                cbar=draw_cbar,
                cbar_kws={"label": metric_label, "shrink": 0.85} if draw_cbar else {},
            )

            if draw_cbar and heatmap.collections:
                colorbar = heatmap.collections[0].colorbar
                colorbar.ax.tick_params(labelsize=12)
                colorbar.set_label(metric_label, fontsize=13)

            # Row label: model name, left column only.
            if col_idx == 0:
                ax.set_ylabel(
                    model_display_name(model),
                    fontsize=16,
                    fontweight="bold",
                    labelpad=12,
                )
            else:
                ax.set_ylabel("")

            # Column label: item kind, top row only.
            if row_idx == 0:
                ax.set_title(
                    f"All {item_kind}",
                    fontsize=16,
                    fontweight="bold",
                    pad=12,
                )
            else:
                ax.set_title("")

            ax.tick_params(axis="x", rotation=30, labelsize=13)
            ax.tick_params(axis="y", rotation=0, labelsize=13)

    fig.suptitle(
        f"Embedding {title_prefix} - {dataset}\n"
        f"{', '.join(TASKS)} | {', '.join(MICROSTRUCTURES)}",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2×2 similarity grid to {output_path}")
    
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
        ("jaccard", jaccard_matrix, "Weighted Jaccard similarity", "weighted Jaccard similarity"),
        ("cosine", cosine_matrix, "Cosine similarity", "cosine similarity"),
    ]

    for dataset in DATASETS:
        for metric_name, matrix_fn, metric_label, title_prefix in metric_configs:

            # ── existing per-model outputs (unchanged) ────────────────────
            for model in REGION_MODELS:
                plot_rows = []

                for item_kind, proportions, top_items, item_id_column, top_n in item_configs:
                    full_subset = proportions[
                        (proportions["dataset"] == dataset)
                        & (proportions["model"]   == model)
                    ]
                    top_subset = top_items[
                        (top_items["dataset"] == dataset)
                        & (top_items["model"]   == model)
                    ]
                    top_scores  = stable_item_scores(top_subset,  item_id_column)
                    full_scores = stable_item_scores(full_subset, item_id_column)

                    top_metric_matrix  = matrix_fn(top_scores,  REPRESENTATION_ORDER)
                    full_metric_matrix = matrix_fn(full_scores, REPRESENTATION_ORDER)

                    stem             = output_stem(metric_name, dataset, model, item_kind)
                    top_metric_matrix .to_csv(CSV_DIR / f"{stem}.csv")
                    full_metric_matrix.to_csv(CSV_DIR / f"{stem}_full_{item_kind}.csv")
                    top_subset        .to_csv(CSV_DIR / f"{stem}_top_{item_kind}.csv", index=False)

                    plot_rows.append((item_kind, top_n, top_metric_matrix, full_metric_matrix))

                plot_path = PLOTS_DIR / f"{output_stem(metric_name, dataset, model, 'combined')}.pdf"
                plot_model_similarity_grid(
                    plot_rows, dataset, model, plot_path,
                    metric_label=metric_label, title_prefix=title_prefix,
                )

            # ── new 2×2 grid: both models × both item kinds ───────────────
            grid_data: dict[tuple[str, str], pd.DataFrame] = {}

            for model in REGION_MODELS:
                for item_kind, proportions, _, item_id_column, _ in item_configs:
                    full_subset = proportions[
                        (proportions["dataset"] == dataset)
                        & (proportions["model"]   == model)
                    ]
                    full_scores = stable_item_scores(full_subset, item_id_column)
                    grid_data[(model, item_kind)] = matrix_fn(
                        full_scores, REPRESENTATION_ORDER
                    )

            grid_path = (
                PLOTS_DIR
                / f"{metric_name}_{dataset}_{'_'.join(MICROSTRUCTURES)}"
                  f"_{'_'.join(TASKS)}_2x2_grid.pdf"
            )
            plot_dataset_full_similarity_grid(
                data=grid_data,
                dataset=dataset,
                metric_label=metric_label,
                title_prefix=title_prefix,
                output_path=grid_path,
            )

    print(f"Rows after performance filtering: {len(selection_data):,}")
    print(f"Selected rows: {selection_data['selected'].sum():,}")


if __name__ == "__main__":
    main()
