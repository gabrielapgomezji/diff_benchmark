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
    model_display_name,
    network_selection_proportions,
    prepare_selection_data,
    region_selection_proportions,
)


DATASETS = ["hcp"]
TASKS = ["binary_classification"]
MICROSTRUCTURES = ["md"]
TOP_N_REGIONS = 10
TOP_N_NETWORKS = 5


def top_item_heatmap(
    proportions: pd.DataFrame,
    model: str,
    item_id_column: str,
    item_name_column: str,
) -> pd.DataFrame:
    model_data = proportions[proportions["model"] == model]
    item_columns = list(dict.fromkeys([item_id_column, item_name_column]))
    top_regions = (
        model_data.groupby(item_columns, as_index=False)["selection_proportion"]
        .mean()
        .sort_values(["selection_proportion", item_id_column], ascending=[False, True])
        .head(TOP_N_REGIONS)
    )

    heatmap = model_data[
        model_data[item_id_column].isin(top_regions[item_id_column])
    ].pivot_table(
        index=item_name_column,
        columns="region_representation",
        values="selection_proportion",
        aggfunc="mean",
    )

    region_order = top_regions[item_name_column].tolist()
    return heatmap.reindex(index=region_order, columns=REPRESENTATION_ORDER)


def output_stem(model: str, item_kind: str) -> str:
    if item_kind == "regions":
        return f"top10_regions_{model}_heatmap"

    return f"top10_{item_kind}_{model}_heatmap"


def plot_heatmap(
    heatmap: pd.DataFrame,
    model: str,
    output_path: Path,
    item_label: str,
) -> None:
    sns.set_theme(context="notebook", style="white")

    width = max(8, 0.9 * len(heatmap.columns))
    fig, ax = plt.subplots(figsize=(width, 7))
    # sns.heatmap(
    #     heatmap,
    #     ax=ax,
    #     cmap="mako",
    #     vmin=0,
    #     vmax=1,
    #     annot=True,
    #     annot_kws={"fontsize": 14},# "fontweight": "bold"},
    #     fmt=".2f",
    #     linewidths=0.4,
    #     linecolor="white",
    #     cbar_kws={"label": "Selection score"},
    # )
    hm = sns.heatmap(
        heatmap,
        ax=ax,
        cmap="mako",
        vmin=0,
        vmax=1,
        annot=True,
        annot_kws={"fontsize": 14},# "fontweight": "bold"},
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Selection score"},
    )
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=13)          # colorbar tick numbers
    cbar.set_label("Selection score", fontsize=16)  # colorbar title

    ax.set_title(
        f"Top 10 selected {item_label} - {model_display_name(model)}\n"
        f"{', '.join(DATASETS)} | {', '.join(TASKS)} | {', '.join(MICROSTRUCTURES)}",
        fontsize=18,
    )
    ax.set_xlabel("Region representation", fontsize=16)
    ax.set_ylabel(item_label.capitalize()[:-1], fontsize=16)
    ax.tick_params(axis="x", rotation=30, labelsize=13)
    ax.tick_params(axis="y", rotation=0, labelsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
        ("regions", region_proportions, "region_id", "region_name"),
        (
            "networks",
            network_selection_proportions(region_proportions),
            "network_name",
            "network_name",
        ),
    ]

    for item_kind, proportions, item_id_column, item_name_column in item_configs:
        for model in REGION_MODELS:
            heatmap = top_item_heatmap(
                proportions,
                model,
                item_id_column,
                item_name_column,
            )
            stem = output_stem(model, item_kind)
            csv_path = CSV_DIR / f"{stem}.csv"
            png_path = PLOTS_DIR / f"{stem}.pdf"
            heatmap.to_csv(csv_path)
            plot_heatmap(heatmap, model, png_path, item_kind)
            print(f"Saved {model} {item_kind} heatmap to {png_path}")
            print(f"Saved {model} {item_kind} heatmap data to {csv_path}")

    print(f"Rows after performance filtering: {len(selection_data):,}")
    print(f"Selected rows: {selection_data['selected'].sum():,}")


if __name__ == "__main__":
    main()
