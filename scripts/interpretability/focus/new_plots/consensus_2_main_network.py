import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
import numpy as np

from utils import (
    CSV_DIR,
    DATA_PATH,
    PLOTS_DIR,
    REGION_MODELS,
    REPRESENTATION_ORDER,
    cosine_matrix,
    ensure_output_dirs,
    model_display_name,
    main_network_selection_proportions,
    prepare_selection_data,
    region_selection_proportions,
    # network_ccc,
    # _pivot_network_scores
)


DATASETS = ["camcan", "hcp"]
TASKS = ["binary_classification"]
MICROSTRUCTURES = ["md"]
TOP_N_REGIONS = 20
HISTOGRAM_BINS = 20
selected_embedding = ["flatten", "summary_stats", "mean_std", "percentiles", "pca"]#REPRESENTATION_ORDER.copy() #["summary_stats", "mean_std", "percentiles", "pca", "flatten"]


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


def cosine_similarity_matrix(
    scores: pd.DataFrame,
    group_column: str,
    labels: list[str],
    item_column: str,
) -> pd.DataFrame:
    vectors = {
        label: (
            scores[scores[group_column] == label]
            .groupby(item_column)["average_score"]
            .mean()
        )
        for label in labels
    }
    return cosine_matrix(vectors, labels)


def plot_network_cosine_heatmap(network_scores: pd.DataFrame) -> None:
    sns.set_theme(context="notebook", style="white")

    combo_specs = [
        ("hcp", "region_group_lasso"),
        ("hcp", "region_elasticnet"),
        ("camcan", "region_group_lasso"),
        ("camcan", "region_elasticnet"),
    ]
    labels = [f"{dataset.upper()}::{model_display_name(model)}" for dataset, model in combo_specs]

    vectors = {
        label: (
            network_scores[
                (network_scores["dataset"] == dataset)
                & (network_scores["model"] == model)
            ]
            .groupby("network_name")["average_score"]
            .mean()
        )
        for label, (dataset, model) in zip(labels, combo_specs)
    }
    matrix = cosine_matrix(vectors, labels)

    fig, ax = plt.subplots(figsize=(9, 8))
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
        cbar_kws={"label": "Cosine similarity"},
    )
    ax.set_title("Network cosine similarity")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    fig.suptitle(
        f"Network cosine similarity across HCP and CamCAN | Group Lasso and Elasticnet\n"
        f"{', '.join(TASKS)} | {', '.join(MICROSTRUCTURES)} | embeddings: {', '.join(selected_embedding)}"
    )
    fig.tight_layout()
    output_path = PLOTS_DIR / "consensus2_main_network_cosine_similarity_heatmaps.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved network cosine similarity heatmaps to {output_path}")


def plot_network_barplot(network_scores: pd.DataFrame) -> None:
    sns.set_theme(context="notebook", style="whitegrid")

    pair_order = [
        ("hcp", "region_group_lasso"),
        ("hcp", "region_elasticnet"),
        ("camcan", "region_group_lasso"),
        ("camcan", "region_elasticnet"),
    ]
    pair_labels = {
        (dataset, model): f"{dataset.upper()}::{model_display_name(model)}"
        for dataset, model in pair_order
    }
    network_order = (
        network_scores[
            (network_scores["dataset"] == "hcp")
            & (network_scores["model"] == "region_group_lasso")
        ]
        .sort_values(["average_score", "network_name"], ascending=[False, True])["network_name"]
        .tolist()
    )

    plot_data = network_scores.copy()
    plot_data["dataset_model"] = plot_data.apply(
        lambda row: pair_labels[(row["dataset"], row["model"])], axis=1
    )

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.barplot(
        data=plot_data,
        x="network_name",
        y="average_score",
        hue="dataset_model",
        order=network_order,
        hue_order=[pair_labels[pair] for pair in pair_order],
        palette="mako",
        ax=ax,
    )

    ax.set_title(
        f"Network consensus scores by dataset and model\n"
        f"Ordered by HCP::Group Lasso | {', '.join(TASKS)} | {', '.join(MICROSTRUCTURES)} | embeddings: {', '.join(selected_embedding)}"
    )
    ax.set_xlabel("Network")
    ax.set_ylabel("Average consensus score")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Dataset::Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    output_path = PLOTS_DIR / "consensus2_main_network_barplot.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved network bar plot to {output_path}")


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
        output_path = PLOTS_DIR / "consensus2_main_average_score_histogram.pdf"
    else:
        output_path = PLOTS_DIR / f"consensus2_main_{item_label}_average_score_histogram.pdf"
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
        output_path = PLOTS_DIR / "consensus2_main_top20_parcels.pdf"
    else:
        output_path = PLOTS_DIR / f"consensus2_main_{item_label}_top20.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved top consensus {item_label} plot to {output_path}")

def plot_network_scores_scatter(
    network_scores: pd.DataFrame,
    linked: bool = False,
) -> plt.Figure:
    """
    Two-panel scatter of network consensus scores.

    Panel 1  X = Group Lasso score,  Y = ElasticNet score
             ○ = HCP,  □ = CamCAN,  colour = network

    Panel 2  X = HCP score,  Y = CamCAN score
             ○ = Group Lasso,  □ = ElasticNet,  colour = network

    Parameters
    ----------
    network_scores : DataFrame
        Consensus scores with columns
        [dataset, model, network_name, average_score].
        Typically the output of ``consensus_scores()`` for networks.
    linked : bool
        If True, draw a dashed line between the two markers (○ and □)
        of the same network within each panel.
    """
    sns.set_theme(context="notebook", style="white")

    networks     = sorted(network_scores["network_name"].unique())
    palette      = sns.color_palette("tab20", n_colors=len(networks))
    color_map    = dict(zip(networks, palette))

    pivot_models, pivot_datasets = _pivot_network_scores(network_scores)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    global_max = network_scores["average_score"].max()
    ax_lim     = [0, max(global_max * 1.08, 0.05)]

    # ── Panel 1: Group Lasso (x) vs ElasticNet (y) ──────────────────────────
    ax = axes[0]

    dataset_cfg = {
        "hcp":    ("o", "HCP"),
        "camcan": ("s", "CamCAN"),
    }

    for dataset, (marker, _label) in dataset_cfg.items():
        sub = pivot_models[pivot_models["dataset"] == dataset]
        for _, row in sub.iterrows():
            net = row["network_name"]
            x   = row.get("region_group_lasso", np.nan)
            y   = row.get("region_elasticnet",  np.nan)
            ax.scatter(x, y,
                       color=color_map[net], marker=marker,
                       s=80, zorder=3, linewidths=0.8,
                       edgecolors="white" if marker == "o" else color_map[net])

    if linked:
        for net in networks:
            sub   = pivot_models[pivot_models["network_name"] == net]
            r_hcp = sub[sub["dataset"] == "hcp"]
            r_cc  = sub[sub["dataset"] == "camcan"]
            if r_hcp.empty or r_cc.empty:
                continue
            ax.plot(
                [r_hcp["region_group_lasso"].iat[0], r_cc["region_group_lasso"].iat[0]],
                [r_hcp["region_elasticnet"].iat[0],  r_cc["region_elasticnet"].iat[0]],
                "--", color=color_map[net], alpha=0.45, linewidth=1, zorder=2,
            )

    ax.plot(ax_lim, ax_lim, "--", color="grey", alpha=0.35, linewidth=1)
    ax.set_xlim(ax_lim); ax.set_ylim(ax_lim)
    ax.set_xlabel("Group Lasso — average consensus score")
    ax.set_ylabel("ElasticNet — average consensus score")
    ax.set_title("Estimator comparison\n(○ HCP  □ CamCAN)")

    # ── Panel 2: HCP (x) vs CamCAN (y) ──────────────────────────────────────
    ax = axes[1]

    model_cfg = {
        "region_group_lasso": ("o", "Group Lasso"),
        "region_elasticnet":  ("s", "ElasticNet"),
    }

    for model, (marker, _label) in model_cfg.items():
        sub = pivot_datasets[pivot_datasets["model"] == model]
        for _, row in sub.iterrows():
            net = row["network_name"]
            x   = row.get("hcp",    np.nan)
            y   = row.get("camcan", np.nan)
            ax.scatter(x, y,
                       color=color_map[net], marker=marker,
                       s=80, zorder=3, linewidths=0.8,
                       edgecolors="white" if marker == "o" else color_map[net])

    if linked:
        for net in networks:
            sub  = pivot_datasets[pivot_datasets["network_name"] == net]
            r_gl = sub[sub["model"] == "region_group_lasso"]
            r_en = sub[sub["model"] == "region_elasticnet"]
            if r_gl.empty or r_en.empty:
                continue
            ax.plot(
                [r_gl["hcp"].iat[0],    r_en["hcp"].iat[0]],
                [r_gl["camcan"].iat[0], r_en["camcan"].iat[0]],
                "--", color=color_map[net], alpha=0.45, linewidth=1, zorder=2,
            )

    ax.plot(ax_lim, ax_lim, "--", color="grey", alpha=0.35, linewidth=1)
    ax.set_xlim(ax_lim); ax.set_ylim(ax_lim)
    ax.set_xlabel("HCP — average consensus score")
    ax.set_ylabel("CamCAN — average consensus score")
    ax.set_title("Dataset comparison\n(○ Group Lasso  □ ElasticNet)")

    # ── Shared legend ────────────────────────────────────────────────────────
    network_handles = [
        mpatches.Patch(color=color_map[net], label=net) for net in networks
    ]
    marker_handles = [
        mlines.Line2D([], [], color="dimgrey", marker="o", linestyle="None",
                      markersize=7, label="HCP / Group Lasso"),
        mlines.Line2D([], [], color="dimgrey", marker="s", linestyle="None",
                      markersize=7, label="CamCAN / ElasticNet"),
    ]
    if linked:
        marker_handles.append(
            mlines.Line2D([], [], color="dimgrey", linestyle="--",
                          linewidth=1, alpha=0.7, label="same network (linked)")
        )

    fig.legend(
        handles=network_handles + [mpatches.Patch(color="none", label="")] + marker_handles,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        title="Network / Marker",
        fontsize=8,
        title_fontsize=9,
    )

    suffix = "linked" if linked else "plain"
    fig.suptitle(
        f"Network consensus scores — estimator & dataset comparisons ({suffix})\n"
        f"{', '.join(TASKS)} | {', '.join(MICROSTRUCTURES)} | "
        f"embeddings: {', '.join(selected_embedding)}"
    )

    return fig
# def plot_network_ccc_linked(ccc_df):

#     networks = sorted(ccc_df["network"].unique())

#     palette = sns.color_palette("tab20", n_colors=len(networks))
#     color_map = dict(zip(networks, palette))

#     fig, axes = plt.subplots(
#         1,
#         2,
#         figsize=(13, 6),
#         constrained_layout=True,
#     )

#     ax = axes[0]

#     for network in networks:
#         row = ccc_df.loc[ccc_df["network"] == network].iloc[0]

#         x = row["hcp_models"]
#         y = row["camcan_models"]

#         ax.plot(
#             [x, x],
#             [y, y],
#             "--",
#             color=color_map[network],
#             alpha=0.6,
#         )

#         ax.scatter(
#             x,
#             y,
#             marker="o",
#             s=100,
#             color=color_map[network],
#             label=network,
#         )

#         ax.scatter(
#             x,
#             y,
#             marker="s",
#             s=100,
#             facecolors="none",
#             edgecolors=color_map[network],
#             linewidth=2,
#         )

#     ax.plot([0, 1], [0, 1], "k--", alpha=0.4)

#     ax.set_xlabel("HCP CCC (GL ↔ EN)")
#     ax.set_ylabel("CamCAN CCC (GL ↔ EN)")
#     ax.set_title("Across models")

#     ax = axes[1]

#     for network in networks:
#         row = ccc_df.loc[ccc_df["network"] == network].iloc[0]

#         x = row["gl_datasets"]
#         y = row["en_datasets"]

#         ax.plot(
#             [x, x],
#             [y, y],
#             "--",
#             color=color_map[network],
#             alpha=0.6,
#         )

#         ax.scatter(
#             x,
#             y,
#             marker="o",
#             s=100,
#             color=color_map[network],
#         )

#         ax.scatter(
#             x,
#             y,
#             marker="s",
#             s=100,
#             facecolors="none",
#             edgecolors=color_map[network],
#             linewidth=2,
#         )

#     ax.plot([0, 1], [0, 1], "k--", alpha=0.4)

#     ax.set_xlabel("Group Lasso CCC (HCP ↔ CamCAN)")
#     ax.set_ylabel("ElasticNet CCC (HCP ↔ CamCAN)")
#     ax.set_title("Across datasets")

#     handles, labels = axes[0].get_legend_handles_labels()

#     fig.legend(
#         handles,
#         labels,
#         bbox_to_anchor=(1.03, 0.5),
#         loc="center left",
#         title="Network",
#     )

#     return fig
from scipy.stats import kendalltau as scipy_kendalltau


# ── 1. Compute Kendall tau per (model, embedding) ────────────────────────────

def network_kendall_tau(network_proportions: pd.DataFrame) -> pd.DataFrame:
    """
    For every (model, embedding) combination, rank the networks by their
    selection proportion separately for HCP and CamCAN, then compute
    Kendall's tau between those two rankings.

    Parameters
    ----------
    network_proportions : DataFrame
        Output of ``network_selection_proportions()``.
        Must contain: dataset, model, region_representation,
                      network_name, selection_proportion.
        Do NOT pass the consensus ``scores`` DataFrame — that one has
        already averaged out ``region_representation``.

    Returns
    -------
    DataFrame with columns:
        model, region_representation, kendall_tau, p_value, n_networks
    Sorted by model then embedding, pivotable into a model × embedding table.
    """
    results = []

    for (model, embedding), group in network_proportions.groupby(
        ["model", "region_representation"]
    ):
        hcp_vec = (
            group[group["dataset"] == "hcp"]
            .set_index("network_name")["selection_proportion"]
        )
        camcan_vec = (
            group[group["dataset"] == "camcan"]
            .set_index("network_name")["selection_proportion"]
        )

        shared_networks = hcp_vec.index.intersection(camcan_vec.index)
        n = len(shared_networks)

        if n < 3:
            tau, pval = np.nan, np.nan
        else:
            tau, pval = scipy_kendalltau(
                hcp_vec.loc[shared_networks].to_numpy(),
                camcan_vec.loc[shared_networks].to_numpy(),
            )

        results.append({
            "model":                 model,
            "region_representation": embedding,
            "kendall_tau":           round(tau,  4) if not np.isnan(tau)  else np.nan,
            "p_value":               round(pval, 4) if not np.isnan(pval) else np.nan,
            "n_networks":            n,
        })

    return (
        pd.DataFrame(results)
        .sort_values(["model", "region_representation"])
        .reset_index(drop=True)
    )


# ── 2. Render as a coloured matplotlib table ─────────────────────────────────

def plot_network_kendall_table(kendall_df: pd.DataFrame) -> plt.Figure:
    """
    Render a model × embedding pivot of Kendall tau values as a
    colour-coded table figure.

    Cells are coloured on a Red→Yellow→Green scale mapped to [−1, 1].
    Significance stars are appended to each cell:
        *** p < 0.001  ** p < 0.01  * p < 0.05  (ns) otherwise.
    """
    # ── pivot tau and p-value separately ────────────────────────────────────
    tau_pivot = kendall_df.pivot(
        index="model",
        columns="region_representation",
        values="kendall_tau",
    )
    pval_pivot = kendall_df.pivot(
        index="model",
        columns="region_representation",
        values="p_value",
    )

    # pretty row labels
    tau_pivot.index   = [model_display_name(m) for m in tau_pivot.index]
    pval_pivot.index  = tau_pivot.index

    # consistent column order
    col_order = [e for e in selected_embedding if e in tau_pivot.columns]
    tau_pivot  = tau_pivot[col_order]
    pval_pivot = pval_pivot[col_order]

    def stars(p):
        if np.isnan(p):   return ""
        if p < 0.001:     return "***"
        if p < 0.01:      return "**"
        if p < 0.05:      return "*"
        return "(ns)"

    # build cell text: "0.83***"
    cell_text = [
        [
            f"{tau_pivot.iloc[r, c]:.3f}{stars(pval_pivot.iloc[r, c])}"
            if not np.isnan(tau_pivot.iloc[r, c]) else "—"
            for c in range(tau_pivot.shape[1])
        ]
        for r in range(tau_pivot.shape[0])
    ]

    n_rows, n_cols = tau_pivot.shape
    fig, ax = plt.subplots(figsize=(n_cols * 2.0 + 2.5, n_rows * 0.75 + 2.0))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=tau_pivot.index.tolist(),
        colLabels=tau_pivot.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 1.8)

    # colour cells: header/index rows get a dark background; data cells by tau
    cmap = plt.cm.RdYlGn
    header_colour = "#2e2e2e"
    header_text   = "white"

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)

        is_header = row_idx == 0
        is_index  = col_idx == -1

        if is_header or is_index:
            cell.set_facecolor(header_colour)
            cell.get_text().set_color(header_text)
            cell.get_text().set_weight("bold")
        else:
            tau_val = tau_pivot.iloc[row_idx - 1, col_idx]
            if np.isnan(tau_val):
                cell.set_facecolor("#f0f0f0")
            else:
                intensity = (tau_val + 1.0) / 2.0   # map [-1, 1] → [0, 1]
                cell.set_facecolor(cmap(intensity))
                # darken text on very bright cells
                cell.get_text().set_color(
                    "black" if 0.25 < intensity < 0.85 else "white"
                )

    ax.set_title(
        f"Kendall τ — network selection ranking: CamCAN vs HCP\n"
        f"{', '.join(TASKS)} | {', '.join(MICROSTRUCTURES)} | "
        f"embeddings: {', '.join(selected_embedding)}\n"
        f"Significance: *** p<0.001  ** p<0.01  * p<0.05  (ns) p≥0.05",
        fontsize=10,
        pad=16,
    )
    fig.tight_layout()
    return fig

def network_global_kendall_tau(network_proportions: pd.DataFrame) -> dict:
    """
    Compute a single Kendall tau comparing the global network ordering
    between HCP and CamCAN, where the global order is obtained by
    averaging selection_proportion across ALL models and embeddings
    for each (dataset, network_name) pair.

    Parameters
    ----------
    network_proportions : DataFrame
        Output of ``network_selection_proportions()``.
        Must contain: dataset, model, region_representation,
                      network_name, selection_proportion.

    Returns
    -------
    dict with keys:
        global_scores   : DataFrame with columns
                          [network_name, hcp_score, camcan_score,
                           hcp_rank, camcan_rank]
                          sorted by hcp_rank
        kendall_tau     : float
        p_value         : float
        n_networks      : int
    """
    # ── 1. Global score per (dataset, network): average over all
    #       models × embeddings ──────────────────────────────────
    global_scores = (
        network_proportions
        .groupby(["dataset", "network_name"], as_index=False)
        .agg(global_score=("selection_proportion", "mean"))
    )

    hcp_scores = (
        global_scores[global_scores["dataset"] == "hcp"]
        .set_index("network_name")["global_score"]
    )
    camcan_scores = (
        global_scores[global_scores["dataset"] == "camcan"]
        .set_index("network_name")["global_score"]
    )

    # ── 2. Align on shared networks ───────────────────────────────
    shared = hcp_scores.index.intersection(camcan_scores.index)
    n = len(shared)

    hcp_aligned    = hcp_scores.loc[shared]
    camcan_aligned = camcan_scores.loc[shared]

    # ── 3. Kendall tau on the raw scores (scipy ranks internally) ─
    if n < 3:
        tau, pval = np.nan, np.nan
    else:
        tau, pval = scipy_kendalltau(
            hcp_aligned.to_numpy(),
            camcan_aligned.to_numpy(),
        )

    # ── 4. Build a readable summary table ─────────────────────────
    summary = (
        pd.DataFrame({
            "network_name":  shared,
            "hcp_score":     hcp_aligned.values,
            "camcan_score":  camcan_aligned.values,
        })
        .assign(
            hcp_rank    = lambda d: d["hcp_score"].rank(ascending=False).astype(int),
            camcan_rank = lambda d: d["camcan_score"].rank(ascending=False).astype(int),
        )
        .sort_values("hcp_rank")
        .reset_index(drop=True)
    )

    return {
        "global_scores": summary,
        "kendall_tau":   tau,
        "p_value":       pval,
        "n_networks":    n,
    }


# def print_global_kendall_summary(result: dict) -> None:
#     """
#     Pretty-print the global Kendall tau result to stdout.
#     """
#     tau  = result["kendall_tau"]
#     pval = result["p_value"]
#     n    = result["n_networks"]

#     if pval < 0.001:  sig = "***"
#     elif pval < 0.01: sig = "**"
#     elif pval < 0.05: sig = "*"
#     else:             sig = "(ns)"

#     print("\n" + "═" * 58)
#     print("  Global network ranking concordance: HCP vs CamCAN")
#     print("  (averaged across all models and embedding representations)")
#     print("═" * 58)
#     # print(f"  n networks  : {n}")
#     # print(f"  Kendall τ   : {tau:.4f}")
#     # print(f"  p-value     : {pval:.4f}  {sig}")
#     # print("═" * 58)

#     # per-network rank comparison
#     df = result["global_scores"].copy()
#     df["hcp_score"]    = df["hcp_score"].map("{:.4f}".format)
#     df["camcan_score"] = df["camcan_score"].map("{:.4f}".format)
#     df["rank_delta"]   = (
#         result["global_scores"]["hcp_rank"]
#         - result["global_scores"]["camcan_rank"]
#     ).map(lambda d: f"{d:+d}")

#     print(
#         df[["network_name", "hcp_score", "hcp_rank",
#             "camcan_score", "camcan_rank", "rank_delta"]]
#         .to_string(index=False)
#     )
#     print("═" * 58 + "\n")
def print_global_kendall_summary(result: dict) -> None:
    """
    Pretty-print the global Kendall tau result to stdout,
    and save the per-network rank table as CSV and LaTeX.
    """
    tau  = result["kendall_tau"]
    pval = result["p_value"]
    n    = result["n_networks"]

    if pval < 0.001:  sig = "***"
    elif pval < 0.01: sig = "**"
    elif pval < 0.05: sig = "*"
    else:             sig = "(ns)"

    # ── build the display DataFrame ──────────────────────────────────────────
    df = result["global_scores"].copy()
    df["rank_delta"] = (
        result["global_scores"]["hcp_rank"]
        - result["global_scores"]["camcan_rank"]
    ).map(lambda d: f"{d:+d}")
    df["hcp_score"]    = df["hcp_score"].map("{:.4f}".format)
    df["camcan_score"] = df["camcan_score"].map("{:.4f}".format)

    display_cols = [
        "network_name", "hcp_score", "hcp_rank",
        "camcan_score", "camcan_rank", "rank_delta",
    ]
    table = df[display_cols].copy()

    # ── stdout ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  Global network ranking concordance: HCP vs CamCAN")
    print("  (averaged across all models and embedding representations)")
    print("═" * 58)
    print(table.to_string(index=False))
    print("═" * 58 + "\n")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = CSV_DIR / "consensus2_networks_global_kendall_table.csv"
    table.to_csv(csv_path, index=False)
    print(f"Saved global Kendall table CSV to {csv_path}")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    # Pretty column headers for the paper
    latex_table = table.rename(columns={
        "network_name": "Network",
        "hcp_score":    "HCP score",
        "hcp_rank":     "HCP rank",
        "camcan_score": "CamCAN score",
        "camcan_rank":  "CamCAN rank",
        "rank_delta":   r"$\Delta$ rank",
    })

    caption = (
        f"Global network selection ranking concordance between HCP and CamCAN "
        f"(averaged across all models and embedding representations). "
        f"Kendall $\\tau = {tau:.4f}$, $p = {pval:.4f}$ ({sig}), "
        f"$n = {n}$ networks."
    )

    latex_str = latex_table.to_latex(
        index=False,
        escape=False,           # keep \tau, \Delta as-is
        caption=caption,
        label="tab:global_kendall",
        column_format="l" + "c" * (len(latex_table.columns) - 1),
        position="ht",
    )

    latex_path = CSV_DIR / "consensus2_main_networks_global_kendall_table.tex"
    latex_path.write_text(latex_str, encoding="utf-8")
    print(f"Saved global Kendall table LaTeX to {latex_path}")

def plot_score_histogram_per_embedding(
    proportions: pd.DataFrame,
    item_label: str,
    item_id_column: str,
) -> None:
    """
    One figure per model.
    Grid: rows = datasets, columns = embeddings.
    Shows the distribution of selection_proportion across items.
    """
    sns.set_theme(context="notebook", style="whitegrid")

    embeddings = [e for e in selected_embedding
                  if e in proportions["region_representation"].unique()]

    for model in REGION_MODELS:
        fig, axes = plt.subplots(
            len(DATASETS),
            len(embeddings),
            figsize=(len(embeddings) * 3.5, len(DATASETS) * 3.5),
            sharex=True,
            sharey=True,
        )
        # ensure axes is always 2-D
        if len(DATASETS) == 1:
            axes = axes[np.newaxis, :]
        if len(embeddings) == 1:
            axes = axes[:, np.newaxis]

        for row, dataset in enumerate(DATASETS):
            for col, embedding in enumerate(embeddings):
                ax = axes[row, col]
                subset = proportions[
                    (proportions["dataset"] == dataset)
                    & (proportions["model"] == model)
                    & (proportions["region_representation"] == embedding)
                ]
                ax.hist(
                    subset["selection_proportion"],
                    bins=HISTOGRAM_BINS,
                    range=(0, 1),
                    color="#3a7ca5",
                    edgecolor="white",
                    linewidth=0.8,
                )
                if row == 0:
                    ax.set_title(embedding, fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{dataset}\nCount", fontsize=8)
                ax.set_xlim(0, 1)
                ax.tick_params(labelsize=7)

        fig.supxlabel("Selection proportion", fontsize=9)
        fig.suptitle(
            f"Per-embedding selection proportion — {item_label}\n"
            f"{model_display_name(model)} | "
            f"{', '.join(TASKS)} | {', '.join(MICROSTRUCTURES)}",
            fontsize=10,
        )
        fig.tight_layout()
        output_path = (
            PLOTS_DIR
            / f"per_embedding_histogram_main_{item_label}_{model}.pdf"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved per-embedding histogram to {output_path}")

# ── Subnetwork → main Yeo network mapping ────────────────────────────────────
# Keys must match the prefix of the network_name values in your data
# (e.g. "Default_A", "Default_B", "DorsAttn_A", "Vis_1" …)
SUBNETWORK_TO_MAIN_NETWORK = {
    "Vis":         "Visual",
    "SomMot":      "SomatoMotor",
    "DorsAttn":    "Dorsal Attention",
    "SalVentAttn": "Ventral Attention/Salience",
    "VentAttn":    "Ventral Attention/Salience",   # alternative label
    "Limbic":      "Limbic",
    "Cont":        "Frontoparietal/Control",
    "Default":     "Default",
}

MAIN_NETWORK_ORDER = [
    "Visual",
    "SomatoMotor",
    "Dorsal Attention",
    "Ventral Attention/Salience",
    "Limbic",
    "Frontoparietal/Control",
    "Default",
]


def _map_to_main_network(subnetwork_name: str) -> str | None:
    """
    Match a subnetwork label (e.g. 'Default_A', 'DorsAttn_B')
    to one of the 7 main Yeo networks by checking which key
    the name starts with.  Returns None if no match found.
    """
    for prefix, main in SUBNETWORK_TO_MAIN_NETWORK.items():
        if subnetwork_name.startswith(prefix):
            return main
    return None


def main_network_rank_table(network_scores: pd.DataFrame) -> pd.DataFrame:
    """
    For each dataset, aggregate subnetwork consensus scores into the
    7 main Yeo networks and compute score, variability, and rank.

    Parameters
    ----------
    network_scores : DataFrame
        Output of ``consensus_scores()`` for networks.
        Must contain: dataset, model, network_name, average_score.
        ``network_name`` must be a subnetwork label whose prefix maps
        to one of the 7 main Yeo networks via SUBNETWORK_TO_MAIN_NETWORK.

    Returns
    -------
    DataFrame with columns:
        main_network,
        <dataset>_score, <dataset>_std, <dataset>_n_subnetworks,
            <dataset>_rank   — repeated for each dataset
    Rows are the 7 main networks, sorted by MAIN_NETWORK_ORDER.
    """
    # ── 1. Map subnetworks to main networks ──────────────────────────────────
    df = network_scores.copy()
    df["main_network"] = df["network_name"].map(_map_to_main_network)

    unmapped = df["main_network"].isna()
    if unmapped.any():
        missing = df.loc[unmapped, "network_name"].unique().tolist()
        print(
            f"[WARNING] {len(missing)} subnetwork(s) could not be mapped "
            f"to a main network and will be excluded:\n  {missing}"
        )
    df = df.dropna(subset=["main_network"])

    # ── 2. Aggregate across models: one score per
    #       (dataset, network_name=subnetwork) ─────────────────────────────
    # consensus_scores already averaged over embeddings; here we also
    # average over models so variability reflects subnetwork spread only
    sub_scores = (
        df.groupby(["dataset", "main_network", "network_name"], as_index=False)
        .agg(subnetwork_score=("average_score", "mean"))   # mean over models
    )

    # ── 3. Aggregate subnetworks → main network per dataset ──────────────────
    agg = (
        sub_scores.groupby(["dataset", "main_network"])
        .agg(
            score          = ("subnetwork_score", "mean"),
            std            = ("subnetwork_score", "std"),
            n_subnetworks  = ("subnetwork_score", "count"),
        )
        .reset_index()
    )
    # std is NaN when only one subnetwork — replace with 0
    agg["std"] = agg["std"].fillna(0.0)

    # ── 4. Rank within each dataset (rank 1 = highest score) ─────────────────
    agg["rank"] = (
        agg.groupby("dataset")["score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # ── 5. Pivot to wide format: one column group per dataset ─────────────────
    datasets = sorted(agg["dataset"].unique())
    pieces = []
    for dataset in datasets:
        sub = (
            agg[agg["dataset"] == dataset]
            .set_index("main_network")
            [["score", "std", "n_subnetworks", "rank"]]
            .rename(columns={
                "score":         f"{dataset}_score",
                "std":           f"{dataset}_std",
                "n_subnetworks": f"{dataset}_n_subnetworks",
                "rank":          f"{dataset}_rank",
            })
        )
        pieces.append(sub)

    wide = pd.concat(pieces, axis=1).reset_index().rename(
        columns={"main_network": "Network"}
    )

    # enforce canonical network order; add any unexpected networks at the end
    ordered = [n for n in MAIN_NETWORK_ORDER if n in wide["Network"].values]
    extras  = [n for n in wide["Network"].values if n not in MAIN_NETWORK_ORDER]
    wide = (
        wide.set_index("Network")
        .loc[ordered + extras]
        .reset_index()
    )

    return wide, sub_scores   # also return sub_scores for the plot


def save_main_network_rank_table(
    table: pd.DataFrame,
    datasets: list[str],
) -> None:
    """
    Save the main network rank table as CSV and LaTeX.
    """
    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = CSV_DIR / "consensus2_main_network_rank_table.csv"
    table.to_csv(csv_path, index=False)
    print(f"Saved main network rank table CSV to {csv_path}")

    # ── LaTeX: round numerics, build pretty column names ─────────────────────
    latex_table = table.copy()

    col_rename = {"Network": "Network"}
    for dataset in datasets:
        col_rename[f"{dataset}_score"]        = f"{dataset.upper()} score"
        col_rename[f"{dataset}_std"]          = f"{dataset.upper()} SD"
        col_rename[f"{dataset}_n_subnetworks"]= r"$n_{\text{sub}}$"
        col_rename[f"{dataset}_rank"]         = f"{dataset.upper()} rank"

    # format floats
    for dataset in datasets:
        latex_table[f"{dataset}_score"] = latex_table[f"{dataset}_score"].map("{:.3f}".format)
        latex_table[f"{dataset}_std"]   = latex_table[f"{dataset}_std"].map("{:.3f}".format)

    latex_table = latex_table.rename(columns=col_rename)

    n_cols   = len(latex_table.columns)
    col_fmt  = "l" + "c" * (n_cols - 1)
    caption  = (
        "Main Yeo-network consensus scores, within-network subnetwork "
        "variability (SD), and ranking for each dataset. "
        "Score = mean consensus score averaged over all models and embedding "
        "representations; SD = standard deviation across subnetworks "
        "belonging to each main network; rank 1 = most selected."
    )

    latex_str = latex_table.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label="tab:main_network_ranks",
        column_format=col_fmt,
        position="ht",
    )

    latex_path = CSV_DIR / "consensus2_main_network_rank_table.tex"
    latex_path.write_text(latex_str, encoding="utf-8")
    print(f"Saved main network rank table LaTeX to {latex_path}")


def plot_main_network_rank_table(
    table: pd.DataFrame,
    sub_scores: pd.DataFrame,
    datasets: list[str],
) -> plt.Figure:
    """
    Render the rank table as a colour-coded matplotlib figure.
    Score cells are coloured low→high (mako).
    Rank cells are coloured 1=dark (best) → 7=light.
    SD cells are coloured low→high (Oranges: low variability = good).
    Subnetwork score distributions shown as strip plots alongside.
    """
    sns.set_theme(context="notebook", style="white")

    networks = table["Network"].tolist()
    n_nets   = len(networks)

    # one panel per dataset: left = coloured table, right = strip plot
    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(len(datasets) * 9, n_nets * 0.65 + 2.5),
        constrained_layout=True,
    )
    if len(datasets) == 1:
        axes = [axes]

    score_cmap = plt.cm.get_cmap("mako")
    sd_cmap    = plt.cm.get_cmap("Oranges_r")   # low SD = darker (better)

    for ax, dataset in zip(axes, datasets):
        ax.axis("off")

        score_col = f"{dataset}_score"
        std_col   = f"{dataset}_std"
        rank_col  = f"{dataset}_rank"
        nsub_col  = f"{dataset}_n_subnetworks"

        col_headers = ["Network", "Score", "SD", "Rank", "n sub"]
        col_data    = [
            table["Network"],
            table[score_col].map("{:.3f}".format),
            table[std_col].map("{:.3f}".format),
            table[rank_col].astype(str),
            table[nsub_col].astype(str),
        ]

        cell_text = list(zip(*[c.tolist() for c in col_data]))
        n_rows = len(cell_text)
        n_cols = len(col_headers)

        tbl = ax.table(
            cellText=cell_text,
            colLabels=col_headers,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.4, 2.0)

        score_vals = table[score_col].to_numpy(dtype=float)
        std_vals   = table[std_col].to_numpy(dtype=float)
        rank_vals  = table[rank_col].to_numpy(dtype=int)

        score_min, score_max = score_vals.min(), score_vals.max()
        std_min,   std_max   = std_vals.min(),   std_vals.max()

        header_colour = "#1a1a2e"

        for (ri, ci), cell in tbl.get_celld().items():
            cell.set_edgecolor("#dddddd")
            cell.set_linewidth(0.5)

            if ri == 0:                          # header row
                cell.set_facecolor(header_colour)
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
                continue

            data_row = ri - 1                    # 0-based index into table

            if ci == 0:                          # network name column
                cell.set_facecolor("#f5f5f5")
                cell.get_text().set_weight("bold")
                cell.get_text().set_color("#1a1a2e")

            elif ci == 1:                        # score column → mako
                norm = (
                    (score_vals[data_row] - score_min) / (score_max - score_min)
                    if score_max > score_min else 0.5
                )
                colour = score_cmap(norm)
                cell.set_facecolor(colour)
                cell.get_text().set_color("white" if norm > 0.55 else "black")

            elif ci == 2:                        # SD column → Oranges_r
                norm = (
                    (std_vals[data_row] - std_min) / (std_max - std_min)
                    if std_max > std_min else 0.5
                )
                colour = sd_cmap(norm)
                cell.set_facecolor(colour)
                cell.get_text().set_color("black")

            elif ci == 3:                        # rank column → grey scale
                norm  = 1.0 - (rank_vals[data_row] - 1) / max(n_nets - 1, 1)
                grey  = 0.85 - 0.55 * norm       # rank 1 → dark, rank 7 → light
                cell.set_facecolor((grey, grey, grey))
                cell.get_text().set_color("white" if grey < 0.5 else "black")
                cell.get_text().set_weight("bold")

            else:                                # n_sub column → plain
                cell.set_facecolor("white")

        ax.set_title(
            dataset.upper(),
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

    fig.suptitle(
        f"Main Yeo-network consensus ranking — score & subnetwork variability\n"
        f"{', '.join(TASKS)} | {', '.join(MICROSTRUCTURES)} | "
        f"embeddings: {', '.join(selected_embedding)}",
        fontsize=11,
    )
    return fig

def plot_top_items_single(
    scores: pd.DataFrame,
    score_column: str,
    item_label: str,
    item_id_column: str,
    item_name_column: str,
    suptitle: str,
    output_path,
) -> None:
    """
    Core plotting logic shared by plot_top_items and
    plot_top_items_per_embedding.

    Parameters
    ----------
    scores : DataFrame
        Must contain: dataset, model, item_id_column,
                      item_name_column, score_column.
    score_column : str
        Column to use as the bar length (average_score or
        selection_proportion).
    """
    sns.set_theme(context="notebook", style="whitegrid")

    fig, axes = plt.subplots(
        len(DATASETS),
        len(REGION_MODELS),
        figsize=(15, 12),
        sharex=True,
    )

    global_max = scores[score_column].max()

    for row, dataset in enumerate(DATASETS):
        for col, model in enumerate(REGION_MODELS):
            ax = axes[row, col]
            subset = scores[
                (scores["dataset"] == dataset)
                & (scores["model"]   == model)
            ]
            top_items = (
                subset
                .sort_values(
                    [score_column, item_id_column],
                    ascending=[False, True],
                )
                .head(TOP_N_REGIONS)
                .sort_values(score_column)          # ascending for barh
            )
            ax.barh(
                top_items[item_name_column],
                top_items[score_column],
                color="#7b8c3a",
            )
            ax.set_title(f"{dataset} | {model_display_name(model)}")
            ax.set_xlabel(score_column.replace("_", " ").capitalize())
            ax.set_ylabel("")
            ax.set_xlim(0, max(0.01, global_max * 1.08))
            ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_top_items(
    scores: pd.DataFrame,
    item_label: str,
    item_id_column: str,
    item_name_column: str,
) -> None:
    """Consensus top-items plot (average over all embeddings)."""
    if item_label == "parcels":
        output_path = PLOTS_DIR / "consensus2_main_top20_parcels_main.pdf"
    else:
        output_path = PLOTS_DIR / f"consensus2_main_{item_label}_top20.pdf"

    plot_top_items_single(
        scores=scores,
        score_column="average_score",
        item_label=item_label,
        item_id_column=item_id_column,
        item_name_column=item_name_column,
        suptitle=(
            f"Top {TOP_N_REGIONS} consensus {item_label} "
            f"across embedding representations\n"
            f"{', '.join(TASKS)} | {', '.join(MICROSTRUCTURES)}"
        ),
        output_path=output_path,
    )


def plot_top_items_per_embedding(
    proportions: pd.DataFrame,
    item_label: str,
    item_id_column: str,
    item_name_column: str,
) -> None:
    """
    One top-items figure per embedding, saved individually.

    Uses selection_proportion directly from proportions
    (no averaging across embeddings).
    """
    embeddings = [
        e for e in selected_embedding
        if e in proportions["region_representation"].unique()
    ]

    for embedding in embeddings:
        subset = proportions[
            proportions["region_representation"] == embedding
        ].copy()

        if item_label == "parcels":
            output_path = (
                PLOTS_DIR
                / f"consensus2_main_top{TOP_N_REGIONS}_parcels_{embedding}.pdf"
            )
        else:
            output_path = (
                PLOTS_DIR
                / f"consensus2_main_{item_label}_top{TOP_N_REGIONS}_{embedding}.pdf"
            )

        plot_top_items_single(
            scores=subset,
            score_column="selection_proportion",
            item_label=item_label,
            item_id_column=item_id_column,
            item_name_column=item_name_column,
            suptitle=(
                f"Top {TOP_N_REGIONS} {item_label} — embedding: {embedding}\n"
                f"{', '.join(TASKS)} | {', '.join(MICROSTRUCTURES)}"
            ),
            output_path=output_path,
        )

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
    selection_data = selection_data[
        selection_data["region_representation"].isin(selected_embedding)
    ].copy()
    region_proportions = region_selection_proportions(selection_data)
    item_configs = [
        ("parcels", region_proportions, "region_id", "region_name"),
        (
            "networks",
            main_network_selection_proportions(region_proportions),
            "network_name",
            "network_name",
        ),
    ]

    for item_label, proportions, item_id_column, item_name_column in item_configs:
        scores = consensus_scores(proportions, item_id_column, item_name_column)
        if item_label == "parcels":
            scores_path = CSV_DIR / "consensus2_average_scores.csv"
            top_scores_path = CSV_DIR / "consensus2_top20_parcels.csv"
        else:
            scores_path = CSV_DIR / f"consensus2_{item_label}_average_scores.csv"
            top_scores_path = CSV_DIR / f"consensus2_{item_label}_top20.csv"
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

        plot_score_histogram(scores, item_label)
        plot_top_items(scores, item_label, item_id_column, item_name_column)

        # ── new: per-embedding breakdown ──────────────────────────────────────────
        plot_score_histogram(scores, item_label)
        plot_top_items(scores, item_label, item_id_column, item_name_column)

        # ── new ──────────────────────────────────────────────────────────────────
        plot_top_items_per_embedding(
            proportions, item_label, item_id_column, item_name_column
        )
        print(f"Saved consensus {item_label} average scores to {scores_path}")
        print(f"Saved top consensus {item_label} to {top_scores_path}")

        if item_label == "networks":
            plot_network_cosine_heatmap(scores)
            plot_network_barplot(scores)
            # Kendall tau: network ranking agreement CamCAN vs HCP
            kendall_df = network_kendall_tau(proportions)          # raw proportions

            kendall_path = CSV_DIR / "consensus2_networks_kendall_tau.csv"
            kendall_df.to_csv(kendall_path, index=False)
            print(f"Saved Kendall tau table to {kendall_path}")

            fig = plot_network_kendall_table(kendall_df)
            fig.savefig(
                PLOTS_DIR / "consensus2_main_networks_kendall_table.pdf",
                dpi=300, bbox_inches="tight",
            )
            plt.close(fig)
            print("Saved Kendall tau table figure.")
            global_kt = network_global_kendall_tau(proportions)
            print_global_kendall_summary(global_kt)

            # save the per-network rank table
            global_kt["global_scores"].to_csv(
                CSV_DIR / "consensus2_networks_global_kendall.csv",
                index=False,
            )
            rank_table, sub_scores = main_network_rank_table(scores)

            save_main_network_rank_table(rank_table, DATASETS)

            fig = plot_main_network_rank_table(rank_table, sub_scores, DATASETS)
            fig.savefig(
                PLOTS_DIR / "consensus2_main_network_rank_table.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
    print("Saved main network rank table figure.")
    print(f"Rows after performance filtering: {len(selection_data):,}")
    print(f"Selected rows: {selection_data['selected'].sum():,}")
    print(f"Embedding representations averaged: {', '.join(selected_embedding)}")


if __name__ == "__main__":
    main()
