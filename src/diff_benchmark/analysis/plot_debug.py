from pathlib import Path

import pandas as pd

SPLIT_STYLE = {
    "train": {"color": "tab:blue"},
    "val": {"color": "tab:red"},
}


METRIC_LINESTYLE = {
    "solid": "-",
    "dashed": "--",
}

# def load_debug_logs(run_id: str, debug_dir: Path) -> pd.DataFrame:
#     torch_path = debug_dir / f"torch_debug_{run_id}.parquet"
#     lightning_path = debug_dir / f"lightning_debug_{run_id}.parquet"

#     if torch_path.exists():
#         df = pd.read_parquet(torch_path)
#     elif lightning_path.exists():
#         df = pd.read_parquet(lightning_path)
#     else:
#         raise FileNotFoundError(
#             f"No debug logs found for run_id={run_id}"
#         )
#     breakpoint()
#     # keep epoch-level rows only
#     df = df[df["batch"].isna()].copy()
#     return df


def load_debug_logs(file_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    df = df[df["batch"].isna()].copy()  # keep only epoch-level rows

    # Extract fold from filename
    fold_str = file_path.stem.split("_")[-1]
    if fold_str.startswith("fold"):
        df["fold"] = int(fold_str.replace("fold", ""))
    else:
        df["fold"] = None

    return df


import matplotlib.pyplot as plt


def plot_metric(ax, df, metric, title):
    for split in ["train", "val"]:
        d = df[df["split"] == split]
        if metric not in d.columns:
            continue

        ax.plot(
            d["epoch"],
            d[metric],
            label=split,
            color=SPLIT_STYLE[split]["color"],
            linestyle="-",  # always solid for single-metric plots
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_two_metrics(
    ax,
    df,
    metric_a,
    metric_b,
    title,
):
    metric_styles = {
        metric_a: METRIC_LINESTYLE["solid"],
        metric_b: METRIC_LINESTYLE["dashed"],
    }

    for metric, linestyle in metric_styles.items():
        for split in ["train", "val"]:
            d = df[df["split"] == split]
            if metric not in d.columns:
                continue

            ax.plot(
                d["epoch"],
                d[metric],
                label=f"{metric}-{split}",
                color=SPLIT_STYLE[split]["color"],
                linestyle=linestyle,
            )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_classification_debug(df, run_id, output_dir):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        gridspec_kw={"hspace": 0.4, "wspace": 0.25},
    )

    plot_metric(axes[0, 0], df, "loss", "Loss")

    if "weighted_accuracy" in df.columns and df["weighted_accuracy"].notna().any():
        plot_metric(axes[0, 1], df, "weighted_accuracy", "Weighted Accuracy")
    else:
        plot_metric(axes[0, 1], df, "accuracy", "Accuracy")

    plot_two_metrics(
        axes[1, 0],
        df,
        metric_a="precision",
        metric_b="recall",
        title="Precision & Recall",
    )

    plot_metric(axes[1, 1], df, "f1", "F1 score")

    fig.suptitle(f"{run_id} – Classification debug", fontsize=14)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"debug_training_{run_id}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_regression_debug(df, run_id, output_dir):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        gridspec_kw={"hspace": 0.4, "wspace": 0.25},
    )

    plot_metric(axes[0, 0], df, "loss", "Loss")
    plot_metric(axes[0, 1], df, "explained_variance", "Explained Variance")

    plot_two_metrics(
        axes[1, 0],
        df,
        metric_a="mse",
        metric_b="mape",
        title="MSE & MAPE",
    )

    plot_metric(axes[1, 1], df, "r2", "R²")

    fig.suptitle(f"{run_id} – Regression debug", fontsize=14)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"debug_training_{run_id}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def infer_prediction_task(df: pd.DataFrame) -> str:
    """
    Infer prediction task from available non-null metrics.
    """
    # epoch_df = df[df["batch"].isna()]
    epoch_df = df

    classification_metrics = {
        "accuracy",
        "weighted_accuracy",
        "precision",
        "recall",
        "f1",
    }
    regression_metrics = {"mse", "mape", "r2", "explained_variance"}

    present_metrics = {
        col
        for col in epoch_df.columns
        if col not in {"split", "epoch", "batch", "loss"}
        and epoch_df[col].notna().any()
    }

    if present_metrics & classification_metrics:
        return "binary_classification"
    if present_metrics & regression_metrics:
        return "regression"

    raise ValueError(f"Could not infer prediction task from metrics: {present_metrics}")


def plot_debug_run(
    run_id: str,
    debug_dir: Path,
    output_root: Path,
):
    torch_files = list(debug_dir.glob(f"torch_debug_{run_id}*.parquet"))
    lightning_files = list(debug_dir.glob(f"lightning_debug_{run_id}*.parquet"))
    all_files = torch_files + lightning_files

    if not all_files:
        raise FileNotFoundError(f"No debug logs found for run_id={run_id}")

    df_list = [load_debug_logs(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    prediction_task = infer_prediction_task(df)

    output_dir = output_root / run_id / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in df["fold"].dropna().unique():
        df_fold = df[df["fold"] == fold]
        prediction_task = infer_prediction_task(df_fold)

        fold_id = f"_fold{int(fold)}"
        if prediction_task == "binary_classification":
            plot_classification_debug(df_fold, f"{run_id}{fold_id}", output_dir)
        else:
            plot_regression_debug(df_fold, f"{run_id}{fold_id}", output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot debug training logs for a given run."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run ID to plot debug logs for.",
    )
    parser.add_argument(
        "--debug_dir",
        default="./data/results/parquet/debug",
        type=str,
        help="Directory containing debug log parquet files.",
    )
    parser.add_argument(
        "--results_dir",
        default="./data/results",
        type=str,
        help="Root directory to save the plots.",
    )
    args = parser.parse_args()
    # Example usage
    run_id = args.run_id
    debug_dir = Path(args.debug_dir)
    results_dir = Path(args.results_dir)

    plot_debug_run(
        run_id=run_id,  # 2dcnn_a5942b19
        debug_dir=debug_dir,
        output_root=results_dir / "plots",
    )
