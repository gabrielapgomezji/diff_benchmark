from pathlib import Path
import pandas as pd

def load_debug_logs(run_id: str, debug_dir: Path) -> pd.DataFrame:
    torch_path = debug_dir / f"torch_debug_{run_id}.parquet"
    lightning_path = debug_dir / f"lightning_debug_{run_id}.parquet"

    if torch_path.exists():
        df = pd.read_parquet(torch_path)
    elif lightning_path.exists():
        df = pd.read_parquet(lightning_path)
    else:
        raise FileNotFoundError(
            f"No debug logs found for run_id={run_id}"
        )

    # keep epoch-level rows only
    df = df[df["batch"].isna()].copy()
    return df


import matplotlib.pyplot as plt

def plot_metric(ax, df, metric, title):
    for split, style in [("train", "-"), ("val", "--")]:
        d = df[df["split"] == split]
        if metric in d.columns:
            ax.plot(d["epoch"], d[metric], style, label=split)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(True)
    ax.legend()


def plot_classification_debug(df, run_id, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    plot_metric(axes[0, 0], df, "loss", "Loss")
    plot_metric(axes[0, 1], df, "accuracy", "Accuracy")

    # precision + recall together
    ax = axes[1, 0]
    for metric in ["precision", "recall"]:
        for split, style in [("train", "-"), ("val", "--")]:
            d = df[df["split"] == split]
            if metric in d.columns:
                ax.plot(
                    d["epoch"],
                    d[metric],
                    linestyle=style,
                    label=f"{metric}-{split}",
                )
    ax.set_title("Precision & Recall")
    ax.legend()
    ax.grid(True)

    plot_metric(axes[1, 1], df, "f1", "F1 score")

    fig.suptitle(f"{run_id} – Classification debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "debug_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regression_debug(df, run_id, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    plot_metric(axes[0, 0], df, "loss", "Loss")
    plot_metric(axes[0, 1], df, "explained_variance", "Explained variance")

    # mse + mape together
    ax = axes[1, 0]
    for metric in ["mse", "mape"]:
        for split, style in [("train", "-"), ("val", "--")]:
            d = df[df["split"] == split]
            if metric in d.columns:
                ax.plot(
                    d["epoch"],
                    d[metric],
                    linestyle=style,
                    label=f"{metric}-{split}",
                )
    ax.set_title("MSE & MAPE")
    ax.legend()
    ax.grid(True)

    plot_metric(axes[1, 1], df, "r2", "R²")

    fig.suptitle(f"{run_id} – Regression debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "debug_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_debug_run(
    run_id: str,
    prediction_task: str,
    debug_dir: Path,
    output_root: Path,
):
    df = load_debug_logs(run_id, debug_dir)

    output_dir = output_root / run_id / "debug"

    if prediction_task == "classification":
        plot_classification_debug(df, run_id, output_dir)
    else:
        plot_regression_debug(df, run_id, output_dir)


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
        "--prediction_task",
        type=str,
        choices=["classification", "regression"],
        required=True,
        help="Type of prediction task.",
    )
    parser.add_argument(
        "--debug_dir",
        type=Path,
        required=True,
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
    prediction_task = args.prediction_task
    debug_dir = args.debug_dir
    results_dir = args.output_root
    breakpoint()
    plot_debug_run(
        run_id=run_id,
        prediction_task=prediction_task,
        debug_dir=debug_dir,
        output_root=results_dir / "plots_debug",
    )