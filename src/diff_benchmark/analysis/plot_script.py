from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


# ---------------------------------------------------------------------
# IO utilities
# ---------------------------------------------------------------------

def load_predictions_and_targets(
    predictions_path: Path,
    targets_path: Path,
    run_id: str,
):
    df_pred = pd.read_parquet(predictions_path)
    df_tgt = pd.read_parquet(targets_path)
    
    df_pred = df_pred[df_pred["run_id"] == run_id]

    df = df_pred.merge(
        df_tgt,
        on=["dataset", "sample_id", "target"],
        how="inner",
        suffixes=("_pred", "_true"),
    )

    df = df.rename(columns={"value": "true", "prediction": "pred"})
    return df


def load_metrics(metrics_path: Path, run_id: str) -> pd.DataFrame:
    df = pd.read_parquet(metrics_path)
    return df[df["run_id"] == run_id]


def get_prediction_task(metrics_df, run_id):
    # breakpoint()
    return (
        metrics_df
        .loc[metrics_df["run_id"] == run_id, "prediction_task"]
        .iloc[0]
    )


def infer_classification_type(df: pd.DataFrame) -> str:
    """
    binary or multiclass
    """
    n_classes = df["true"].nunique()
    return "binary" if n_classes == 2 else "multiclass"


# ---------------------------------------------------------------------
# Regression plots
# ---------------------------------------------------------------------

def plot_true_vs_pred_regression(df, run_id, model, output_dir):
    fig, ax = plt.subplots(figsize=(6, 6))

    for split, marker, alpha in [("train", "o", 0.4), ("test", "s", 0.7)]:
        d = df[df["split"] == split]
        ax.scatter(d["true"], d["pred"], label=split, alpha=alpha, marker=marker)

    min_v = min(df["true"].min(), df["pred"].min())
    max_v = max(df["true"].max(), df["pred"].max())
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--")

    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title(f"{model} – True vs Predicted")
    ax.legend()
    ax.grid(True)
    
    # Set axis limits
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "regression_true_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_regression(df, run_id, model, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))

    for split, marker, alpha in [("train", "o", 0.4), ("test", "s", 0.7)]:
        d = df[df["split"] == split]
        residuals = d["pred"] - d["true"]
        ax.scatter(d["true"], residuals, label=split, alpha=alpha, marker=marker)

    ax.axhline(0, linestyle="--")
    ax.set_xlabel("True value")
    ax.set_ylabel("Residual (pred − true)")
    ax.set_title(f"{model} – Residuals")
    ax.legend()
    ax.grid(True)

    fig.savefig(output_dir / "regression_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Classification plots
# ---------------------------------------------------------------------

def plot_confusion_matrix(cm, classes, title, output_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_classification_confusions(df, model, output_dir):
    classes = np.sort(df["true"].unique())

    for split in ["train", "test"]:
        d = df[df["split"] == split]
        y_true = d["true"].astype(int)
        y_pred = d["pred"].astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=classes)

        plot_confusion_matrix(
            cm,
            classes=classes,
            title=f"{model} – {split} confusion matrix",
            output_path=output_dir / f"confusion_{split}.png",
        )


def plot_binary_roc(df, model, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))

    for split, alpha in [("train", 0.4), ("test", 0.8)]:
        d = df[df["split"] == split]
        y_true = d["true"].astype(int)
        y_score = d["pred"].astype(float)
        
        # Remap labels to {0, 1} if they're {1, 2}
        unique_labels = np.sort(y_true.unique())
        if len(unique_labels) == 2 and unique_labels[0] != 0:
            # Labels are not {0, 1}, remap them
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            y_true_binary = y_true.map(label_map)
        else:
            y_true_binary = y_true

        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{split} (AUC={roc_auc:.3f})", alpha=alpha)

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model} – ROC curve")
    ax.legend()
    ax.grid(True)

    fig.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    

def plot_metrics_summary(metrics_df, model, output_dir):
    """
    One figure with subplots, one per metric.
    Shows train vs test across folds.
    """
    metrics = sorted(metrics_df["metric"].unique())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        n_metrics, 1,
        figsize=(6, 3 * n_metrics),
        sharex=True,
    )

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        d = metrics_df[metrics_df["metric"] == metric]

        for split, marker, alpha in [
            ("train", "o", 0.5),
            ("test", "s", 0.9),
        ]:
            ds = d[d["split"] == split]
            ax.plot(
                ds["fold"],
                ds["value"],
                marker=marker,
                linestyle="--",
                alpha=alpha,
                label=split,
            )

        ax.set_title(metric)
        ax.set_ylabel("value")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("fold")
    fig.suptitle(f"{model} – metrics summary", y=1.02)

    fig.tight_layout()
    fig.savefig(
        output_dir / "metrics_summary.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def plot_run(
    run_id: str,
    metrics_dir: Path,
    predictions_path: Path,
    targets_path: Path,
    output_root: Path,
):
    df = load_predictions_and_targets(predictions_path, targets_path, run_id)
    metrics_df = load_metrics(
        metrics_dir,
        run_id,
    )

    model = df["model"].iloc[0]
    # prediction_task = infer_prediction_task(df)
    prediction_task = get_prediction_task(
        metrics_df,
        run_id,
    )
    
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if prediction_task == "regression":
        plot_true_vs_pred_regression(df, run_id, model, output_dir)
        plot_residuals_regression(df, run_id, model, output_dir)

    else:
        plot_classification_confusions(df, model, output_dir)
        clf_type = infer_classification_type(df)

        if clf_type == "binary":
            plot_binary_roc(df, model, output_dir)
    
    plot_metrics_summary(metrics_df, model, output_dir)


# ---------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--results_dir", default="./data/results", type=str
    )
    parser.add_argument(
        "--metrics_dir", default="./data/results/parquet/analysis_results", type=str
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics_dir = Path(args.metrics_dir)

    plot_run(
        run_id=args.run_id, #2dcnn_be425892'2dcnn_08ef30ab', '2dcnn_341c8a9b', 
        # '2dcnn_76059b89',
    #    '2dcnn_be425892', 'linear_2ddaa507', 'linear_3addbf07',
    #    'linear_cf6ab721'
        metrics_dir=metrics_dir / "metrics.parquet",
        predictions_path=results_dir / "parquet/data/predictions.parquet",
        targets_path=results_dir / "parquet/data/targets.parquet",
        output_root=results_dir / "plots",
    )
