import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_true_vs_pred(
    y_true,
    y_pred,
    fold_idx: int | None = None,
    run_id: str | None = None,
    prediction_type: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """Scatter plot of true vs predicted values for a regression task.

    Saves the figure to *output_dir* (defaults to the current working
    directory).  Does **not** call :func:`plt.show` so it is safe to use in
    batch / headless contexts.

    Args:
        y_true: Array of ground-truth target values.
        y_pred: Array of model predictions.
        fold_idx: Cross-validation fold index; included in the filename.
        run_id: Run identifier; included in the filename.
        prediction_type: Short label for the target type shown in the title.
        output_dir: Directory to save the figure.  Defaults to ``Path(".")``.
    """
    if output_dir is None:
        output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, s=20)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

    pearson_corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(
        f"True vs Predicted Values {prediction_type}.\n"
        f"Pearson Correlation: {pearson_corr:.3f}"
    )
    ax.grid(True)
    fig.tight_layout()

    filename = f"true_vs_pred_{fold_idx}_{run_id}_{prediction_type}.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
