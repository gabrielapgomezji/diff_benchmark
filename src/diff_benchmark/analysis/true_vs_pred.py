import matplotlib.pyplot as plt
import numpy as np

def plot_true_vs_pred(y_true, y_pred, fold_idx=None, run_id=None, type=None):
    """
    Scatter plot of true vs predicted values for regression.
    """
    plt.figure(figsize=(8, 8))

    # Scatter points
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    
    corr_matrix = np.corrcoef(y_true, y_pred)
    pearson_corr = corr_matrix[0, 1]

    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"True vs Predicted Values {type}. \n Pearson Correlation: {pearson_corr:.3f}")
    plt.savefig(f"true_vs_pred_{fold_idx}_{run_id}_{type}.png")

    plt.grid(True)
    plt.tight_layout()
    plt.show()