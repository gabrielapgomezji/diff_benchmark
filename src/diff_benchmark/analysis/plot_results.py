import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error


def plot_predictions_vs_targets(summary_path: Path, output_dir: Path):
    # Load JSON summary
    with open(summary_path, "r") as f:
        summary = json.load(f)

    train_preds = np.array(summary["train_predictions_mean"])
    train_targets = np.array(summary["train_targets_mean"])
    test_preds = np.array(summary["test_predictions_mean"])
    test_targets = np.array(summary["test_targets_mean"])

    # Compute MSE
    train_mse = np.array(summary["train_score_mean"])
    test_mse = np.array(summary["test_score_mean"])

    # Build traces for each feature (dimension)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features = train_targets.shape[-1]

    for i in range(n_features):
        # Compute per-feature MSE
        train_mse = mean_squared_error(train_targets[:, i], train_preds[:, i])
        test_mse = mean_squared_error(test_targets[:, i], test_preds[:, i])

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=train_targets[:, i],
                y=train_preds[:, i],
                mode="markers",
                marker=dict(color="red", symbol="circle", size=6),
                name="Train",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=test_targets[:, i],
                y=test_preds[:, i],
                mode="markers",
                marker=dict(color="blue", symbol="x", size=6),
                name="Test",
            )
        )

        # Identity line
        min_val = min(train_targets[:, i].min(), test_targets[:, i].min())
        max_val = max(train_targets[:, i].max(), test_targets[:, i].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                showlegend=False,
            )
        )

        # Layout
        fig.update_layout(
            title=f"Feature {i+1} | MSE Train: {train_mse:.4f}, Test: {test_mse:.4f}",
            xaxis_title="Target Value",
            yaxis_title="Predicted Value",
            width=900,
            height=700,
            legend=dict(itemsizing="constant"),
        )

        # Save .html
        html_path = output_dir / f"feature_{i+1}_pred_vs_target.html"
        fig.write_html(str(html_path))

        # Optional: Save as PDF too (requires Kaleido)
        # pdf_path = output_dir / f"feature_{i+1}_pred_vs_target.pdf"
        # fig.write_image(str(pdf_path), format="pdf")

        print(f"Saved plot for Feature {i+1} to {html_path}")
