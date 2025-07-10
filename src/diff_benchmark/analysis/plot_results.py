import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from diff_benchmark.scores.scores import mse_score


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


def plot_folds_predictions_vs_targets(summary_path: Path, output_dir: Path):
    with open(summary_path, "r") as f:
        fold_results = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = ["circle", "x", "square", "diamond", "cross", "star"]  # Up to 6 folds
    n_folds = len(fold_results)

    # Check first fold to determine number of features
    # n_features = len(fold_results[0]["train"]["predictions"][0])
    example_pred = fold_results[0]["train"]["predictions"][0]

    if isinstance(example_pred, (float, int)):
        n_features = 1
    else:
        n_features = len(example_pred)

    for feat_idx in range(n_features):
        fig = go.Figure()

        for fold_data in fold_results:
            fold = fold_data["fold"]
            symbol = symbols[fold % len(symbols)]

            train_preds = np.array(fold_data["train"]["predictions"])
            test_preds = np.array(fold_data["test"]["predictions"])

            train_targets = np.array(fold_data["train"]["targets"])
            test_targets = np.array(fold_data["test"]["targets"])
            
            if n_features == 1:
                train_pred_vals = train_preds
                train_target_vals = train_targets
                test_pred_vals = test_preds
                test_target_vals = test_targets
            else:
                train_pred_vals = train_preds[:, feat_idx]
                train_target_vals = train_targets[:, feat_idx]
                test_pred_vals = test_preds[:, feat_idx]
                test_target_vals = test_targets[:, feat_idx]

            # train_mse = mean_squared_error(train_targets[:, feat_idx], train_preds[:, feat_idx])
            # test_mse = mean_squared_error(test_targets[:, feat_idx], test_preds[:, feat_idx])
            train_mse = mean_squared_error(train_target_vals, train_pred_vals)
            test_mse = mean_squared_error(test_target_vals, test_pred_vals)

            fig.add_trace(
                go.Scatter(
                    x=train_target_vals,
                    y=train_pred_vals,
                    mode="markers",
                    marker=dict(color="red", symbol=symbol, size=6),
                    name=f"Train Fold {fold+1} (MSE={train_mse:.4f})",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=test_target_vals,
                    y=test_pred_vals,
                    mode="markers",
                    marker=dict(color="blue", symbol=symbol, size=6),
                    name=f"Test Fold {fold+1} (MSE={test_mse:.4f})",
                )
            )

        # Identity line
        # all_preds = [
        #     np.array(f["train"]["predictions"])[:, feat_idx] for f in fold_results
        # ] + [
        #     np.array(f["test"]["predictions"])[:, feat_idx] for f in fold_results
        # ]
        # all_vals = np.concatenate(all_preds)
        # min_val, max_val = all_vals.min(), all_vals.max()
        min_val = min(train_target_vals.min(), test_target_vals.min())
        max_val = max(train_target_vals.max(), test_target_vals.max())

        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                showlegend=False,
            )
        )

        fig.update_layout(
            title=f"Feature {feat_idx+1} | Predictions vs Targets \n MSE Train: {train_mse:.4f}, Test: {test_mse:.4f}",
            xaxis_title="Target",
            yaxis_title="Predicted",
            width=900,
            height=700,
            legend=dict(itemsizing="constant"),
        )

        html_path = output_dir / f"feature_{feat_idx+1}_pred_vs_target.html"
        fig.write_html(str(html_path))
        print(f"Saved: {html_path}")