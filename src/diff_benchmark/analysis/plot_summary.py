# from pathlib import Path
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


# def plot_metrics_summary(
#     metrics_path: Path,
#     output_dir: Path,
# ):
#     df = pd.read_parquet(metrics_path)

#     # mean across folds
#     df_mean = (
#         df.groupby(
#             ["model_name", "dataset", "prediction_task", "split", "metric"],
#             as_index=False
#         )["value"]
#         .mean()
#     )

#     output_dir.mkdir(parents=True, exist_ok=True)

#     for task in df_mean["prediction_task"].unique():
#         df_task = df_mean[df_mean["prediction_task"] == task]

#         metrics = sorted(df_task["metric"].unique())
#         splits = ["train", "test"]

#         fig = make_subplots(
#             rows=len(metrics),
#             cols=len(splits),
#             shared_xaxes=True,
#             subplot_titles=[
#                 f"{m.upper()} – {s}"
#                 for m in metrics
#                 for s in splits
#             ],
#             vertical_spacing=0.06,
#             horizontal_spacing=0.08,
#         )

#         for row_idx, metric in enumerate(metrics, start=1):
#             for col_idx, split in enumerate(splits, start=1):
#                 d = df_task[
#                     (df_task["metric"] == metric)
#                     & (df_task["split"] == split)
#                 ]

#                 if d.empty:
#                     continue

#                 fig.add_trace(
#                     go.Bar(
#                         x=d["model_name"],
#                         y=d["value"],
#                         name=f"{metric}-{split}",
#                         showlegend=False,
#                     ),
#                     row=row_idx,
#                     col=col_idx,
#                 )

#         fig.update_layout(
#             title=f"Model comparison – {task.capitalize()} (mean over folds)",
#             height=300 * len(metrics),
#             template="plotly_white",
#         )

#         fig.update_xaxes(tickangle=45)

#         fig.write_html(output_dir / f"metrics_summary_{task}.html")

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_train_vs_test_scatter(
    metrics_path: Path,
    output_dir: Path,
):
    df = pd.read_parquet(metrics_path)

    # aggregate across folds
    df_agg = (
        df.groupby(
            ["model_name", "prediction_task", "metric", "split"],
            as_index=False
        )["value"]
        .agg(mean="mean", std="std")
    )

    df_wide = (
        df_agg
        .pivot(
            index=["model_name", "prediction_task", "metric"],
            columns="split",
            values=["mean", "std"]
        )
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for task in df_wide["prediction_task"].unique():
        df_task = df_wide[df_wide["prediction_task"] == task]

        metrics = sorted(df_task["metric"].unique())

        fig = make_subplots(
            rows=1,
            cols=len(metrics),
            subplot_titles=[m.upper() for m in metrics],
            horizontal_spacing=0.08,
        )

        for col_idx, metric in enumerate(metrics, start=1):
            d = df_task[df_task["metric"] == metric]

            if d.empty:
                continue

            # dot size = avg std between train & test
            size = (
                d["std"]["train"].fillna(0)
                + d["std"]["test"].fillna(0)
            ) / 2

            fig.add_trace(
                go.Scatter(
                    x=d["mean"]["train"],
                    y=d["mean"]["test"],
                    mode="markers+text",
                    text=d["model_name"],
                    textposition="top center",
                    customdata=np.stack(
                        [
                            d["std"]["train"].fillna(0),
                            d["std"]["test"].fillna(0),
                        ],
                        axis=1,
                    ),
                    marker=dict(
                        size=10 + 40 * size / size.max() if size.max() > 0 else 12,
                        opacity=0.75,
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Train mean: %{x:.4f}<br>"
                        "Test mean: %{y:.4f}<br>"
                        "Train std: %{customdata[0]:.4f}<br>"
                        "Test std: %{customdata[1]:.4f}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )


            # diagonal y=x
            min_v = min(
                d["mean"]["train"].min(),
                d["mean"]["test"].min(),
            )
            max_v = max(
                d["mean"]["train"].max(),
                d["mean"]["test"].max(),
            )

            fig.add_trace(
                go.Scatter(
                    x=[min_v, max_v],
                    y=[min_v, max_v],
                    mode="lines",
                    line=dict(dash="dash", color="gray"),
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )

            fig.update_xaxes(title_text="Train (mean)")
            fig.update_yaxes(title_text="Test (mean)")

        fig.update_layout(
            title=f"Train vs Test performance – {task.capitalize()}",
            template="plotly_white",
            height=500,
        )

        fig.write_html(
            output_dir / f"train_vs_test_{task}.html"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot metrics summary across models and datasets."
    )
    parser.add_argument(
        "--metrics-path",
        default="./data/results/parquet/analysis_results/metrics.parquet",
        type=str,
        help="Path to the metrics parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/results/plots",
        type=str,
        help="Directory to save the output plots.",
    )

    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    output_dir = Path(args.output_dir)

    # plot_metrics_summary(
    #     metrics_path=metrics_path,
    #     output_dir=output_dir,
    # )
    plot_train_vs_test_scatter(
        metrics_path=metrics_path,
        output_dir=output_dir,
    )