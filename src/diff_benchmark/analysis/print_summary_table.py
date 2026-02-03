import json
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf


def is_successful_experiment(exp_dir: Path) -> bool:
    metadata_path = exp_dir / "metadata.yaml"
    if not metadata_path.exists():
        return False

    metadata = OmegaConf.load(metadata_path)
    return metadata.get("status") == "success"
    
def table_best_means(
    df: pd.DataFrame,
    primary_metric: str = "accuracy"
) -> pd.DataFrame:

    # Find all metrics containing the primary metric string
    related_metrics = df[df["metric"].str.contains(primary_metric, case=False, na=False)]["metric"].unique()
    
    df_filt = df[
        (df["metric"].isin(related_metrics)) &
        (df["split"] == "test")
    ]

    df_sorted = df_filt.sort_values("mean", ascending=False)

    df_best = (
        df_sorted
        .groupby(
            ["dataset", "prediction_task", "metric", "model_name"],
            as_index=False
        )
        .first()
    )

    return df_best


def select_best_runs(
    df: pd.DataFrame,
    primary_metric: str = "accuracy"
) -> dict:

    df_test = df[
        (df["metric"] == primary_metric) &
        (df["split"] == "test")
    ]

    best = {}

    group_cols = ["dataset", "prediction_task", "model_name"]

    for keys, df_group in df_test.groupby(group_cols):
        best_row = df_group.loc[df_group["mean"].idxmax()]
        best[keys] = best_row["run_id"]

    return best


def table_detailed(
    df_metrics: pd.DataFrame,
    best_runs: dict,
    primary_metric: str = "accuracy"
) -> pd.DataFrame:

    rows = []
    
    # Find all related metrics
    related_metrics = df_metrics[
        df_metrics["metric"].str.contains(primary_metric, case=False, na=False)
    ]["metric"].unique()

    for (dataset, task, model), run_id in best_runs.items():
        df_run = df_metrics[
            (df_metrics["run_id"] == run_id) &
            (df_metrics["metric"].isin(related_metrics))
        ]

        for _, row in df_run.iterrows():
            rows.append({
                "dataset": dataset,
                "task": task,
                "model_name": model,
                "metric": row["metric"],
                "fold": row["fold"],
                "split": row["split"],
                "value": row["value"],
            })

    return pd.DataFrame(rows).sort_values(
        ["dataset", "task", "model_name", "metric", "fold", "split"]
    )



def table_folds_wide(
    df_metrics: pd.DataFrame,
    best_runs: dict,
    split: str = "test",
    primary_metric: str = "accuracy"
) -> pd.DataFrame:

    rows = []
    
    # Find all related metrics
    related_metrics = df_metrics[
        df_metrics["metric"].str.contains(primary_metric, case=False, na=False)
    ]["metric"].unique()

    for (dataset, task, model), run_id in best_runs.items():
        for metric in related_metrics:
            df_run = df_metrics[
                (df_metrics["run_id"] == run_id) &
                (df_metrics["metric"] == metric) &
                (df_metrics["split"] == split)
            ]
            
            if df_run.empty:
                continue

            row = {
                "dataset": dataset,
                "task": task,
                "model": model,
                "metric": metric,
            }

            for _, r in df_run.iterrows():
                row[f"fold{r['fold']}"] = r["value"]

            row["mean"] = df_run["value"].mean()
            row["std"] = df_run["value"].std()

            rows.append(row)

    df = pd.DataFrame(rows)

    fold_cols = sorted(
        [c for c in df.columns if c.startswith("fold")],
        key=lambda x: int(x.replace("fold", ""))
    )

    return df[["dataset", "task", "model", "metric", *fold_cols, "mean", "std"]]



def print_table(df: pd.DataFrame):
    if df.empty:
        print("(empty table)")
        return
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
