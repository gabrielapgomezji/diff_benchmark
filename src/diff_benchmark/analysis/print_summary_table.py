import json
import pandas as pd
from pathlib import Path


def load_global_metrics(metrics_path: Path) -> pd.DataFrame:
    """
    Load the combined metrics.parquet (fold-level metrics for all experiments).
    """
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found")
    
    df = pd.read_parquet(metrics_path)
    return df
    
def table_best_means(df_metrics: pd.DataFrame, primary_metric: str = "accuracy") -> pd.DataFrame:
    """
    Compute best mean results per model using the precomputed 'mean' column.
    """
    df_filtered = df_metrics[(df_metrics["metric"] == primary_metric) & (df_metrics["split"] == "test")]

    # sort by mean descending
    df_sorted = df_filtered.sort_values("mean", ascending=False)
    
    # pick one row per model
    df_best = df_sorted.groupby("model_name", as_index=False).first()
    return df_best


def select_best_runs(df_metrics: pd.DataFrame, primary_metric: str = "accuracy") -> dict:
    """
    Select the run_id with highest test mean for each model.
    """
    best = {}
    df_test = df_metrics[(df_metrics["metric"] == primary_metric) & (df_metrics["split"] == "test")]

    for model in df_test["model_name"].unique():
        df_model = df_test[df_test["model_name"] == model]
        best_row = df_model.loc[df_model["mean"].idxmax()]
        best[model] = best_row["run_id"]
    
    return best


def table_detailed(df_metrics: pd.DataFrame, best_runs: dict, primary_metric: str = "accuracy") -> pd.DataFrame:
    rows = []

    for model, run_id in best_runs.items():
        df_run = df_metrics[(df_metrics["run_id"] == run_id) & (df_metrics["model_name"] == model) & (df_metrics["metric"] == primary_metric)]

        for _, row in df_run.iterrows():
            rows.append({
                "model_name": model,
                "fold": row["fold"],
                "split": row["split"],
                "value": row["value"]
            })

    return pd.DataFrame(rows).sort_values(["model_name", "fold", "split"])


def table_folds_wide(df_metrics: pd.DataFrame, best_runs: dict, split: str = "test", primary_metric: str = "accuracy") -> pd.DataFrame:
    rows = []

    for model, run_id in best_runs.items():
        df_run = df_metrics[(df_metrics["run_id"] == run_id) &
                            (df_metrics["model_name"] == model) &
                            (df_metrics["metric"] == primary_metric) &
                            (df_metrics["split"] == split)]

        row = {"model": model}
        for _, r in df_run.iterrows():
            row[f"fold{r['fold']}"] = r["value"]

        # summary stats
        row["mean"] = df_run["value"].mean()
        row["std"] = df_run["value"].std()

        rows.append(row)

    df = pd.DataFrame(rows)
    # reorder columns
    fold_cols = sorted([c for c in df.columns if c.startswith("fold")], key=lambda x: int(x.replace("fold", "")))
    return df[["model", *fold_cols, "mean", "std"]]


def print_table(df: pd.DataFrame):
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
