import json
import pandas as pd
from pathlib import Path


def load_all_runs(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)
    
def table_best_means(all_runs: list[dict]) -> pd.DataFrame:
    rows = []

    for r in all_runs:
        res = r["results"]
        if res["test_average_score"] is None:
            continue

        rows.append({
            "model_name": r["model_name"],
            "train_mean": res["train_average_score"],
            "test_mean": res["test_average_score"],
        })

    df = pd.DataFrame(rows)

    return (
        df.sort_values("test_mean", ascending=False)
          .groupby("model_name", as_index=False)
          .first()
          .sort_values("test_mean", ascending=False)
    )


def select_best_runs(all_runs: list[dict]) -> dict:
    best = {}

    for r in all_runs:
        model = r["model_name"]
        test_mean = r["results"]["test_average_score"]

        if test_mean is None:
            continue

        if model not in best or test_mean > best[model]["results"]["test_average_score"]:
            best[model] = r

    return best


def print_table(df: pd.DataFrame):
    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

def select_best_runs(all_runs: list[dict]) -> dict:
    best = {}

    for r in all_runs:
        model = r["model_name"]
        test_mean = r["results"]["test_average_score"]

        if test_mean is None:
            continue

        if model not in best or test_mean > best[model]["results"]["test_average_score"]:
            best[model] = r

    return best

def table_detailed(best_runs: dict, primary_metric: str) -> pd.DataFrame:
    rows = []

    for model, run in best_runs.items():
        res = run["results"]

        for fold, fold_res in res["folds"].items():
            fold_idx = int(fold.split("_")[-1])
            rows.append({
                "model_name": model,
                "fold": fold_idx,
                "train": fold_res["train"]["score"],
                "test": fold_res["test"]["score"],
                "train_mean": res["train_average_score"],
                "train_std": res["train_std_score"],
                "test_mean": res["test_average_score"],
                "test_std": res["test_std_score"],
            })

    return pd.DataFrame(rows).sort_values(["model_name", "fold"])

def table_folds_wide(
    best_runs: dict,
    split: str = "test",  # or "train"
) -> pd.DataFrame:
    rows = []

    for model, run in best_runs.items():
        res = run["results"]
        row = {"model": model}

        # collect folds
        for fold_key, fold_res in res["folds"].items():
            fold_idx = int(fold_key.split("_")[-1])
            row[f"fold{fold_idx}"] = fold_res[split]["score"]

        # add summary stats
        row["mean"] = res[f"{split}_average_score"]
        row["std"] = res[f"{split}_std_score"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # ensure fold columns are ordered
    fold_cols = sorted(
        [c for c in df.columns if c.startswith("fold")],
        key=lambda x: int(x.replace("fold", ""))
    )

    return df[["model", *fold_cols, "mean", "std"]]
