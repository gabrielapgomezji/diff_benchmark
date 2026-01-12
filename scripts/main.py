import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from diff_benchmark.analysis.save_results import (
    is_cached,
    save_model_results,
)
from diff_benchmark.analysis.true_vs_pred import plot_true_vs_pred
from diff_benchmark.data.prepare_data import DatasetPreparation
from diff_benchmark.models.model_configurations import get_model, make_run_id
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.utils.parquet_helper import ParquetSaver, metrics_to_rows
from diff_benchmark.utils.config_loader import load_configs
from diff_benchmark.utils.job_manager import run_jobs
from diff_benchmark.utils.scores import compute_metrics
from diff_benchmark.utils.summary_saver import update_summary, compute_summary_stats

parser = argparse.ArgumentParser()
parser.add_argument(
    "--methods", nargs="+", type=str, default=["2dcnn_torch"], help="Method to use"
)
args = parser.parse_args()

general_config, model_config = load_configs(args)


def run_single_model(model_name, model_config, general_config, results_path):
    metrics_rows = []

    config = general_config

    datasets_by_name = {
        d["name"]: d for d in general_config["datasets"]["datasets_list"]
    }
    dataset_selected = datasets_by_name[model_config["dataset"]]
    dataset_selected = DatasetConfig(
                **dataset_selected,
                metric_to_compute=general_config["datasets"]["metric_to_compute"],
                scale=general_config["datasets"]["scale"],
                region=general_config["data_preparation"]["region"],
            )
    torch_dataset_preparator = DatasetPreparation(
        model_name=model_name,
        model_config=model_config,
        general_config=general_config,
        source_dataset=dataset_selected,
    )
    
    dataset, preprocessed = torch_dataset_preparator.pipeline()

    targets_path = Path(results_path) / "parquet" / "data" / "targets.parquet"
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    
    target_name = config["target_columns"][0]
    rows = [
        {"dataset": dataset_selected.name, "sample_id": sid, "target": target_name, "value": float(v)}
        for sid, v in zip(dataset.subject_ids, dataset.targets.numpy())
    ]
    saver = ParquetSaver(
        path=targets_path,
        key_columns=["dataset", "sample_id", "target"],
        columns=["dataset", "sample_id", "target", "value"]
    )
    saver.add_rows(rows)
    saver.save()

    specs = preprocessed.get_specs()
    print(specs)

    indices = preprocessed.get_fold_indices()

    local_config = copy.deepcopy(model_config)
    local_config["model_name"] = model_name
    run_id = make_run_id(model_name, local_config)
    local_config["run_id"] = run_id

    if is_cached(run_id, Path(results_path) / "analysis_results"):
        print(f"Skipping {model_name} (run_id={run_id}) - already cached.")
        return model_name, run_id
    print(f"\nRunning model: {model_name} with run_id: {run_id}")

    train_scores, test_scores = [], []
    train_preds, test_preds = [], []
    train_targets, test_targets = [], []

    summary = {
        "model_name": model_name,
        # "preprocessing": {
        #     "data_type": config.get("data_type", "images"),
        #     "csv_file": str(config.get("csv_file", "")),
        #     "target_columns": config.get("target_columns", [])
        # },
        "pipeline": {
            "run_id": run_id,
            "comment": local_config.get("comment", ""),
        },
        "results": {
            "train_average_score": None,  # will fill after loop
            "train_std_score": None,  # will fill after loop
            "test_average_score": None,  # will fill after loop
            "test_std_score": None,  # will fill after loop
            "number_folds": len(indices),
            "folds": {},  # will fill inside loop
        },
    }
    exclude_keys = {"comment", "name", "model_name"}
    for key, value in local_config.items():
        if key not in exclude_keys:
            summary["pipeline"][key] = value

    save_model_results(
        summary, Path(results_path) / "analysis_results" / f"{run_id}_partial.json"
    )
    
    predictions_path = Path(results_path) / "parquet" / "data" / "predictions.parquet"
    key_cols = ["run_id", "model", "dataset", "fold", "split", "sample_id", "target"]
    pred_saver = ParquetSaver(predictions_path, key_columns=key_cols,
                            columns=[
                                "run_id", "model", "dataset", "fold", "split",
                                "sample_id", "target", "prediction"
                            ])

    for fold_idx, (train_idx, test_idx) in enumerate(indices):
        try:
            local_config["fold_idx"] = fold_idx
            train_loader, test_loader = preprocessed.get_dataloader_fold(
                dataset,
                fold_idx,
                indices,
                batch_size=local_config["data"]["batch_size"],
            )
            train_idx, test_idx = indices[fold_idx]
            targets = dataset.targets.numpy()
            y_train = np.array(targets[train_idx]).squeeze()
            y_test = np.array(targets[test_idx]).squeeze()

            local_config["backbone"]["prediction_task"] = config.get(
                "prediction_task", "regression"
            )
            local_config["backend"]["run_id"] = run_id
            model = get_model(model_name, local_config)

            model.fit(train_loader)
            train_pred = model.predict(train_loader)
            plot_true_vs_pred(
                y_train, train_pred, fold_idx=fold_idx, run_id=run_id, type="train"
            )
            train_score = compute_metrics(y_train, train_pred, prediction_task=local_config["backbone"]["prediction_task"])
            print(train_score)

            train_scores.append(train_score)
            train_preds.append(train_pred.tolist())
            train_targets.append(y_train.tolist())
            
            train_subject_ids = np.asarray(dataset.subject_ids)[train_idx]
            train_rows = [
                {
                    "run_id": run_id,
                    "model": model_name,
                    "dataset": dataset_selected.name,
                    "fold": fold_idx,
                    "split": "train",
                    "sample_id": sid,
                    "target": target_name,
                    "prediction": float(pred),
                }
                for sid, pred in zip(train_subject_ids, train_pred)
            ]
            pred_saver.add_rows(train_rows)
            
            test_pred = model.predict(test_loader)
            plot_true_vs_pred(
                y_test, test_pred, fold_idx=fold_idx, run_id=run_id, type="test"
            )
            test_score = compute_metrics(y_test, test_pred, prediction_task=local_config["backbone"]["prediction_task"])
            print(test_score)

            test_scores.append(test_score)
            test_preds.append(test_pred.tolist())
            test_targets.append(y_test.tolist())

            test_subject_ids = np.asarray(dataset.subject_ids)[test_idx]
            test_rows = [
                {
                    "run_id": run_id,
                    "model": model_name,
                    "dataset": dataset_selected.name,
                    "fold": fold_idx,
                    "split": "test",
                    "sample_id": sid,
                    "target": target_name,
                    "prediction": float(pred),
                }
                for sid, pred in zip(test_subject_ids, test_pred)
            ]
            pred_saver.add_rows(test_rows)
            pred_saver.save()

            
            primary_metric = {"classification": "accuracy", "regression": "mse"}[local_config["backbone"]["prediction_task"]]
            summary = update_summary(summary, fold_idx, train_score, test_score, y_train, train_pred, y_test, test_pred, primary_metric)

            metrics_rows.extend(
                metrics_to_rows(
                    train_score,
                    run_id=run_id,
                    model_name=model_name,
                    dataset=dataset_selected.name,
                    prediction_task=local_config["backbone"]["prediction_task"],
                    fold=fold_idx,
                    split="train",
                )
            )
            
            metrics_rows.extend(
                metrics_to_rows(
                    test_score,
                    run_id=run_id,
                    model_name=model_name,
                    dataset=dataset_selected.name,
                    prediction_task=local_config["backbone"]["prediction_task"],
                    fold=fold_idx,
                    split="test",
                )
            )

            save_model_results(
                summary,
                Path(results_path) / "analysis_results" / f"{run_id}_partial.json",
            )
        except Exception as e:
            print(f"Crash in fold {fold_idx} of {run_id}: {e}")
            save_model_results(
                summary,
                Path(results_path) / "analysis_results" / f"{run_id}_crashed.json",
            )
            raise
    
    metrics_path = Path(results_path) / "parquet" / "analysis_results" / f"metrics_{run_id}.parquet"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics_rows)
    df.to_parquet(metrics_path, index=False)

    summary["results"]["train_average_score"], summary["results"]["train_std_score"] = compute_summary_stats(train_scores, primary_metric)
    summary["results"]["test_average_score"], summary["results"]["test_std_score"] = compute_summary_stats(test_scores, primary_metric)

    save_model_results(summary, Path(results_path) / "analysis_results")
    return model_name, run_id


models_to_run = model_config["models"]
# run_single_model(
#     model_name=models_to_run[0]["name"],
#     model_config=models_to_run[0]["params"],
#     general_config=general_config,
#     results_path="./data/results",
# )


# 1. Group the models by backend (deep learning vs sklearn)
# 2. Get from the slurm config yaml the required ressources for each backend
# 3. Start the jobs in parallel by backend groups, setting the slurm config accordingly + get submitit jobs
# 4. Await the jobs and collect the results

results = run_jobs(
    run_fn=run_single_model,
    fn_kwargs_list=[
        {
            "model_name": model["name"],
            "model_config": model["params"],
            "general_config": general_config,
            "results_path": "./data/results",
        }
        for model in models_to_run
    ],
    parallel_type="slurm",
    slurm_cfg={
        "slurm_partition": "parietal,normal,gpu",
        "tasks_per_node": 1,           # == --ntasks=1 (on 1 node)
        "slurm_gpus_per_task": 1,            # == --gpus-per-task=1 (recommended here)
        "slurm_cpus_per_gpu": 10, 
        "timeout_min": 900,
    },
    n_jobs=50,
)

import warnings
for result in results:
    if not result.ok:
        warnings.warn(f"Job failed with exception: {result.exception}") 

metrics_dir = Path("./data/results/parquet/analysis_results")
global_path = metrics_dir / "metrics.parquet"

if global_path.exists():
    df_global = pd.read_parquet(global_path)
    existing_run_ids = set(df_global["run_id"].unique())
else:
    df_global = None
    existing_run_ids = set()

new_dfs = []

for p in metrics_dir.glob("metrics_*.parquet"):
    run_id = p.stem.replace("metrics_", "")
    if run_id not in existing_run_ids:
        new_dfs.append(pd.read_parquet(p))

if new_dfs:
    df_new = pd.concat(new_dfs, ignore_index=True)
    if df_global is not None:
        df_out = pd.concat([df_global, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_parquet(global_path, index=False)


