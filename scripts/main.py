import argparse
import copy
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

from diff_benchmark.analysis.plot_history import plot_history_from_file
from diff_benchmark.analysis.plot_results import plot_folds_predictions_vs_targets
from diff_benchmark.analysis.save_results import (
    is_cached,
    save_fold_results,
    save_model_results,
)
from diff_benchmark.analysis.scores_summary import summarize_folds_to_csv
from diff_benchmark.analysis.true_vs_pred import plot_true_vs_pred
from diff_benchmark.data.dataloaders import PreprocessedData
from diff_benchmark.data.generate_dataset import CustomDataset
from diff_benchmark.models.model_configurations import get_model, make_run_id
from diff_benchmark.preprocessing.preprocess_demographic_data import (
    DefaultDemographicsPreprocessor,
)
from diff_benchmark.utils.config_loader import load_configs
from diff_benchmark.utils.data_pipeline import get_data_pipeline
# from diff_benchmark.utils.job_manager import run_jobs
from diff_benchmark.utils.job_manager_wrap import run_jobs
from diff_benchmark.utils.scores import accuracy_score, compute_metrics
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.data.prepare_data import DatasetPreparation
    
    
parser = argparse.ArgumentParser()
parser.add_argument(
    "--methods", nargs="+", type=str, default=["2dcnn_torch"], help="Method to use"
)
args = parser.parse_args()

general_config, model_config = load_configs(args)


def run_single_model(model_name, model_config, general_config, results_path):
    config = general_config
    
    for dataset2prepare in general_config["datasets"]["datasets_list"]:
        if dataset2prepare["name"] == "hcp":
            dataset = DatasetConfig(
                **dataset2prepare,
                metric_to_compute=general_config["datasets"]["metric_to_compute"],
                scale=general_config["datasets"]["scale"],
            )
            dataset2work = dataset
    
    torch_dataset_preparator = DatasetPreparation(
                model_name=model_name,
                model_config=model_config,
                general_config=general_config,
                source_dataset=dataset2work,
            )
    dataset, preprocessed = torch_dataset_preparator.pipeline()

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
    per_fold_results = []

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

    for fold_idx, (train_idx, test_idx) in enumerate(indices):
        try:
            local_config["fold_idx"] = fold_idx
            # print(
            #     f"Fold {fold_idx+1} - Train samples: {len(train_idx)}, test_samples: {len(test_idx)}"
            # )
            train_loader, test_loader = preprocessed.get_dataloader_fold(
                dataset, fold_idx, indices, batch_size=local_config["batch_size"]
            )
            train_idx, test_idx = indices[fold_idx]
            targets = dataset.targets.numpy()
            y_train = np.array(targets[train_idx]).squeeze()
            y_test = np.array(targets[test_idx]).squeeze()

            local_config["prediction_task"] = config.get(
                "prediction_task", "regression"
            )
            model = get_model(model_name, local_config)

            model.fit(train_loader)
            train_pred = model.predict(train_loader)
            plot_true_vs_pred(
                y_train, train_pred, fold_idx=fold_idx, run_id=run_id, type="train"
            )
            train_score = mean_squared_error(y_train, train_pred)
            # train_score = accuracy_score(y_train, train_pred)
            # train_score = compute_metrics(y_train, train_pred)
            print(train_score)

            train_scores.append(train_score)
            train_preds.append(train_pred.tolist())
            train_targets.append(y_train.tolist())

            test_pred = model.predict(test_loader)
            plot_true_vs_pred(
                y_test, test_pred, fold_idx=fold_idx, run_id=run_id, type="test"
            )
            test_score = mean_squared_error(y_test, test_pred)
            # test_score = accuracy_score(y_test, test_pred)
            # test_score = compute_metrics(y_test, test_pred)
            print(test_score)

            test_scores.append(test_score)
            test_preds.append(test_pred.tolist())
            test_targets.append(y_test.tolist())

            summary["results"]["folds"][f"fold_{fold_idx+1}"] = {
                "train": {
                    # "score": float(train_score),
                    "score": train_score,
                    "predictions": train_pred.tolist(),
                    "targets": y_train.tolist(),
                },
                "test": {
                    # "score": float(test_score),
                    "score": test_score,
                    "predictions": test_pred.tolist(),
                    "targets": y_test.tolist(),
                },
            }

            per_fold_results.append(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "train": {
                        # "score": float(train_score),
                        "score": train_score,
                        "predictions": train_pred.tolist(),
                        "targets": y_train.tolist(),
                    },
                    "test": {
                        # "score": float(test_score),
                        "score": test_score,
                        "predictions": test_pred.tolist(),
                        "targets": y_test.tolist(),
                    },
                }
            )
            training_log_path = (
                Path("./data/results/logs") / f"{run_id}_training_log.json"
            )
            training_log_path.parent.mkdir(parents=True, exist_ok=True)
            training_history_plot_path = (
                Path("./data/results/plots") / f"training_history_{run_id}.png"
            )
            training_history_plot_path.parent.mkdir(parents=True, exist_ok=True)

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
    # summary["results"]["train_average_score"] = float(np.mean(train_scores["accuracy"]))
    # summary["results"]["train_std_score"] = float(np.std(train_scores["accuracy"]))
    # summary["results"]["test_average_score"] = float(np.mean(test_scores["accuracy"]))
    # summary["results"]["test_std_score"] = float(np.std(test_scores["accuracy"]))
    summary["results"]["train_average_score"] = float(np.mean(train_scores))
    summary["results"]["train_std_score"] = float(np.std(train_scores))
    summary["results"]["test_average_score"] = float(np.mean(test_scores))
    summary["results"]["test_std_score"] = float(np.std(test_scores))

    save_model_results(summary, Path(results_path) / "analysis_results")
    return model_name, run_id


models_to_run = model_config["models"]

run_single_model(
    model_name=models_to_run[0]["name"],
    model_config=models_to_run[0]["params"],
    general_config=general_config,
    results_path="./data/results",
)
# results = run_jobs(run_single_model, models_to_run, model_config, general_config)
results = run_jobs(run_fn=run_single_model,
                   fn_kwargs_list=[
                       {
                           "model_name": model["name"],
                           "model_config": model["params"],
                           "general_config": general_config,
                       }
                       for model in models_to_run
                    ],
                    parallel_type=None,
                    slurm_cfg={
                    "cpus_per_task": 1,
                    "timeout_min": 900,
                    "mem_gb": 50,
                    },
                    n_jobs=50,
            )

# results is a list of (model_name, per_fold_results)
# for model_name, run_id in results:
#     print(f"Completed model: {model_name}")
