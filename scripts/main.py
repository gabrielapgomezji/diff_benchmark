import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from joblib import Parallel, delayed

from diff_benchmark.analysis.plot_history import plot_history_from_file
from diff_benchmark.analysis.plot_results import plot_folds_predictions_vs_targets
from diff_benchmark.analysis.save_results import (
    is_cached,
    save_fold_results,
    save_model_results,
)
from diff_benchmark.analysis.scores_summary import summarize_folds_to_csv
from diff_benchmark.dataloaders.dataloaders import PreprocessedData
from diff_benchmark.dataset.generate_dataset import CustomDataset
from diff_benchmark.models.model_configurations import get_model, make_run_id
from diff_benchmark.preprocessing.preprocess_demographic_data import (
    DefaultDemographicsPreprocessor,
)
from diff_benchmark.preprocessing.wrapper_brain_data import (  # LcotEmbedHcpPipeline,
    DefaultHcpPipeline,
    ImageHcpPipeline,
)
from diff_benchmark.scores.scores import accuracy_score  # , mse_score
from diff_benchmark.utils.job_manager import run_jobs

DEBUG = True

config_path = Path(__file__).parent.parent / "configuration.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

json_path = (
    Path(__file__).parent.parent / "src/diff_benchmark/models/model_configurations.json"
)
with open(json_path, "r") as f:
    model_configs = json.load(f)["models"]

# config["metric_to_compute"] = "md"
brain_preparator = ImageHcpPipeline(config)
# brain_preparator = DefaultHcpPipeline(config)
brain_df = brain_preparator.run_pipeline()
brain_df = brain_df.reset_index()


##### NEXT TESTING STEPS
preprocessor = DefaultDemographicsPreprocessor(config["csv_file"])
demographics_df = preprocessor.preprocess(config["target_columns"])


common_subjects = set(brain_df["subject_id"].astype(str)) & set(
    demographics_df["Subject"].astype(str)
)
demographics_filtered = demographics_df[
    demographics_df["Subject"].astype(str).isin(common_subjects)
]
brain_filtered = brain_df[brain_df["subject_id"].astype(str).isin(common_subjects)]

# DATASET GENERATION

X = brain_filtered  # .drop(columns=["subject_id"]).to_numpy()
y = np.array(demographics_filtered["Gender"])
gender = np.array(demographics_filtered["Gender"])

dataset = CustomDataset(X, y, gender)

# ----------- CROSS VALIDATION + TRAINING + TESTING -----------


preprocessed = PreprocessedData(
    X, y, gender, n_splits=config["n_splits"], random_state=config["random_state"]
)

specs = preprocessed.get_specs()
print(specs)

# folds = preprocessed.get_folds_as_dataloaders(batch_size=16)
indices = preprocessed.get_fold_indices()


def run_single_model(model_name, config, dataset, preprocessed, indices, results_path):
    local_config = copy.deepcopy(config)
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
            "input_slices": local_config.get("input_slices"),
            "num_classes": local_config.get("num_classes"),
            "device": local_config.get("device"),
            "learning_rate": local_config.get("learning_rate"),
            "pretrained": local_config.get("pretrained"),
            "freeze_backbone": local_config.get("freeze_backbone"),
            "batch_size": local_config.get("batch_size"),
            "epochs": local_config.get("epochs"),
            "dropout": local_config.get("dropout"),
            "weight_decay": local_config.get("weight_decay"),
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
    save_model_results(
        summary, Path(results_path) / "analysis_results" / f"{run_id}_partial.json"
    )

    for fold_idx, (train_idx, test_idx) in enumerate(indices[:1]):
        try:
            # print(
            #     f"Fold {fold_idx+1} - Train samples: {len(train_idx)}, test_samples: {len(test_idx)}"
            # )
            train_loader, test_loader = preprocessed.get_dataloader_fold(
                dataset, fold_idx, indices, batch_size=local_config["batch_size"]
            )
            # _, y_train, _, _, y_test, _ = preprocessed.get_arrays_from_indices(
            #     dataset, fold_idx, indices
            # )
            train_idx, test_idx = indices[fold_idx]
            targets = dataset.targets.numpy()
            y_train = np.array(targets[train_idx]).squeeze()
            y_test = np.array(targets[test_idx]).squeeze()

            model = get_model(model_name, local_config)

            # --------- Train / Val / Test Model ---------
            # print("Training...")
            # device = torch.device("cpu")
            # model = model.to(device)
            model.fit(train_loader)
            train_pred = model.predict(train_loader)
            train_score = accuracy_score(y_train, train_pred)
            print(train_score)

            train_scores.append(train_score)
            train_preds.append(train_pred.tolist())
            train_targets.append(y_train.tolist())

            # print("Testing...")
            test_pred = model.predict(test_loader)
            test_score = accuracy_score(y_test, test_pred)
            print(test_score)

            test_scores.append(test_score)
            test_preds.append(test_pred.tolist())
            test_targets.append(y_test.tolist())

            summary["results"]["folds"][f"fold_{fold_idx+1}"] = {
                "train": {
                    "score": float(train_score),
                    "predictions": train_pred.tolist(),
                    "targets": y_train.tolist(),
                },
                "test": {
                    "score": float(test_score),
                    "predictions": test_pred.tolist(),
                    "targets": y_test.tolist(),
                },
            }

            # print(f"Done Fold {fold_idx + 1}")
            if DEBUG:
                per_fold_results.append(
                    {
                        "model": model_name,
                        "fold": fold_idx,
                        "train": {
                            "score": float(train_score),
                            "predictions": train_pred.tolist(),
                            "targets": y_train.tolist(),
                        },
                        "test": {
                            "score": float(test_score),
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
                plot_history_from_file(
                    training_log_path, save_path=training_history_plot_path
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

    summary["results"]["train_average_score"] = float(np.mean(train_scores))
    summary["results"]["train_std_score"] = float(np.std(train_scores))
    summary["results"]["test_average_score"] = float(np.mean(test_scores))
    summary["results"]["test_std_score"] = float(np.std(test_scores))

    # print("\n Saving results...")
    # if DEBUG:
    #     save_fold_results(
    #         model_name=model_name,
    #         fold_results=per_fold_results,
    #         output_dir=Path(results_path) / "analysis_results",
    #     )
    save_model_results(summary, Path(results_path) / "analysis_results")
    return model_name, run_id


models_to_run = config["models"]

results = run_jobs(
    run_single_model, models_to_run, dataset, preprocessed, indices, config
)

# results is a list of (model_name, per_fold_results)
for model_name, run_id in results:
    print(f"Completed model: {model_name}")


# ------------ EVALUATION AND ANALYSIS ------------


# -------- PLOT PER FOLD PRED VS TARGETS --------
for model_entry in models_to_run:
    name = model_entry["name"]
    plot_folds_predictions_vs_targets(
        summary_path=Path(config["results_path"])
        / "analysis_results"
        / f"{name}_fold_results.json",
        output_dir=Path(config["results_path"]) / "analysis_results" / "plots",
    )

    summarize_folds_to_csv(
        fold_results_path=Path(config["results_path"])
        / "analysis_results"
        / f"{name}_fold_results.json",
        output_csv_path=Path(config["results_path"])
        / "analysis_results"
        / f"{name}_score_stats.csv",
    )


###### VERY EARLY IN TESTING
# preparator = LcotEmbedHcpPipeline(config)
# preparator.extract_raw_data("100206")
# preparator.compute_microstructure("100206")
# breakpoint()
