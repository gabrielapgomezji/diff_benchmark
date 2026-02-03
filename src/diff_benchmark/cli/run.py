from pathlib import Path
import os

import numpy as np
import pandas as pd

import logging
from diff_benchmark.analysis.save_results import (
    #is_cached,
    save_model_results,
)
from diff_benchmark.analysis.true_vs_pred import plot_true_vs_pred
from diff_benchmark.data.prepare_data import DatasetPreparation
from diff_benchmark.models.model_configurations import get_model#, make_run_id
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.utils.parquet_helper import ParquetSaver, metrics_to_rows
from diff_benchmark.utils.job_manager import run_jobs
from diff_benchmark.utils.scores import compute_metrics
from diff_benchmark.utils.summary_saver import update_summary, compute_summary_stats
from diff_benchmark.utils.logger import setup_logger, configure_logging
from omegaconf import OmegaConf
from diff_benchmark.cli.utils import build_config_grid, cartesian_cfgs
from diff_benchmark.utils.run_id import make_run_id, is_cached
from datetime import datetime
import socket


def run_single_model(cfg_og, model_name, results_path):
    cfg = OmegaConf.merge(cfg_og)
    logger = setup_logger("Job.run_single_model")
    metrics_rows = []

    # run_id = make_run_id(cfg.model.name, cfg)
    # cfg.runtime.run_id = run_id
    run_id = cfg.runtime.run_id
    
    experiment_dir = (
        Path(results_path)
        / "experiments"
        / f"exp_{run_id}"
    )

    metadata = {
        "run_id": run_id,
        "experiment_hash": cfg.runtime.experiment_hash,
        "model": model_name,
        "dataset": cfg.dataset.name,
        "status": "running",
        "n_folds_expected": cfg.data.data_partition.n_splits,
        "n_folds_completed": 0,
        "start_time": datetime.utcnow().isoformat(),
        "hostname": socket.gethostname(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }

    OmegaConf.save(metadata, experiment_dir / "metadata.yaml")
    # Create directory tree
    (experiment_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "debug").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, experiment_dir / "config.yaml")
    
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    cluster_cfg = cfg.cluster.paths[dataset_cfg["name"]]
    
    dataset_selected = DatasetConfig(
        **dataset_cfg,
        base_dir=Path(cluster_cfg.base_dir),
        results_dir=Path(cluster_cfg.results_dir),
    )


    torch_dataset_preparator = DatasetPreparation(
        cfg=cfg,
        source_dataset=dataset_selected,
    )

    dataset, preprocessed = torch_dataset_preparator.pipeline()
    print("Data preparation completed.")
    targets_path = experiment_dir / "predictions" / "targets.parquet"
    targets_path.parent.mkdir(parents=True, exist_ok=True)

    target_name = cfg.target.target_column[0]
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
    logger.debug(f"Dataset specs: {specs}")
    print(f"Dataset specs: {specs}")

    indices = preprocessed.get_fold_indices()

    logger.info(f"Running model: {model_name} with run_id: {run_id}")
    print(f"Running model: {model_name} with run_id: {run_id}")

    train_scores, test_scores = [], []
    train_preds, test_preds = [], []
    train_targets, test_targets = [], []

    # summary = {
    #     "model_name": model_name,
    #     "config": OmegaConf.to_container(cfg, resolve=True),
    #     "results": {
    #         "train_average_score": None,  # will fill after loop
    #         "train_std_score": None,  # will fill after loop
    #         "test_average_score": None,  # will fill after loop
    #         "test_std_score": None,  # will fill after loop
    #         "number_folds": len(indices),
    #         "folds": {},  # will fill inside loop
    #     },
    # }
    # # cfg_path = Path(results_path) / "analysis_results" / f"{run_id}_config.yaml" # CHECK TO INCLUDE
    # # OmegaConf.save(cfg, cfg_path) # CHECK TO INCLUDE

    # save_model_results(
    #     summary, Path(results_path) / "analysis_results" / f"{run_id}_partial.json"
    # )
    
    predictions_path = experiment_dir / "predictions" / "predictions.parquet"
    key_cols = ["run_id", "model", "dataset", "fold", "split", "sample_id", "target"]
    pred_saver = ParquetSaver(predictions_path, key_columns=key_cols,
                            columns=[
                                "run_id", "model", "dataset", "fold", "split",
                                "sample_id", "target", "prediction"
                            ])

    for fold_idx, (train_idx, test_idx) in enumerate(indices):
        try:
            logger.info(f"Run ID: {run_id} - Fold {fold_idx+1}/{len(indices)}")
            print(f"Run ID: {run_id} - Fold {fold_idx+1}/{len(indices)}")
            # local_config["fold_idx"] = fold_idx
            train_loader, test_loader = preprocessed.get_dataloader_fold(
                dataset,
                fold_idx,
                indices,
                num_workers=cfg.data.num_workers,
                batch_size=cfg.data.batch_size,
            )
            train_idx, test_idx = indices[fold_idx]
            targets = dataset.targets.numpy()
            y_train = np.array(targets[train_idx]).squeeze()
            y_test = np.array(targets[test_idx]).squeeze()

            # local_config["backbone"]["prediction_task"] = config.get(
            #     "prediction_task", "regression"
            # )
            # local_config["backend"]["run_id"] = run_id
            model = get_model(cfg.model.name, 
                          OmegaConf.to_container(cfg, resolve=True),)

            model.set_fold(fold_idx)
            model.fit(train_loader)
            train_pred = model.predict(train_loader)
            
            # plot_true_vs_pred(
            #     y_train, train_pred, fold_idx=fold_idx, run_id=run_id, type="train"
            # )
            train_score = compute_metrics(y_train, train_pred, prediction_task=cfg.pred_head.prediction_task)

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
            # plot_true_vs_pred(
            #     y_test, test_pred, fold_idx=fold_idx, run_id=run_id, type="test"
            # )
            test_score = compute_metrics(y_test, test_pred, prediction_task=cfg.pred_head.prediction_task)
            logger.info(f"Fold {fold_idx} - Train score: {train_score}, Test score: {test_score}")
            print(f"Fold {fold_idx} - Train score: {train_score}, Test score: {test_score}")

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

            primary_metric = {"binary_classification": "accuracy", "regression": "mse"}[cfg.pred_head.prediction_task]
            # summary = update_summary(summary, fold_idx, train_score, test_score, y_train, train_pred, y_test, test_pred, primary_metric)

            metrics_rows.extend(
                metrics_to_rows(
                    train_score,
                    run_id=run_id,
                    model_name=model_name,
                    dataset=dataset_selected.name,
                    prediction_task=cfg.pred_head.prediction_task,
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
                    prediction_task=cfg.pred_head.prediction_task,
                    fold=fold_idx,
                    split="test",
                )
            )

            # save_model_results(
            #     summary,
            #     Path(results_path) / "analysis_results" / f"{run_id}_partial.json",
            # )
            metadata["n_folds_completed"] += 1
            OmegaConf.save(metadata, experiment_dir / "metadata.yaml")
        except Exception as e:
            logger.exception(f"Crash in fold {fold_idx} of {run_id}: {e}")
            print(f"Crash in fold {fold_idx} of {run_id}: {e}")
            # save_model_results(
            #     summary,
            #     Path(results_path) / "analysis_results" / f"{run_id}_crashed.json",
            # )
            metadata["status"] = "crashed"
            metadata["error"] = str(e)
            OmegaConf.save(metadata, experiment_dir / "metadata.yaml")
            raise
    
    metadata["status"] = "success"
    metadata["end_time"] = datetime.utcnow().isoformat()
    OmegaConf.save(metadata, experiment_dir / "metadata.yaml")
    metrics_path = experiment_dir / "metrics" / "fold_metrics.parquet"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics_rows)
    df.to_parquet(metrics_path, index=False)

    # summary["results"]["train_average_score"], summary["results"]["train_std_score"] = compute_summary_stats(train_scores, primary_metric)
    # summary["results"]["test_average_score"], summary["results"]["test_std_score"] = compute_summary_stats(test_scores, primary_metric)

    # save_model_results(summary, Path(results_path) / "analysis_results")
    return model_name, run_id

# KEEP RESULTS AND UPDATE GLOBAL METRICS FILE
# import warnings
# for result in results:
#     if not result.ok:
#         warnings.warn(f"Job failed:\n{result.traceback}")

# metrics_dir = Path("./data/results/parquet/analysis_results")
# global_path = metrics_dir / "metrics.parquet"

# if global_path.exists():
#     df_global = pd.read_parquet(global_path)
#     existing_run_ids = set(df_global["run_id"].unique())
# else:
#     df_global = None
#     existing_run_ids = set()

# new_dfs = []

# for p in metrics_dir.glob("metrics_*.parquet"):
#     run_id = p.stem.replace("metrics_", "")
#     if run_id not in existing_run_ids:
#         new_dfs.append(pd.read_parquet(p))

# if new_dfs:
#     df_new = pd.concat(new_dfs, ignore_index=True)
#     if df_global is not None:
#         df_out = pd.concat([df_global, df_new], ignore_index=True)
#     else:
#         df_out = df_new

#     df_out.to_parquet(global_path, index=False)

#################

import hydra
from omegaconf import DictConfig
import itertools


def cartesian_overrides(sweep_cfg: DictConfig):
    keys = list(sweep_cfg.keys())
    vals = [list(sweep_cfg[k]) for k in keys]

    overrides = []
    for combo in itertools.product(*vals):
        overrides.append(
            [f"{k}={v}" for k, v in zip(keys, combo)]
        )

    return overrides


CONFIG_DIR = str(Path(__file__).parent.parent / "configs")
def main():
    configure_logging(logging.DEBUG)
    results_path = Path("./exp_outputs")
    experiments_root = results_path / "experiments"
    experiments_root.mkdir(parents=True, exist_ok=True)
    
    # 1) compose base once
    with hydra.initialize(version_base="1.3", config_path="pkg://diff_benchmark.configs"):
        base = hydra.compose(config_name="main")

        override_sets = cartesian_overrides(base.choices)
        job_cfgs = [hydra.compose(config_name="main", overrides=ovr) for ovr in override_sets]

    # 2) job_cfgs are fully composed, each with different defaults selections
    all_confs = []
    for job_cfg in job_cfgs:
        all_confs.extend(cartesian_cfgs(job_cfg))  # or just run_one(job_cfg)
    
    filtered_confs = []
    skipped = 0

    # 3) Compute IDs + cache filtering
    for cfg in all_confs:
        force = cfg.runtime.force

        run_id, experiment_hash = make_run_id(cfg, force=force)

        # Attach IDs to config *before* job submission
        cfg.runtime.run_id = run_id
        cfg.runtime.experiment_hash = experiment_hash
        experiment_dir = experiments_root / f"exp_{run_id}"
        if is_cached(run_id, experiments_root) and not force:
            print(
                f"Skipping cached experiment: {run_id} "
                f"(hash={experiment_hash}). Use --force to rerun."
            )
            skipped += 1
            continue
        experiment_dir.mkdir(parents=True, exist_ok=True)
        filtered_confs.append(cfg)
 
    if not filtered_confs:
        print("Nothing to run. Exiting.")
        return

    fn_kwargs_list = [
        {
            "cfg_og": cfg_i,
            "model_name": cfg_i.model.name,
            "results_path": results_path,
        }
        for cfg_i in filtered_confs
    ]

    cluster_cfg = all_confs[0].cluster
    parallel_type = (
        None
        if cluster_cfg.conf.parallel_type not in ["slurm", "joblib"]
        else cluster_cfg.conf.parallel_type
    )

    results = run_jobs(
        run_fn=run_single_model,
        fn_kwargs_list=fn_kwargs_list,
        parallel_type=parallel_type,
        slurm_cfg=cluster_cfg.slurm_cfg,
        n_jobs=cluster_cfg.conf.n_jobs,
        wait_for_results=cluster_cfg.conf.wait_for_results,
    )

if __name__ == "__main__":
    main()