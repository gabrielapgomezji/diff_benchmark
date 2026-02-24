import logging
import os
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from diff_benchmark.analysis.save_results import (  # is_cached,
    save_model_results,
)
from diff_benchmark.analysis.true_vs_pred import plot_true_vs_pred
from diff_benchmark.cli.utils import build_config_grid, cartesian_cfgs
from diff_benchmark.data.prepare_data import DatasetPreparation
from diff_benchmark.models.model_configurations import get_model  # , make_run_id
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.utils.job_manager import run_jobs  # , JobResult
from diff_benchmark.utils.logger import configure_logging, setup_logger
from diff_benchmark.utils.parquet_helper import ParquetSaver, metrics_to_rows
from diff_benchmark.utils.run_id import get_learning_curve_id, is_cached, make_run_id
from diff_benchmark.utils.scores import compute_metrics


def run_single_model(cfg_og, model_name, results_path):
    cfg = OmegaConf.merge(cfg_og)
    logger = setup_logger("Job.run_single_model")
    metrics_rows = []

    run_id = cfg.runtime.run_id
    learning_curve_id = get_learning_curve_id(cfg)
    cfg.runtime.learning_curve_id = learning_curve_id
    print(f"Computing a learning curve experiment: {cfg.runtime.learning_curve_exp}")
    experiment_dir = Path(results_path) / "experiments" / f"exp_{run_id}"

    metadata = {
        "run_id": run_id,
        "learning_curve_id": learning_curve_id,
        "experiment_hash": cfg.runtime.experiment_hash,
        "model": model_name,
        "dataset": cfg.dataset.name,
        "tissue_type": cfg.dataset.tissue_type,
        "primary_metric": cfg.dataset.metric_to_compute,
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
        {
            "dataset": dataset_selected.name,
            "sample_id": sid,
            "target": target_name,
            "value": float(v),
        }
        for sid, v in zip(dataset.subject_ids, dataset.targets.numpy())
    ]
    saver = ParquetSaver(
        path=targets_path,
        key_columns=["dataset", "sample_id", "target"],
        columns=["dataset", "sample_id", "target", "value"],
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

    predictions_path = experiment_dir / "predictions" / "predictions.parquet"
    key_cols = ["run_id", "model", "dataset", "fold", "split", "sample_id", "target"]
    pred_saver = ParquetSaver(
        predictions_path,
        key_columns=key_cols,
        columns=[
            "run_id",
            "model",
            "dataset",
            "fold",
            "split",
            "sample_id",
            "target",
            "prediction",
        ],
    )

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

            model = get_model(
                cfg.model.name,
                OmegaConf.to_container(cfg, resolve=True),
            )

            model.set_fold(fold_idx)
            model.fit(train_loader)
            train_pred = model.predict(train_loader)

            train_score = compute_metrics(
                y_train, train_pred, prediction_task=cfg.pred_head.prediction_task
            )

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

            test_score = compute_metrics(
                y_test, test_pred, prediction_task=cfg.pred_head.prediction_task
            )
            logger.info(
                f"Fold {fold_idx} - Train score: {train_score}, Test score: {test_score}"
            )
            print(
                f"Fold {fold_idx} - Train score: {train_score}, Test score: {test_score}"
            )

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

            primary_metric = {"binary_classification": "accuracy", "regression": "mse"}[
                cfg.pred_head.prediction_task
            ]

            metrics_rows.extend(
                metrics_to_rows(
                    train_score,
                    run_id=run_id,
                    model_name=model_name,
                    dataset=dataset_selected.name,
                    prediction_task=cfg.pred_head.prediction_task,
                    tissue_type=cfg.dataset.tissue_type,
                    primary_metric=cfg.dataset.metric_to_compute,
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
                    tissue_type=cfg.dataset.tissue_type,
                    primary_metric=cfg.dataset.metric_to_compute,
                    fold=fold_idx,
                    split="test",
                )
            )

            metadata["n_folds_completed"] += 1
            OmegaConf.save(metadata, experiment_dir / "metadata.yaml")
        except Exception as e:
            logger.exception(f"Crash in fold {fold_idx} of {run_id}: {e}")
            print(f"Crash in fold {fold_idx} of {run_id}: {e}")

            metadata["status"] = "crashed"
            metadata["error"] = str(e)
            metadata["end_time"] = datetime.utcnow().isoformat()
            OmegaConf.save(metadata, experiment_dir / "metadata.yaml")

            # Save partial metrics for completed folds before crashing
            if metrics_rows:
                logger.info(
                    f"Saving partial metrics for {len(metrics_rows)} completed fold(s)"
                )
                metrics_path = experiment_dir / "metrics" / "fold_metrics.parquet"
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(metrics_rows)
                df.to_parquet(metrics_path, index=False)

            # Don't re-raise - continue to save what we have
            break

    # Only mark as success if all folds completed
    if metadata["n_folds_completed"] == cfg.data.data_partition.n_splits:
        metadata["status"] = "success"
    elif metadata.get("status") != "crashed":
        metadata["status"] = "partial"

    if "end_time" not in metadata:
        metadata["end_time"] = datetime.utcnow().isoformat()

    OmegaConf.save(metadata, experiment_dir / "metadata.yaml")

    # Save metrics for all completed folds (success or partial)
    if metrics_rows:
        metrics_path = experiment_dir / "metrics" / "fold_metrics.parquet"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(metrics_rows)
        df.to_parquet(metrics_path, index=False)

    return model_name, run_id


import itertools

import hydra
from omegaconf import DictConfig


def cartesian_overrides(sweep_cfg: DictConfig):
    keys = list(sweep_cfg.keys())
    vals = [list(sweep_cfg[k]) for k in keys]

    overrides = []
    for combo in itertools.product(*vals):
        overrides.append([f"{k}={v}" for k, v in zip(keys, combo)])

    return overrides


CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


def main():
    configure_logging(logging.DEBUG)
    results_path = Path("./exp_outputs")
    experiments_root = results_path / "experiments"
    experiments_root.mkdir(parents=True, exist_ok=True)

    # 1) compose base once
    with hydra.initialize(
        version_base="1.3", config_path="pkg://diff_benchmark.configs"
    ):
        base = hydra.compose(config_name="main")

        override_sets = cartesian_overrides(base.choices)
        job_cfgs = [
            hydra.compose(config_name="main", overrides=ovr) for ovr in override_sets
        ]

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

    # failed = [
    #     (i, r) for i, r in enumerate(results)
    #     if isinstance(r, JobResult) and not r.ok
    # ]
    # if failed:
    #     for i, r in failed:
    #         cfg_i = filtered_confs[i]
    #         print(
    #             f"\n❌ Job {i} FAILED "
    #             f"(model={cfg_i.model.name}, dataset={cfg_i.dataset.name})\n"
    #             f"   Error: {r.error}\n"
    #             f"   Traceback:\n{r.traceback}"
    #         )
    #     raise SystemExit(1)


if __name__ == "__main__":
    main()
