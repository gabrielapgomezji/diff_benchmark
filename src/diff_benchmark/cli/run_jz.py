import logging
import os
import socket
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from diff_benchmark.analysis.save_results import save_model_results
from diff_benchmark.analysis.true_vs_pred import plot_true_vs_pred
from diff_benchmark.cli.utils import build_config_grid, cartesian_cfgs
from diff_benchmark.data.prepare_data import DatasetPreparation
from diff_benchmark.models.model_configurations import get_model
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.utils.job_manager import run_jobs
from diff_benchmark.utils.logger import configure_logging, setup_logger
from diff_benchmark.utils.parquet_helper import ParquetSaver, metrics_to_rows
from diff_benchmark.utils.run_id import get_learning_curve_id, is_cached, make_run_id
from diff_benchmark.utils.scores import compute_metrics


def _build_prediction_rows(
    run_id: str,
    model_name: str,
    dataset_name: str,
    fold_idx: int,
    split: str,
    target_name: str,
    subject_ids: np.ndarray,
    predictions: np.ndarray,
) -> list[dict]:
    """Build prediction row dicts ready for :class:`ParquetSaver`.

    Args:
        run_id: Unique run identifier.
        model_name: Model name string.
        dataset_name: Dataset name string.
        fold_idx: Cross-validation fold index.
        split: ``"train"`` or ``"test"``.
        target_name: Target variable name.
        subject_ids: Array of subject identifiers for this split.
        predictions: Array of scalar predictions aligned with *subject_ids*.

    Returns:
        List of row dicts, one per subject.
    """
    return [
        {
            "run_id": run_id,
            "model": model_name,
            "dataset": dataset_name,
            "fold": fold_idx,
            "split": split,
            "sample_id": sid,
            "target": target_name,
            "prediction": float(pred),
        }
        for sid, pred in zip(subject_ids, predictions)
    ]


def run_single_model(cfg_og, model_name: str, results_path: Path):
    """Run one experiment: full cross-validation loop for a single model config.

    Persists per-fold metrics, predictions, and experiment metadata to disk
    under ``results_path/experiments/exp_<run_id>/``.  Partial results are
    saved even when a fold crashes.

    Args:
        cfg_og: OmegaConf config for the experiment.
        model_name: Model name used for logging and as a metadata field.
        results_path: Root directory for experiment outputs.

    Returns:
        Tuple of ``(model_name, run_id)`` on completion.
    """
    cfg = OmegaConf.merge(cfg_og)
    logger = setup_logger("Job.run_single_model")

    run_id = cfg.runtime.run_id
    learning_curve_id = get_learning_curve_id(cfg)
    cfg.runtime.learning_curve_id = learning_curve_id

    logger.info(f"Computing a learning curve experiment: {cfg.runtime.learning_curve_exp}")

    experiment_dir = Path(results_path) / "experiments" / f"exp_{run_id}"

    # ------------------------------------------------------------------ #
    # Metadata and directory setup                                        #
    # ------------------------------------------------------------------ #
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

    for sub in ("metrics", "predictions", "debug", "logs"):
        (experiment_dir / sub).mkdir(parents=True, exist_ok=True)

    OmegaConf.save(metadata, experiment_dir / "metadata.yaml")
    OmegaConf.save(cfg, experiment_dir / "config.yaml")

    # ------------------------------------------------------------------ #
    # Data preparation                                                    #
    # ------------------------------------------------------------------ #
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    cluster_cfg = cfg.cluster.paths[dataset_cfg["name"]]

    dataset_selected = DatasetConfig(
        **dataset_cfg,
        base_dir=Path(cluster_cfg.base_dir),
        results_dir=Path(cluster_cfg.results_dir),
    )

    torch_dataset, preprocessed = DatasetPreparation(
        cfg=cfg, source_dataset=dataset_selected
    ).pipeline()

    logger.info("Data preparation completed.")

    # Persist target values once (shared across folds).
    targets_path = experiment_dir / "predictions" / "targets.parquet"
    target_name = cfg.target.target_column[0]
    target_rows = [
        {
            "dataset": dataset_selected.name,
            "sample_id": sid,
            "target": target_name,
            "value": float(v),
        }
        for sid, v in zip(torch_dataset.subject_ids, torch_dataset.targets.numpy())
    ]
    target_saver = ParquetSaver(
        path=targets_path,
        key_columns=["dataset", "sample_id", "target"],
        columns=["dataset", "sample_id", "target", "value"],
    )
    target_saver.add_rows(target_rows)
    target_saver.save()

    specs = preprocessed.get_specs()
    logger.debug(f"Dataset specs: {specs}")

    indices = preprocessed.get_fold_indices()

    logger.info(f"Running model: {model_name} with run_id: {run_id}")

    # ------------------------------------------------------------------ #
    # Prediction saver (shared across folds)                             #
    # ------------------------------------------------------------------ #
    predictions_path = experiment_dir / "predictions" / "predictions.parquet"
    pred_key_cols = ["run_id", "model", "dataset", "fold", "split", "sample_id", "target"]
    pred_saver = ParquetSaver(
        predictions_path,
        key_columns=pred_key_cols,
        columns=pred_key_cols + ["prediction"],
    )

    metrics_rows: list[dict] = []

    # ------------------------------------------------------------------ #
    # Cross-validation loop                                               #
    # ------------------------------------------------------------------ #
    for fold_idx, (train_idx, test_idx) in enumerate(indices):
        try:
            logger.info(f"Run ID: {run_id} — Fold {fold_idx + 1}/{len(indices)}")

            train_loader, test_loader = preprocessed.get_dataloader_fold(
                torch_dataset, fold_idx, indices,
                num_workers=cfg.data.num_workers,
                batch_size=cfg.data.batch_size,
            )
            # breakpoint()
            targets_np = torch_dataset.targets.numpy()
            y_train = np.array(targets_np[train_idx]).squeeze()
            y_test = np.array(targets_np[test_idx]).squeeze()

            model = get_model(model_name, OmegaConf.to_container(cfg, resolve=True))
            model.set_fold(fold_idx)

            logger.info(f"Fitting model on fold {fold_idx}...")
            model.fit(train_loader)

            # Training split
            train_pred = model.predict(train_loader)
            train_score = compute_metrics(
                y_train, train_pred, prediction_task=cfg.pred_head.prediction_task
            )
            train_subject_ids = np.asarray(torch_dataset.subject_ids)[train_idx]
            pred_saver.add_rows(
                _build_prediction_rows(
                    run_id, model_name, dataset_selected.name,
                    fold_idx, "train", target_name, train_subject_ids, train_pred,
                )
            )

            # Test split
            test_pred = model.predict(test_loader)
            test_score = compute_metrics(
                y_test, test_pred, prediction_task=cfg.pred_head.prediction_task
            )
            logger.info(
                f"Fold {fold_idx} — Train: {train_score} | Test: {test_score}"
            )

            test_subject_ids = np.asarray(torch_dataset.subject_ids)[test_idx]
            pred_saver.add_rows(
                _build_prediction_rows(
                    run_id, model_name, dataset_selected.name,
                    fold_idx, "test", target_name, test_subject_ids, test_pred,
                )
            )
            pred_saver.save()

            # Accumulate metrics
            shared_meta = dict(
                run_id=run_id,
                model_name=model_name,
                dataset=dataset_selected.name,
                prediction_task=cfg.pred_head.prediction_task,
                tissue_type=cfg.dataset.tissue_type,
                primary_metric=cfg.dataset.metric_to_compute,
                fold=fold_idx,
            )
            metrics_rows.extend(metrics_to_rows(train_score, split="train", **shared_meta))
            metrics_rows.extend(metrics_to_rows(test_score, split="test", **shared_meta))

            metadata["n_folds_completed"] += 1
            OmegaConf.save(metadata, experiment_dir / "metadata.yaml")

        except Exception as e:
            logger.exception(f"Crash in fold {fold_idx} of {run_id}: {e}")

            metadata["status"] = "crashed"
            metadata["error"] = str(e)
            metadata["end_time"] = datetime.utcnow().isoformat()
            OmegaConf.save(metadata, experiment_dir / "metadata.yaml")

            # Persist any metrics collected before the crash.
            if metrics_rows:
                _save_fold_metrics(metrics_rows, experiment_dir)

            break

    # ------------------------------------------------------------------ #
    # Final metadata and metrics save                                     #
    # ------------------------------------------------------------------ #
    n_expected = cfg.data.data_partition.n_splits
    if metadata["n_folds_completed"] == n_expected:
        metadata["status"] = "success"
    elif metadata.get("status") != "crashed":
        metadata["status"] = "partial"

    metadata.setdefault("end_time", datetime.utcnow().isoformat())
    OmegaConf.save(metadata, experiment_dir / "metadata.yaml")

    if metrics_rows:
        _save_fold_metrics(metrics_rows, experiment_dir)

    return model_name, run_id


def _save_fold_metrics(metrics_rows: list[dict], experiment_dir: Path) -> None:
    """Write accumulated per-fold metrics to disk.

    Args:
        metrics_rows: List of metric row dicts.
        experiment_dir: Experiment directory; metrics are written to
            ``metrics/fold_metrics.parquet``.
    """
    metrics_path = experiment_dir / "metrics" / "fold_metrics.parquet"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_parquet(metrics_path, index=False)


def main():
    """CLI entrypoint for running benchmark experiments on Jean Zay.

    Accepts Hydra-style overrides directly from the command line, e.g.::

        python -m diff_benchmark.cli.run_jz cluster=jean_zay model=linear backend=sklearn

    Overrides are applied on top of ``main_jz.yaml`` defaults.  Any remaining
    sweep axes defined in the config's ``choices`` block are expanded into a
    cartesian product of individual experiment configs.
    """
    import sys

    configure_logging(logging.DEBUG)

    results_path = Path("./exp_outputs")
    experiments_root = results_path / "experiments"
    experiments_root.mkdir(parents=True, exist_ok=True)

    # Collect Hydra-style overrides from CLI args (skip the script name itself
    # and any python -m invocation args that don't contain '=').
    cli_overrides = [arg for arg in sys.argv[1:] if "=" in arg]

    with hydra.initialize(version_base="1.3", config_path="pkg://diff_benchmark.configs"):
        cfg = hydra.compose(config_name="main_jz", overrides=cli_overrides)

    # Expand any remaining sweep axes within the config.
    all_confs = cartesian_cfgs(cfg)

    # Attach run IDs and filter out cached experiments.
    filtered_confs = []
    skipped = 0
    for cfg in all_confs:
        force = cfg.runtime.force
        run_id, experiment_hash = make_run_id(cfg, force=force)
        cfg.runtime.run_id = run_id
        cfg.runtime.experiment_hash = experiment_hash

        if is_cached(run_id, experiments_root) and not force:
            print(
                f"Skipping cached experiment: {run_id} "
                f"(hash={experiment_hash}). Use --force to rerun."
            )
            skipped += 1
            continue

        (experiments_root / f"exp_{run_id}").mkdir(parents=True, exist_ok=True)
        filtered_confs.append(cfg)

    if not filtered_confs:
        print("Nothing to run. Exiting.")
        return

    fn_kwargs_list = [
        {"cfg_og": cfg_i, "model_name": cfg_i.model.name, "results_path": results_path}
        for cfg_i in filtered_confs
    ]

    cluster_cfg = all_confs[0].cluster
    parallel_type = cluster_cfg.conf.parallel_type
    if parallel_type not in ("slurm", "joblib"):
        parallel_type = None

    run_jobs(
        run_fn=run_single_model,
        fn_kwargs_list=fn_kwargs_list,
        parallel_type=parallel_type,
        slurm_cfg=cluster_cfg.slurm_cfg,
        n_jobs=cluster_cfg.conf.n_jobs,
        wait_for_results=cluster_cfg.conf.wait_for_results,
    )


if __name__ == "__main__":
    main()
