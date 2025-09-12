from joblib import Parallel, delayed
import submitit

def run_with_joblib(run_fn, models_to_run, dataset, preprocessed, indices, results_path, n_jobs=5):
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_fn)(
            model_entry["name"],
            {**model_entry["params"]},
            dataset,
            preprocessed,
            indices,
            results_path,
        )
        for model_entry in models_to_run
    )
    return results


def run_with_slurm(run_fn, models_to_run, dataset, preprocessed, indices, results_path, slurm_cfg): 
    submitit.slurm.slurm.SlurmJob.USE_SQUEUE = True  
    executor = submitit.AutoExecutor(folder=slurm_cfg.get("log_folder", "./slurm_logs"))
    executor.update_parameters(
        mem_gb=slurm_cfg.get("mem_gb", 32),
        gpus_per_node=slurm_cfg.get("gpus_per_node", 0),
        tasks_per_node=1,
        cpus_per_task=slurm_cfg.get("cpus_per_task", 4),
        timeout_min=slurm_cfg.get("timeout_min", 120),
        slurm_partition=slurm_cfg.get("partition", "cpu"),
    )

    jobs = []
    for model_entry in models_to_run:
        job = executor.submit(
            run_fn,
            model_entry["name"],
            {**model_entry["params"]},
            dataset,
            preprocessed,
            indices,
            results_path,
        )
        jobs.append(job)

    # This will block until results are ready (remove if you want async)
    results = [job.result() for job in jobs]
    return results


def run_jobs(run_fn, models_to_run, dataset, preprocessed, indices, config):
    if config.get("use_slurm", False):
        print("Running with SLURM...")
        return run_with_slurm(
            run_fn, models_to_run, dataset, preprocessed, indices,
            config.get("results_path", "./data"),
            config.get("slurm", {})
        )
    else:
        print("Running with Joblib...")
        return run_with_joblib(
            run_fn, models_to_run, dataset, preprocessed, indices,
            config.get("results_path", "./data"),
            config.get("n_jobs", 5)
        )
