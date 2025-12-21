from typing import Callable

import submitit
from joblib import Parallel, delayed


def run_single_process(
    run_fn: Callable,
    models_to_run: list,
    results_path: str,
    general_config: dict,
) -> list:
    """Runs a specified function in a single process for multiple models.
    Parameters:
        run_fn (callable): The function to run for each model entry.
        models_to_run (list): A list of model entries, each containing a 'name' and 'params'.
        results_path (str): The path where results will be stored.
        general_config (dict): General configuration dictionary to be passed to the run function.
    Returns:
        list: A list of results returned by the run_fn for each model entry.
    """
    results = []
    for model in models_to_run:
        results.append(
            run_fn(
                model["name"],
                {**model["params"]},
                general_config,
                results_path,
            )
        )
    return results


def run_with_joblib(
    run_fn: Callable,
    models_to_run: list,
    results_path: str,
    general_config: dict,
    n_jobs: int = 5,
) -> list:
    """
    Runs a specified function in parallel using joblib's Parallel and delayed.
    Parameters:
        run_fn (callable): The function to run for each model entry.
        models_to_run (list): A list of model entries, each containing a 'name' and 'params'.
        results_path (str): The path where results will be stored.
        general_config (dict): General configuration dictionary to be passed to the run function.
        n_jobs (int): The number of parallel jobs to run.
    Returns:
        list: A list of results returned by the run_fn for each model entry.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_fn)(
            model_entry["name"],
            {**model_entry["params"]},
            general_config,
            results_path,
        )
        for model_entry in models_to_run
    )
    return results


def run_with_slurm(
    run_fn: Callable,
    models_to_run: list,
    results_path: str,
    slurm_cfg: dict,
    general_config: dict,
) -> list:
    """
    Runs a function with SLURM job scheduling for multiple models.
    Parameters:
        run_fn (callable): The function to run for each model.
        models_to_run (list): A list of dictionaries, each containing the model name and parameters.
        results_path (str): The path where results will be stored.
        slurm_cfg (dict): A configuration dictionary for SLURM parameters, including:
            - log_folder (str): The folder for SLURM logs.
            - mem_gb (int): Memory in GB to allocate for each job.
            - gpus_per_node (int): Number of GPUs to allocate per node.
            - cpus_per_task (int): Number of CPUs to allocate per task.
            - timeout_min (int): Timeout in minutes for each job.
            - partition (str): The SLURM partition to use.
        general_config (dict): General configuration dictionary to be passed to the run function.
    Returns:
        list: A list of results returned by the run_fn for each model entry.
    """
    submitit.slurm.slurm.SlurmJob.USE_SQUEUE = True
    executor = submitit.AutoExecutor(folder=slurm_cfg.get("log_folder", "./slurm_logs"))
    if slurm_cfg["jean_zay"]:
        executor.update_parameters(
            tasks_per_node=1,
            gpus_per_task=1,
            cpus_per_task=slurm_cfg.get("cpus_per_task", 1),
            timeout_min=slurm_cfg.get("timeout_min", 10),
            slurm_additional_parameters={
                "account": "qlr@v100",
                "gres": "gpu:1",
                "constraint": "v100-32g",
            },
            setup=["module purge", "module load pytorch-gpu/py3/2.4.0"],
            slurm_setup=[
                "export OMP_NUM_THREADS=1",
                "export MKL_NUM_THREADS=1",
                "export OPENBLAS_NUM_THREADS=1",
                "export NUMEXPR_NUM_THREADS=1",
            ],
        )
    else:
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
            general_config,
            results_path,
        )
        jobs.append(job)

    # This will block until results are ready (remove if you want async)
    results = [job.result() for job in jobs]
    return results


def run_jobs(
    run_fn: Callable,
    models_to_run: list,
    config: dict,
    general_config: dict,
) -> any:
    """
    Runs jobs using either SLURM, Joblib or single process based on the configuration provided.
    Args:
        run_fn (callable): The function to run for each job.
        models_to_run (list): A list of models to be processed.
        config (dict): Configuration dictionary (not used in this function).
        general_config (dict): General configuration dictionary containing flags for SLURM and Joblib.
    Returns:
        any: The results from the executed jobs, which depends on the implementation of
        run_with_slurm or run_with_joblib.
    """
    if general_config.get("use_slurm", False):
        print("Running with SLURM...")
        return run_with_slurm(
            run_fn,
            models_to_run,
            general_config.get("results_path_logs", "./data"),
            general_config.get("slurm", {}),
            general_config,
        )

    if general_config.get("use_joblib", False):
        print("Running with Joblib...")
        return run_with_joblib(
            run_fn,
            models_to_run,
            general_config.get("results_path_logs", "./data"),
            general_config,
            general_config.get("n_jobs", 5),
        )

    print("Running in a single process...")
    return run_single_process(
        run_fn,
        models_to_run,
        general_config.get("results_path_logs", "./data"),
        general_config,
    )
