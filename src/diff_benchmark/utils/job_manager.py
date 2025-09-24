from typing import Callable

import submitit
from joblib import Parallel, delayed


def run_single_process(
    run_fn: Callable,
    models_to_run: list,
    dataset: any,
    preprocessed: any,
    indices: any,
    results_path: str,
) -> list:
    results = []
    for model in models_to_run:
        results.append(
            run_fn(
                model["name"],
                {**model["params"]},
                dataset,
                preprocessed,
                indices,
                results_path,
            )
        )
    return results


def run_with_joblib(
    run_fn: Callable,
    models_to_run: list,
    dataset: any,
    preprocessed: any,
    indices: any,
    results_path: str,
    n_jobs: int = 5,
) -> list:
    """
    Runs a specified function in parallel using joblib's Parallel and delayed.
    Parameters:
        run_fn (callable): The function to run for each model entry.
        models_to_run (list): A list of model entries, each containing a 'name' and 'params'.
        dataset (any): The dataset to be used in the function.
        preprocessed (any): Preprocessed data to be passed to the function.
        indices (any): Indices to be used in the function.
        results_path (str): The path where results will be stored.
        n_jobs (int, optional): The number of jobs to run in parallel. Default is 5.
    Returns:
        list: A list of results returned by the run_fn for each model entry.
    """
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


def run_with_slurm(
    run_fn: Callable,
    models_to_run: list,
    dataset: any,
    preprocessed: any,
    indices: list,
    results_path: str,
    slurm_cfg: dict,
) -> list:
    """
    Runs a function with SLURM job scheduling for multiple models.
    Parameters:
        run_fn (callable): The function to run for each model.
        models_to_run (list): A list of dictionaries, each containing the model name and parameters.
        dataset (any): The dataset to be used in the function.
        preprocessed (any): Indicates whether the dataset is preprocessed.
        indices (list): A list of indices to be used in the function.
        results_path (str): The path where results will be stored.
        slurm_cfg (dict): A configuration dictionary for SLURM parameters, including:
            - log_folder (str): The folder for SLURM logs.
            - mem_gb (int): Memory in GB to allocate for each job.
            - gpus_per_node (int): Number of GPUs to allocate per node.
            - cpus_per_task (int): Number of CPUs to allocate per task.
            - timeout_min (int): Timeout in minutes for each job.
            - partition (str): The SLURM partition to use.
    Returns:
        list: A list of results from the executed jobs.
    """
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


def run_jobs(
    run_fn: Callable,
    models_to_run: list,
    dataset: any,
    preprocessed: any,
    indices: list,
    config: dict,
) -> any:
    """
    Runs jobs using either SLURM or Joblib based on the configuration provided.
    Parameters:
        run_fn (callable): The function to run for each job.
        models_to_run (list): A list of models to be processed.
        dataset (any): The dataset to be used in the jobs.
        preprocessed (any): Indicates whether the dataset is preprocessed.
        indices (list): A list of indices to specify which jobs to run.
        config (dict): A configuration dictionary that may contain:
            - use_slurm (bool): If True, use SLURM for job management.
            - results_path (str): Path to save the results (default is "./data").
            - slurm (dict): Additional SLURM configuration options.
            - n_jobs (int): Number of jobs to run in parallel with Joblib (default is 5).
    Returns:
        Any: The result of the job execution, which depends on the implementation of
        run_with_slurm or run_with_joblib.
    """

    if config.get("use_slurm", False):
        print("Running with SLURM...")
        return run_with_slurm(
            run_fn,
            models_to_run,
            dataset,
            preprocessed,
            indices,
            config.get("results_path_logs", "./data"),
            config.get("slurm", {}),
        )

    if config.get("use_joblib", False):
        print("Running with Joblib...")
        return run_with_joblib(
            run_fn,
            models_to_run,
            dataset,
            preprocessed,
            indices,
            config.get("results_path_logs", "./data"),
            config.get("n_jobs", 5),
        )

    print("Running in a single process...")
    return run_single_process(
        run_fn,
        models_to_run,
        dataset,
        preprocessed,
        indices,
        config.get("results_path_logs", "./data"),
    )
