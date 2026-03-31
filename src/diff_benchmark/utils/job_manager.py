from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

import submitit
from joblib import Parallel, delayed


@dataclass
class JobResult:
    ok: bool
    value: Any = None
    error: str | None = None
    traceback: str | None = None


def fn_error_catcher(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that wraps a function to ensure exceptions are logged and re-raised.
    This preserves full traceback visibility in local runs and SLURM/Submitit logs,
    and avoids silent failures that would otherwise look like successful jobs.
    Args:
        fn: A callable function to be wrapped.
    Returns:
        A wrapped function that takes a dictionary of keyword arguments and returns
        the wrapped function value on success.
    Raises:
        Re-raises any exception from the wrapped function after logging it.
    """

    @wraps(fn)
    def wrapped(kwargs):
        try:
            return fn(**kwargs)
        except Exception:
            logging.getLogger(__name__).exception("Job execution failed for kwargs=%s", kwargs)
            sys.stdout.flush()
            sys.stderr.flush()
            raise

    return wrapped


def run_jobs(
    run_fn: Callable[..., Any],
    fn_kwargs_list: list[dict[str, Any]],
    parallel_type: str | None,
    n_jobs: int = 1,
    slurm_cfg: dict[str, Any] | None = None,
    wait_for_results: bool = True,
) -> list[Any]:
    """Execute a function multiple times with different keyword arguments.
    Supports sequential, joblib-based parallel, and SLURM-based distributed execution.
    Args:
        run_fn: The function to execute. Will be wrapped with error handling.
        fn_kwargs_list: List of dictionaries containing keyword arguments for each function call.
        parallel_type: Execution mode. One of:
            - None: Sequential execution
            - "joblib": Parallel execution using joblib
            - "slurm": Distributed execution using SLURM
        n_jobs: Number of parallel jobs. Default is 1.
            For joblib: number of parallel workers.
            For slurm: array parallelism level.
        slurm_cfg: Configuration dictionary for SLURM execution. Required if parallel_type="slurm".
            Can contain "log_folder" (default "./slurm_logs") and other SLURM parameters.
    Returns:
        List of function return values for completed jobs.
    Raises:
        ValueError: If parallel_type is "slurm" but slurm_cfg is None, or if parallel_type is unknown.
    Warns:
        UserWarning: If parallel_type="joblib" and n_jobs <= 1, as this may not provide speedup.
    """
    fn_to_run = fn_error_catcher(run_fn)

    if parallel_type is None:
        return [fn_to_run(kw) for kw in fn_kwargs_list]

    if parallel_type == "joblib":
        if n_jobs <= 1:
            warnings.warn(f"n_jobs={n_jobs} may not provide parallel speedup.")
        return Parallel(n_jobs=n_jobs)(delayed(fn_to_run)(kw) for kw in fn_kwargs_list)

    if parallel_type == "slurm":
        if slurm_cfg is None:
            raise ValueError("slurm_cfg must be provided for slurm mode")

        log_folder = slurm_cfg.get("log_folder", "./slurm_logs")

        ex = submitit.AutoExecutor(folder=log_folder)
        ex.update_parameters(**slurm_cfg, slurm_array_parallelism=n_jobs)

        jobs = ex.map_array(fn_to_run, fn_kwargs_list)

        if wait_for_results:
            return [j.result() for j in jobs]
        return jobs

    raise ValueError(f"Unknown parallel_type: {parallel_type!r}")
