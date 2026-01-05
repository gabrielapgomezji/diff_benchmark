from __future__ import annotations

import traceback as tb
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


def fn_error_catcher(fn: Callable[..., Any]) -> Callable[..., JobResult]:
    """
    Decorator that wraps a function to catch exceptions and return a JobResult.
    This decorator executes the wrapped function and captures its return value or any
    exceptions that occur during execution. The result is returned as a JobResult object
    that indicates success or failure along with relevant error information.
    Args:
        fn: A callable function to be wrapped.
    Returns:
        A wrapped function that takes a dictionary of keyword arguments and returns
        a JobResult object. On successful execution, returns JobResult with ok=True
        and the function's return value. On exception, returns JobResult with ok=False,
        the error message as a string, and a formatted traceback.
    Raises:
        None - All exceptions are caught and returned in the JobResult object.
    """

    @wraps(fn)
    def wrapped(kwargs) -> JobResult:
        try:
            return JobResult(ok=True, value=fn(**kwargs))
        except Exception as e:
            return JobResult(ok=False, error=str(e), traceback=tb.format_exc())

    return wrapped


def run_jobs(
    run_fn: Callable[..., Any],
    fn_kwargs_list: list[dict[str, Any]],
    parallel_type: str | None,
    n_jobs: int = 1,
    slurm_cfg: dict[str, Any] | None = None,
) -> list[JobResult]:
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
        List of JobResult objects containing the results of all function executions.
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

        log_folder = slurm_cfg.pop("log_folder", "./slurm_logs")

        ex = submitit.AutoExecutor(folder=log_folder)
        ex.update_parameters(**slurm_cfg, slurm_array_parallelism=n_jobs)

        jobs = ex.map_array(fn_to_run, fn_kwargs_list)

        return [j.result() for j in jobs]

    raise ValueError(f"Unknown parallel_type: {parallel_type!r}")
