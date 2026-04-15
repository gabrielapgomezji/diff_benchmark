import time

import pytest

from diff_benchmark.utils.job_manager import run_jobs


@pytest.fixture
def fn_to_parallelize(request):
    errors = request.param

    def fn(i):
        time.sleep(0.1)
        if errors and i % 10 == 0:
            raise ValueError("dumb error")
        return i

    return fn


@pytest.mark.parametrize(
    "fn_to_parallelize",
    [True, False],
    indirect=True,
)
@pytest.mark.parametrize("parallel_type", [None, "joblib"])
def test_parallel_run(parallel_type, fn_to_parallelize, request):

    kwargs_list = [{"i": j} for j in range(10)]

    errors = request.node.callspec.params["fn_to_parallelize"]

    if errors:
        with pytest.raises(ValueError, match="dumb error"):
            run_jobs(
                fn_to_parallelize,
                fn_kwargs_list=kwargs_list,
                parallel_type=parallel_type,
                n_jobs=2,
            )
    else:
        results = run_jobs(
            fn_to_parallelize,
            fn_kwargs_list=kwargs_list,
            parallel_type=parallel_type,
            n_jobs=2,
        )
        assert len(results) == 10


@pytest.mark.parametrize(
    "fn_to_parallelize",
    [True, False],
    indirect=True,
)
def test_parallel_run_slurm(fn_to_parallelize, request):

    kwargs_list = [{"i": j} for j in range(1000)]

    slurm_cfg = {"cpus_per_task": 1, "timeout_min": 10}

    errors = request.node.callspec.params["fn_to_parallelize"]

    if errors:
        with pytest.raises(ValueError, match="dumb error"):
            run_jobs(
                fn_to_parallelize,
                fn_kwargs_list=kwargs_list,
                parallel_type="slurm",
                slurm_cfg=slurm_cfg,
                n_jobs=500,
            )
    else:
        results = run_jobs(
            fn_to_parallelize,
            fn_kwargs_list=kwargs_list,
            parallel_type="slurm",
            slurm_cfg=slurm_cfg,
            n_jobs=500,
        )
        assert len(results) == 1000
