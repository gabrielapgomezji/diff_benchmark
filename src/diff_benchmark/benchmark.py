from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import logging

from diff_benchmark.utils.job_manager import run_jobs
from diff_benchmark.cli.run import run_single_model


class Benchmark:
    """
    Programmatic API for running diff-benchmark experiments.
    """

    def __init__(self, config: DictConfig | dict):
        if not isinstance(config, DictConfig):
            config = OmegaConf.create(config)

        self.cfg = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # self.results_path = Path(
        #     self.cfg.runtime.get("results_dir", "./exp_outputs")
        # )
        self.results_path = Path("./exp_outputs")
        self.results_path.mkdir(parents=True, exist_ok=True)

    def run(self):
        """
        Run benchmark experiments.
        """

        parallel_type = (
            None
            if self.cfg.cluster.conf.parallel_type not in ["slurm", "joblib"]
            else self.cfg.cluster.conf.parallel_type
        )

        self.logger.info("Starting benchmark run")
        self.logger.debug(f"Parallel type: {parallel_type}")

        results = run_jobs(
            run_fn=run_single_model,
            fn_kwargs_list=[
                {
                    "cfg_og": self.cfg,
                    "model_name": self.cfg.model.name,
                    "results_path": self.results_path,
                }
            ],
            parallel_type=parallel_type,
            slurm_cfg=self.cfg.cluster.slurm_cfg,
            n_jobs=self.cfg.runtime.get("n_jobs", 1),
        )

        self.logger.info("Benchmark finished")
        return results
