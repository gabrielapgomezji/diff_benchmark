from pathlib import Path

import yaml

general_config_path = Path(__file__).parent.parent / "config/configuration_general.yaml"
with open(general_config_path, "r", encoding="utf-8") as f:
    general_config = yaml.safe_load(f)

from diff_benchmark.preprocessing.brain_feature_extraction import (
    DefaultPipeline,
)
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig

for dataset2prepare in general_config["datasets"]["datasets_list"]:
    if dataset2prepare["name"] == "abide":
        dataset = DatasetConfig(
            **dataset2prepare,
            metric_to_compute=general_config["datasets"]["metric_to_compute"],
            scale=general_config["datasets"]["scale"],
        )
        brain_preparator = DefaultPipeline(dataset)
        # subject_id = "101915" #hcp
        # subject_id = "01187" # wand
        # subject_id = "CC110037" # camcan
        # subject_id = "29182"  # abide

        # from pathlib import Path
        # def parse_subject_ids(dataset):
        #     base = Path(dataset.base_dir)

        #     glob_patterns = {
        #         "multicenter-bids": "*/sub-*",
        #         "bids": "sub-*",
        #         "hcp": "*",
        #     }

        #     try:
        #         pattern = glob_patterns[dataset.data_reading]
        #     except KeyError:
        #         raise ValueError(f"Unknown data_reading: {dataset.data_reading}")

        #     subjects = []

        #     for p in base.glob(pattern):
        #         name = p.name
        #         sid = name if dataset.data_reading == "hcp" else name[4:]

        #         subjects.append(sid)

        #     return sorted(subjects)

        # subject_list = parse_subject_ids(dataset)

        # brain_preparator._get_required_raw_files(subject_list[0])
        brain_preparator.run_pipeline()

        # subject_id = "76884" # wand
        # brain_preparator.verify_raw_files(subject_id)
        # brain_preparator.compute_microstructure(subject_id)
