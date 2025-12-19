from pathlib import Path

import yaml


general_config_path = Path(__file__).parent.parent / "config/configuration_general.yaml"
with open(general_config_path, "r", encoding="utf-8") as f:
    general_config = yaml.safe_load(f)
    
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.preprocessing.brain_data_preparation import (
    DefaultPipeline,
)

for dataset2prepare in general_config["datasets"]["datasets_list"]:
    if dataset2prepare["name"] == "hcp":
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

        # def parse_subject_id(dataset: str):
        #     """
        #     Return the numeric part of a subject folder.
        #     Accepts any folder name that contains only digits.
        #     Returns None if not a valid subject directory.
        #     """
        #     from glob import glob

        #     base_folder = dataset.base_dir

        #     if dataset.data_reading == "multicenter-bids":
        #         extra = "/*/*sub-*"
        #     if dataset.data_reading == "bids":
        #         extra = "/*sub-*"
        #     if dataset.data_reading == "hcp":
        #         extra = "/*"
        #     file_list = glob(f"{base_folder}{extra}")
        #     file_list = [
        #         sub_id
        #         for x in file_list
        #         if (sub_id := x.split("/")[-1].replace("sub-", "")).isdigit()
        #     ]
        #     return sorted(file_list)

        # subject_list = parse_subject_id(dataset)
        breakpoint()
        subject_id = "101915" #hcp
        brain_preparator.verify_raw_files(subject_id)
        brain_preparator.compute_microstructure(subject_id)
