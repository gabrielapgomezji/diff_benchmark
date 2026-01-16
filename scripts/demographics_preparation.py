from pathlib import Path

import bids
import yaml

from diff_benchmark.preprocessing.brain_data_preparation import (
    DefaultMulticenterPipeline,
    DefaultPipeline,
)
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.preprocessing.preparation_pipeline_unified import DemographicsPreparationPipeline

general_config_path = Path(__file__).parent.parent / "config/configuration_general.yaml"
with open(general_config_path, "r", encoding="utf-8") as f:
    general_config = yaml.safe_load(f)

for dataset_to_prepare in general_config["datasets"]["datasets_list"]:
    if dataset_to_prepare["name"] == "abide":
        dataset = DatasetConfig(
            **dataset_to_prepare,
            metric_to_compute=general_config["datasets"]["metric_to_compute"],
            scale=general_config["datasets"]["scale"],
        )
        if dataset_to_prepare["name"] == "abide":
            center_dirs = [
                p
                for p in Path(dataset.base_dir).iterdir()
                if p.is_dir() and not p.name.startswith(".")
            ]
            participants_files = []

            for center_dir in center_dirs:
                layout = bids.BIDSLayout(
                    str(center_dir),
                    derivatives=center_dir / "derivatives",
                    validate=False,
                )
                participants_tsv = layout.get_file("participants.tsv").path
                participants_files.append(participants_tsv)
            cog_file = participants_files

        else:
            layout = bids.BIDSLayout(
                str(dataset.base_dir),
                derivatives=(Path(dataset.base_dir) / "derivatives"),
                validate=False,
            )
            cog_file = layout.get_file("participants.tsv").path

        preprocessor = DemographicsPreparationPipeline(cog_file)
        demographics_df = preprocessor.preprocess(general_config["target_columns"])
