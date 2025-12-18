from pathlib import Path

from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.preprocessing.preprocess_demographic_data import (
    DefaultDemographicsPreprocessor,
)
from diff_benchmark.preprocessing.wrapper_brain_data_bids import DefaultPipeline, DefaultMulticenterPipeline
import bids
import yaml

general_config_path = (
        Path(__file__).parent.parent / "config/configuration_general.yaml"
    )
with open(general_config_path, "r", encoding="utf-8") as f:
    general_config = yaml.safe_load(f)
for dataset2prepare in general_config["datasets"]["datasets_list"]:
    if dataset2prepare["name"] == "camcan":
        dataset = DatasetConfig(**dataset2prepare, metric_to_compute=general_config["datasets"]["metric_to_compute"], scale=general_config["datasets"]["scale"])
        layout = bids.BIDSLayout(str(dataset.base_dir), derivatives=(Path(dataset.base_dir) / "derivatives"), validate=False)
        breakpoint()
        cog_file = layout.get_file('participants.tsv').path
        preprocessor = DefaultDemographicsPreprocessor(cog_file)
        demographics_df = preprocessor.preprocess(general_config["target_columns"])
breakpoint()