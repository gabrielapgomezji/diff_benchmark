import argparse

from diff_benchmark.data.prepare_data import DatasetPreparation
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.utils.config_loader import load_configs

parser = argparse.ArgumentParser()
parser.add_argument(
    "--methods", nargs="+", type=str, default=["2dcnn_torch"], help="Method to use"
)
args = parser.parse_args()

general_config, model_config = load_configs(args)

models_to_run = model_config["models"]
for dataset2prepare in general_config["datasets"]["datasets_list"]:
    if dataset2prepare["name"] == "abide":
        dataset = DatasetConfig(
            **dataset2prepare,
            metric_to_compute=general_config["datasets"]["metric_to_compute"],
            scale=general_config["datasets"]["scale"],
        )
        dataset2work = dataset

torch_dataset_preparator = DatasetPreparation(
    model_name=models_to_run[0]["name"],
    model_config=models_to_run[0],
    general_config=general_config,
    source_dataset=dataset2work,
)
# breakpoint()
torch_dataset, preprocessed = torch_dataset_preparator.pipeline()
