from typing import Tuple
import numpy as np

from diff_benchmark.models.model_configurations import get_model
from diff_benchmark.data.dataloaders import PreprocessedData
from diff_benchmark.data.generate_dataset import CustomDataset

from diff_benchmark.preprocessing.preprocess_demographic_data import (
    DefaultDemographicsPreprocessor,
)

from diff_benchmark.preprocessing.brain_data_preparation import (
    DefaultPipeline,
    ImagePipeline,
)
from diff_benchmark.preprocessing.wrapper_brain_data import DataPreparationBrain
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig


def get_data_pipeline(data_type: str, dataset:DatasetConfig) -> DataPreparationBrain:
    """Factory function to get the appropriate data pipeline based on data_type.
    Args:
        data_type (str): Type of data pipeline to use. One of ['images', 'array'].
        config (dict): Configuration dictionary for the data pipeline.
    Returns:
        DataPreparationBrain: An instance of the selected data pipeline.
    Raises:
        ValueError: If an unknown data_type is provided.
    """
    if data_type == "images":
        print("Using Image Pipeline")
        brain_preparator = ImagePipeline(dataset)
    elif data_type == "array":
        print("Using Default Array Pipeline")
        brain_preparator = DefaultPipeline(dataset)
    else:
        raise ValueError(
            f"Unknown data_type '{data_type}'. Must be one of ['lcot_embed', 'images', 'array']."
        )

    return brain_preparator


def prepare_dataset_and_preprocessed(
    model_name: str,
    model_config: dict,
    config: dict,
) -> Tuple["CustomDataset", "PreprocessedData"]:
    """
    End-to-end data preparation:
    - Extract microstructure features
    - Extract and preprocess demographics
    - Align subjects
    - Build dataset and preprocessed objects
    """
    breakpoint()
    # -------- MODEL & PIPELINE --------
    model = get_model(model_name, model_config)
    data_type = model.data_type

    for dataset2prepare in general_config["datasets"]["datasets_list"]:
        if dataset2prepare["name"] == "camcan":
            dataset = DatasetConfig(
                **dataset2prepare,
                metric_to_compute=general_config["datasets"]["metric_to_compute"],
                scale=general_config["datasets"]["scale"],
            )
    breakpoint()    
    brain_preparator = get_data_pipeline(data_type, dataset)
    brain_df = (
        brain_preparator
        .load_features()
        .reset_index()
    )

    # -------- DEMOGRAPHICS --------
    preprocessor = DefaultDemographicsPreprocessor(
        config["data_paths"]["csv_file"]
    )
    demographics_df = preprocessor.preprocess(
        config["target_columns"]
    )

    # -------- SUBJECT ALIGNMENT --------
    brain_df["subject_id"] = brain_df["subject_id"].astype(str)
    demographics_df["Subject"] = demographics_df["Subject"].astype(str)

    common_subjects = set(brain_df["subject_id"]) & set(demographics_df["Subject"])

    brain_filtered = brain_df[
        brain_df["subject_id"].isin(common_subjects)
    ]
    demographics_filtered = demographics_df[
        demographics_df["Subject"].isin(common_subjects)
    ]

    # -------- DATASET CREATION --------
    X = brain_filtered
    y = np.asarray(demographics_filtered[config["target_columns"][0]])
    gender = np.asarray(demographics_filtered["Gender"])

    dataset = CustomDataset(X, y, gender)
    preprocessed = PreprocessedData(X, y, gender, config=config)

    return dataset, preprocessed

import argparse
from diff_benchmark.utils.config_loader import load_configs

parser = argparse.ArgumentParser()
parser.add_argument(
    "--methods", nargs="+", type=str, default=["2dcnn_torch"], help="Method to use"
)
args = parser.parse_args()

general_config, model_config = load_configs(args)

breakpoint()

prepare_dataset_and_preprocessed(
    model_name=args.methods[0],
    model_config=model_config,
    config=general_config,
)