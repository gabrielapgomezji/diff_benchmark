from typing import List, Tuple, Union

import bids
import numpy as np
import pandas as pd

from diff_benchmark.data.dataloaders import PreprocessedData
from diff_benchmark.data.generate_dataset import CustomDataset
from diff_benchmark.models.model_configurations import get_model
from diff_benchmark.preprocessing.brain_data_preparation import (
    DefaultPipeline,
    ImagePipeline,
)
from diff_benchmark.preprocessing.datasets_dataclasses import DatasetConfig
from diff_benchmark.preprocessing.preprocess_demographic_data import (
    DefaultDemographicsPreprocessor,
)
from diff_benchmark.preprocessing.wrapper_brain_base import DataPreparationBrain
from diff_benchmark.utils.logger import setup_logger
from omegaconf import DictConfig, OmegaConf

logger = setup_logger(__name__)



def get_data_pipeline(data_type: str, dataset: DatasetConfig) -> DataPreparationBrain:
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
        logger.info("Using Image Pipeline")
        print("Using Image Pipeline")
        brain_preparator = ImagePipeline(dataset)
    elif data_type == "array":
        logger.info("Using Default Array Pipeline")
        print("Using Default Array Pipeline")
        brain_preparator = DefaultPipeline(dataset)
    else:
        raise ValueError(
            f"Unknown data_type '{data_type}'. Must be one of ['images', 'array']."
        )

    return brain_preparator


class DatasetPreparation:
    """
    End-to-end data preparation:
    - Extract microstructure features
    - Extract and preprocess demographics
    - Align subjects
    - Build dataset and preprocessed objects
    """

    def __init__(
        self,
        cfg: DictConfig,
        source_dataset: DatasetConfig,
    ):
        """
        Initialize the data preparation process.
        """
        self.cfg = cfg
        self.model_name = cfg.model.name
        self.source_dataset = source_dataset

    def _extract_participants_files_from_layouts(
        self,
        layouts: List[bids.BIDSLayout],
    ) -> Union[str, List[str]]:
        """
        Returns a single participants.tsv path or a list (multicenter).
        Args:
            layouts (List[bids.BIDSLayout]): List of BIDS layouts to search.
        Returns:
            Union[str, List[str]]: Path(s) to participants.tsv file(s).
        Raises:
            RuntimeError: If no participants.tsv files are found.
        """
        participants_files = []

        for layout in layouts:
            participants = layout.get_file("participants.tsv")
            if participants is not None:
                participants_files.append(participants.path)

        if not participants_files:
            raise RuntimeError("No participants.tsv found in any BIDS layout")

        return (
            participants_files if len(participants_files) > 1 else participants_files[0]
        )

    def _get_brain_df(self) -> pd.DataFrame:
        """
        Prepare brain DataFrame.
        Returns:
            pd.DataFrame: Preprocessed brain features DataFrame.
        """
        # -------- MODEL & PIPELINE --------

        model = get_model(self.model_name, 
                          OmegaConf.to_container(self.cfg, resolve=True),
                         )

        if hasattr(model, "model"):  # trainer wrapper
            model = model.model
        data_type = model.data_type

        self.brain_preparator = get_data_pipeline(data_type, self.source_dataset)
        brain_df = self.brain_preparator.load_features().reset_index()
        return brain_df

    def _get_demographics_df(self) -> pd.DataFrame:
        """
        Prepare demographics DataFrame.
        Returns:
            pd.DataFrame: Preprocessed demographics DataFrame.
        """
        # -------- DEMOGRAPHICS --------
        if self.source_dataset.name == "hcp":
            cog_file = self.cfg.cluster.paths[self.source_dataset.name].csv_file

        else:
            cog_file = self._extract_participants_files_from_layouts(
                self.brain_preparator.layouts
            )
        preprocessor = DefaultDemographicsPreprocessor(cog_file)
        demographics_df = preprocessor.preprocess(self.cfg.target.target_column)
        return demographics_df

    def _filter_dfs(
        self, brain_df: pd.DataFrame, demographics_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align and filter brain and demographics DataFrames.
        Args:
            brain_df (pd.DataFrame): DataFrame containing brain features.
            demographics_df (pd.DataFrame): DataFrame containing demographics data.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Aligned and filtered brain and demographics Data
        """
        # -------- SUBJECT ALIGNMENT --------
        brain_df["subject_id"] = brain_df["subject_id"].astype(str)
        demographics_df["Subject"] = demographics_df["Subject"].astype(str)
        common_subjects = set(brain_df["subject_id"]) & set(demographics_df["Subject"])

        brain_filtered = brain_df[brain_df["subject_id"].isin(common_subjects)]
        demographics_filtered = demographics_df[
            demographics_df["Subject"].isin(common_subjects)
        ]
        return brain_filtered, demographics_filtered

    def _create_torch_dataset(
        self, brain_filtered: pd.DataFrame, demographics_filtered: pd.DataFrame
    ) -> Tuple[CustomDataset, PreprocessedData]:
        """
        Create CustomDataset and PreprocessedData objects.
        Args:
            brain_filtered (pd.DataFrame): Filtered brain features DataFrame.
            demographics_filtered (pd.DataFrame): Filtered demographics DataFrame.
        Returns:
            Tuple[CustomDataset, PreprocessedData]: Created dataset and preprocessed data.
        """
        # -------- DATASET CREATION --------
        X = brain_filtered
        y = np.asarray(demographics_filtered[self.cfg.target.target_column[0]])
        gender = np.asarray(demographics_filtered["Gender"])
        torch_dataset = CustomDataset(X, y, gender)
        preprocessed = PreprocessedData(X, y, gender, config=self.cfg)
        return torch_dataset, preprocessed

    def pipeline(self) -> Tuple[CustomDataset, PreprocessedData]:
        """
        Orchestrates the data preparation pipeline.
        Returns:
            Tuple[CustomDataset, PreprocessedData]: The prepared dataset and preprocessed data.
        """
        brain_df = self._get_brain_df()
        demographics_df = self._get_demographics_df()
        brain_filtered, demographics_filtered = self._filter_dfs(
            brain_df, demographics_df
        )
        torch_dataset, preprocessed = self._create_torch_dataset(
            brain_filtered, demographics_filtered
        )
        return torch_dataset, preprocessed
