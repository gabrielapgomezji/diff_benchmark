from pathlib import Path

import yaml

from diff_benchmark.dataset.generate_dataset import build_dataset

with open(Path(__file__).parent.parent.parent / "configuration.yml", "r") as f:
    config = yaml.safe_load(f)

df_filtered = preprocess_csv(config["csv_path"], config["target_columns"])

dataset = build_dataset(
    config["results_path"],
    df_filtered,
    h5_filename="mapmri_default_embeddings.h5",  # Model data. (This file is for the computed embeddings)
    output_dataset_filename=Path(config["results_path"]) / "datasets" / "dataset.h5",
)
