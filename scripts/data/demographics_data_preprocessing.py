from pathlib import Path

import yaml

from diff_benchmark.preprocessing.demographics_data import preprocess_csv

# ---------- IMPORT CONFIG ----------
with open(Path(__file__).parent.parent.parent / "configuration.yaml", "r") as f:
    config = yaml.safe_load(f)

# ---------- FILTER DEMOGRAPHICS ----------
df_filtered = preprocess_csv(config["csv_file"], config["target_columns"])
