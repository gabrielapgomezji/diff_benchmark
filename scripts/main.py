# ---------- RUN PREPROCESSING FOR RAW DATA ----------

import yaml
from pathlib import Path
from diff_benchmark.raw_data.process_raw_data import DWIProcessor


config_path = Path(__file__).parent.parent.parent / "configuration.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

processor = DWIProcessor(config)
processor.run_parallel()


# ---------- RUN PREPROCESSING FOR INPUT DATA ----------
from diff_benchmark.preprocessing.preprocess_brain_data import DefaultBrainPreprocessor

# Create preprocessor instance
preprocessor = DefaultBrainPreprocessor(config)

# Run subject preprocessing
preprocessor.preprocess_subject(raw_data_path, sub, save_path)

# ---------- RUN PREPROCESSING FOR TARGET DATA ----------
from diff_benchmark.preprocessing.preprocess_demographic_data import DefaultDemographicsPreprocessor

preprocessor = DefaultDemographicsPreprocessor(config["csv_file"])
df_clean = preprocessor.preprocess(config["target_columns"])

# ----------- SAVE PROCESSED DATA ----------

# ----------- CROSS VALIDATION + TRAINING + TESTING -----------

# ------------ EVALUATION AND ANALYSIS ------------