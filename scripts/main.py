# ---------- RUN PREPROCESSING FOR RAW DATA ----------

import yaml
from pathlib import Path
from os.path import expanduser
from joblib import Parallel, delayed
from diff_benchmark.raw_data.process_raw_data import DWIProcessor

def run_parallel_processing(processor_cls, config: dict, n_jobs: int = 10):
    base_folder = Path(expanduser(config["base_path"]))
    subjects = [p.name for p in base_folder.iterdir() if p.is_dir()]
    processor = processor_cls(config)

    Parallel(n_jobs=n_jobs)(
        delayed(processor.run)(sub) for sub in subjects
    )

config_path = Path(__file__).parent.parent.parent / "configuration.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

run_parallel_processing(DWIProcessor, config, n_jobs=10)


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