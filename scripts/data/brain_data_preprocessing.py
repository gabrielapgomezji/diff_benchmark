import sys
import time
from pathlib import Path
import yaml

from diff_benchmark.preprocessing.brain_data import preprocess_subject

start_time = time.time()
print(f"Script started at: {time.ctime(start_time)}")

# ---------- CONFIG ----------
if len(sys.argv) < 2:
    raise ValueError("Please provide a subject ID as the first argument.")
sub = sys.argv[1]

with open(Path(__file__).parent.parent.parent / "configuration.yml", "r") as f:
    config = yaml.safe_load(f)
        
raw_data_path = (
    Path(config["results_path"])
    / str(sub)
    / "raw_surface_data.h5"
)
save_path = (
    Path(config["results_path"])
    / str(sub)
    / "processed"
)

# ---------- RUN PREPROCESSING ----------
preprocess_subject(raw_data_path=raw_data_path, subject_id=sub, save_path=save_path)

# Record the end time
end_time = time.time()
print(f"Script ended at: {time.ctime(end_time)}")

# Calculate and print the total duration
duration = end_time - start_time
print(f"Total time taken: {duration:.4f} seconds")
