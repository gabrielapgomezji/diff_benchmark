from diff_benchmark.preprocessing.wrapper_brain_data_camcan import (
    DefaultCamcanPipeline,
)
from diff_benchmark.preprocessing.wrapper_brain_data_wand import (
    DefaultWandPipeline,
)
from pathlib import Path
import yaml

general_config_path = (
        Path(__file__).parent.parent / "config/configuration_general.yaml"
    )
with open(general_config_path, "r", encoding="utf-8") as f:
    general_config = yaml.safe_load(f)

# brain_preparator = DefaultCamcanPipeline(general_config)
# subject_id = "CC110037"
brain_preparator = DefaultWandPipeline(general_config)
subject_id = "01187"
brain_preparator.compute_microstructure(subject_id)