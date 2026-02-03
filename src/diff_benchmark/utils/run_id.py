from datetime import datetime
import json
import hashlib
from omegaconf import OmegaConf
from pathlib import Path


TARGET_ABBR = {
    "gender": "g",
    "sex": "g",
    "age": "a",
    "diagnosis": "d",
}

MICROSTRUCTURE_ABBR = {
    "md": "md",
    "mk": "mk",
    "rtop": "ro",
    "rtap": "ra",
    "rtpp": "rp",
}

DATASET_ABBR = {
    "hcp": "h",
    "camcan": "c",
    "abide2": "a",
    "wand": "w",
}

EXCLUDE_KEYS = {
    "runtime",
    "hydra",
    "cluster",
    "slurm",
    "paths",
    "choices"
}

def config_fingerprint(cfg) -> str:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    def strip_keys(d):
        if isinstance(d, dict):
            return {
                k: strip_keys(v)
                for k, v in d.items()
                if k not in EXCLUDE_KEYS
            }
        elif isinstance(d, list):
            return [strip_keys(x) for x in d]
        return d

    clean_cfg = strip_keys(cfg_dict)
    serialized = json.dumps(clean_cfg, sort_keys=True)
    return hashlib.sha1(serialized.encode()).hexdigest()

def abbr(value: str, table: dict, fallback_len=3):
    """
    Convert a name to a short, collision-safe abbreviation.
    """
    key = value.lower() 
    if key in table:
        return table[key]
    return value.lower()[:fallback_len]

def build_readable_prefix(cfg):
    model = cfg.model.name.lower()

    dataset = abbr(cfg.dataset.name, DATASET_ABBR)
    micro = abbr(cfg.dataset.metric_to_compute, MICROSTRUCTURE_ABBR)
    target = abbr(cfg.target.target_column[0], TARGET_ABBR)

    return f"{model}_{dataset}{micro}{target}"

def make_run_id(cfg, force=False):
    """
    Returns:
      run_id: str
      experiment_hash: str
    """
    exp_hash = config_fingerprint(cfg)[:8]
    prefix = build_readable_prefix(cfg)

    base_id = f"{prefix}_{exp_hash}"

    if force:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{base_id}_{ts}", exp_hash

    return base_id, exp_hash


def is_cached(run_id: str, experiments_root: Path) -> bool:
    exp_dir = experiments_root / f"exp_{run_id}"

    meta = exp_dir / "metadata.yaml"
    metrics = exp_dir / "metrics" / "summary.parquet"

    if not meta.exists() or not metrics.exists():
        return False

    metadata = OmegaConf.load(meta)
    return metadata.get("status") == "success"
