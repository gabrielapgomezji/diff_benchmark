import hashlib
import json
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

# Keys excluded from the config fingerprint (runtime / cluster / infrastructure).
_EXCLUDE_KEYS = {"runtime", "hydra", "cluster", "slurm", "paths", "choices", "analysis"}

_TARGET_ABBR: dict[str, str] = {
    "gender": "g",
    "sex": "g",
    "age": "a",
    "diagnosis": "d",
}

_MICROSTRUCTURE_ABBR: dict[str, str] = {
    "md": "md",
    "mk": "mk",
    "rtop": "ro",
    "rtap": "ra",
    "rtpp": "rp",
    "sh": "sh",
}

_TISSUE_TYPE_ABBR: dict[str, str] = {
    "white_matter": "w",
    "gray_matter": "g",
}

_DATASET_ABBR: dict[str, str] = {
    "hcp": "h",
    "camcan": "c",
    "abide2": "a",
    "wand": "w",
}


def _strip_keys(d: object, exclude: frozenset) -> object:
    """Recursively remove *exclude* keys from a nested dict/list structure."""
    if isinstance(d, dict):
        return {k: _strip_keys(v, exclude) for k, v in d.items() if k not in exclude}
    if isinstance(d, list):
        return [_strip_keys(x, exclude) for x in d]
    return d


def config_fingerprint(cfg, exclude_extra: set | None = None) -> str:
    """Compute a SHA-1 hash of the experiment config, ignoring infrastructure keys.

    Args:
        cfg: OmegaConf config object.
        exclude_extra: Additional top-level keys to exclude on top of the defaults.

    Returns:
        40-character hex SHA-1 digest.
    """
    exclude = _EXCLUDE_KEYS
    if exclude_extra:
        exclude = _EXCLUDE_KEYS | frozenset(exclude_extra)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    clean = _strip_keys(cfg_dict, frozenset(exclude))
    serialized = json.dumps(clean, sort_keys=True)
    return hashlib.sha1(serialized.encode()).hexdigest()


def get_learning_curve_id(cfg) -> str:
    """Compute a run-ID that is identical for all points on the same learning curve.

    Behaves like :func:`config_fingerprint` but also excludes ``train_size`` so
    that points with different training-set sizes share the same curve ID.

    Args:
        cfg: OmegaConf config object.

    Returns:
        40-character hex SHA-1 digest.
    """
    return config_fingerprint(cfg, exclude_extra={"train_size"})


def _abbr(value: str, table: dict, fallback_len: int = 3) -> str:
    """Convert *value* to a short abbreviation using *table*, with a truncation fallback.

    Args:
        value: The string to abbreviate.
        table: Lookup dict mapping lowercase strings to abbreviations.
        fallback_len: Maximum character length when *value* is not in *table*.

    Returns:
        Abbreviated string.
    """
    key = value.lower()
    return table.get(key, key[:fallback_len])


def _build_readable_prefix(cfg) -> str:
    """Build a human-readable run-ID prefix from config fields.

    Args:
        cfg: OmegaConf config object.

    Returns:
        Underscore-separated string such as ``"2dcnn_hwhmd_g"``.
    """
    model = cfg.model.name.lower()
    dataset = _abbr(cfg.dataset.name, _DATASET_ABBR)
    tissue = _abbr(cfg.dataset.tissue_type, _TISSUE_TYPE_ABBR)
    micro = _abbr(cfg.dataset.metric_to_compute, _MICROSTRUCTURE_ABBR)
    target = _abbr(cfg.target.target_column[0], _TARGET_ABBR)
    return f"{model}_{dataset}{tissue}{micro}{target}"


def make_run_id(cfg, force: bool = False) -> tuple[str, str]:
    """Construct a run ID and experiment hash from an OmegaConf config.

    Args:
        cfg: OmegaConf config object.
        force: When ``True``, appends a UTC timestamp to guarantee uniqueness.

    Returns:
        Tuple of ``(run_id, experiment_hash)`` where *experiment_hash* is the
        first 8 characters of the SHA-1 fingerprint.
    """
    exp_hash = config_fingerprint(cfg)[:8]
    prefix = _build_readable_prefix(cfg)
    base_id = f"{prefix}_{exp_hash}"

    if force:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{base_id}_{ts}", exp_hash

    return base_id, exp_hash


def is_cached(run_id: str, experiments_root: Path) -> bool:
    """Return ``True`` if *run_id* has a completed, successful experiment on disk.

    An experiment is considered cached when its ``metadata.yaml`` reports
    ``status: success`` **and** its ``metrics/summary.parquet`` file exists.

    Args:
        run_id: Unique run identifier.
        experiments_root: Root directory containing ``exp_<run_id>`` subdirectories.

    Returns:
        ``True`` if the experiment is fully cached, ``False`` otherwise.
    """
    exp_dir = experiments_root / f"exp_{run_id}"
    meta = exp_dir / "metadata.yaml"
    metrics = exp_dir / "metrics" / "summary.parquet"

    if not meta.exists() or not metrics.exists():
        return False

    metadata = OmegaConf.load(meta)
    return metadata.get("status") == "success"
