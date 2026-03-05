import argparse
from pathlib import Path

import yaml

_CONFIG_ROOT = Path(__file__).parent.parent.parent.parent / "config"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs(args: argparse.Namespace) -> tuple[dict, dict]:
    """Load general and model-specific configurations from YAML files.

    Args:
        args: Parsed CLI arguments.  Must expose ``args.methods`` (list of model
            names) and optionally ``args.cluster`` (cluster name suffix).

    Returns:
        Tuple of ``(general_config, model_config)`` where ``model_config``
        contains a ``"models"`` key with all requested model definitions merged.

    Raises:
        FileNotFoundError: If a per-model config file does not exist.
    """
    # Resolve general config, falling back to the cluster-agnostic version.
    cluster = getattr(args, "cluster", None)
    general_config_path = (
        _CONFIG_ROOT / f"configuration_general_{cluster}.yaml" if cluster else None
    )
    if general_config_path is None or not general_config_path.exists():
        general_config_path = _CONFIG_ROOT / "configuration_general.yaml"

    general_config = _load_yaml(general_config_path)

    # Merge configs for each requested model.
    model_config: dict = {"models": []}
    for method in args.methods:
        method_path = _CONFIG_ROOT / f"configuration_{method}.yaml"
        if not method_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found for method '{method}' at {method_path}"
            )

        tmp = _load_yaml(method_path)
        model_config["models"].extend(tmp.pop("models", []))
        model_config.update(tmp)

    return general_config, model_config
