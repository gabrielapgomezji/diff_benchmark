import argparse
from pathlib import Path

import yaml


def load_configs(args: argparse.Namespace) -> tuple[dict, dict]:
    """Load general and model-specific configurations based on provided arguments.
    Args:
        args (argparse.Namespace): Parsed command-line arguments containing 'methods'.
    Returns:
        tuple[dict, dict]: A tuple containing the general configuration and the combined model configuration.
    """
    # =====================
    # Load general config
    # =====================
    general_config_path = (
        Path(__file__).parent.parent.parent.parent
        / f"config/configuration_general_{getattr(args, "cluster")}.yaml"
    )
    if not general_config_path.exists():
        general_config_path = (
            Path(__file__).parent.parent.parent.parent
            / "config/configuration_general.yaml"
        )
    with open(general_config_path, "r", encoding="utf-8") as f:
        general_config = yaml.safe_load(f)

    # =====================
    # Load model configs
    # =====================
    model_config = {"models": []}
    for method in args.methods:  # list of model names
        model_config_path = (
            Path(__file__).parent.parent.parent.parent
            / f"config/configuration_{method}.yaml"
        )
        if not model_config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found for method '{method}' at {model_config_path}"
            )

        with open(model_config_path, "r", encoding="utf-8") as f:
            tmp_config = yaml.safe_load(f)

        # merge the model(s) from this file
        if "models" in tmp_config:
            model_config["models"].extend(tmp_config["models"])

        # add any other top-level keys (like pretrain_path, etc.)
        for key, value in tmp_config.items():
            if key != "models":
                model_config[key] = value

    return general_config, model_config
