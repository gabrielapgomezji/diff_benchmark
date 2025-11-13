from pathlib import Path

import yaml


def load_configs(args):
    # =====================
    # Load general config
    # =====================
    general_config_path = (
        Path(__file__).parent.parent.parent.parent / "config/configuration_general.yaml"
    )
    with open(general_config_path, "r") as f:
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

        with open(model_config_path, "r") as f:
            tmp_config = yaml.safe_load(f)

        # merge the model(s) from this file
        if "models" in tmp_config:
            model_config["models"].extend(tmp_config["models"])

        # add any other top-level keys (like pretrain_path, etc.)
        for key, value in tmp_config.items():
            if key != "models":
                model_config[key] = value

    return general_config, model_config
