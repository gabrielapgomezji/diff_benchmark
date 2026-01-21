from hydra import initialize, compose
from itertools import product
from omegaconf import OmegaConf
from copy import deepcopy
from omegaconf import DictConfig

from omegaconf import DictConfig, OmegaConf
from itertools import product
from copy import deepcopy

from itertools import product
from omegaconf import OmegaConf
from copy import deepcopy

def build_config_grid(cfg: DictConfig):
    """
    Expand list-valued fields into a list of configs,
    where each config has a single value per hyperparameter.
    """
    # 1️⃣ Collect all list-valued leaves
    def find_list_leaves(cfg, prefix=""):
        leaves = {}
        for k, v in cfg.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, DictConfig):
                leaves.update(find_list_leaves(v, key))
            elif isinstance(v, list):
                leaves[key] = v
        return leaves

    grid_params = find_list_leaves(cfg)

    if not grid_params:
        return [cfg]

    keys = list(grid_params.keys())
    values = list(grid_params.values())

    configs = []
    for combo in product(*values):
        cfg_i = deepcopy(cfg)
        for k, v in zip(keys, combo):
            OmegaConf.update(cfg_i, k, v, merge=False)
        configs.append(cfg_i)
    return configs

def find_grid_params(cfg: DictConfig, prefix=""):
    """
    Find all list-valued leaves in a DictConfig.
    Returns { "a.b.c": [v1, v2] }
    """
    grid = {}

    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k

        if isinstance(v, DictConfig):
            grid.update(find_grid_params(v, key))
        elif isinstance(v, list):
            grid[key] = v

    return grid


def build_config_grid2(cfg: DictConfig) -> list[DictConfig]:
    """
    Expands list-valued parameters in cfg into a cartesian product of configs.
    """

    grid_params = find_grid_params(cfg)

    # No grid → single config
    if not grid_params:
        return [cfg]

    keys = list(grid_params.keys())
    values = list(grid_params.values())

    cfgs = []

    for combo in product(*values):
        cfg_i = deepcopy(cfg)

        for k, v in zip(keys, combo):
            OmegaConf.update(cfg_i, k, v, merge=False)

        cfgs.append(cfg_i)

    return cfgs