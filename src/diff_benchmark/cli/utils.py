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



CHOICE_KEY = "_choices_"


def split_base_and_axes(obj):
    """
    Returns (base, axes)
    - base: same structure but sweep nodes replaced by a single default (first value)
    - axes: dict mapping "dot.path" -> list_of_values
    """
    axes = {}
    base = deepcopy(OmegaConf.to_container(obj, resolve=False))

    def rec(node, path=()):
        if isinstance(node, dict):
            if CHOICE_KEY in node and isinstance(node[CHOICE_KEY], list):
                p = ".".join(path)
                axes[p] = node[CHOICE_KEY]
                # replace in base with first value as a default
                return node[CHOICE_KEY][0]
            # normal dict: recurse
            out = {}
            for k, v in node.items():
                out[k] = rec(v, path + (k,))
            return out
        elif isinstance(node, list):
            # lists are treated as data unless explicitly marked
            return node
        else:
            return node

    base = rec(base, ())
    return base, axes

def set_by_dotpath(d, dotpath, value):
    keys = dotpath.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value

def cartesian_cfgs(cfg: DictConfig):
    base, axes = split_base_and_axes(cfg)

    axis_paths = list(axes.keys())
    axis_vals = [axes[p] for p in axis_paths]

    cfgs = []
    for combo in product(*axis_vals):
        inst = deepcopy(base)
        for p, v in zip(axis_paths, combo):
            set_by_dotpath(inst, p, v)
        cfgs.append(OmegaConf.create(inst))

    return cfgs




def build_config_grid(cfg: DictConfig):
    """
    Expand list-valued fields into a list of configs,
    where each config has a single value per hyperparameter.
    """
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