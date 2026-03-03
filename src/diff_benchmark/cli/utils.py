from copy import deepcopy
from itertools import product

from omegaconf import DictConfig, OmegaConf

CHOICE_KEY = "_choices_"


def split_base_and_axes(obj) -> tuple[dict, dict]:
    """Separate a config into a base dict and a set of sweep axes.

    Nodes that carry a ``_choices_`` list are treated as sweep dimensions.
    The returned *base* replaces each such node with its first value (the
    default), and *axes* maps the dotted path to the full list of values.

    Args:
        obj: OmegaConf config object or plain dict.

    Returns:
        Tuple of ``(base, axes)`` where *base* is a plain Python dict and
        *axes* is a ``{"a.b.c": [v1, v2, ...]}`` mapping.
    """
    axes: dict = {}
    raw = deepcopy(OmegaConf.to_container(obj, resolve=False))

    def _recurse(node, path=()):
        if isinstance(node, dict):
            if CHOICE_KEY in node and isinstance(node[CHOICE_KEY], list):
                axes[".".join(path)] = node[CHOICE_KEY]
                # Substitute first value as the default.
                return node[CHOICE_KEY][0]
            return {k: _recurse(v, path + (k,)) for k, v in node.items()}
        if isinstance(node, list):
            return node  # Lists are treated as data, not sweep axes.
        return node

    base = _recurse(raw)
    return base, axes


def _set_by_dotpath(d: dict, dotpath: str, value) -> None:
    """Set a nested dict value using a dot-separated key path.

    Args:
        d: Target (mutable) dictionary.
        dotpath: Key path such as ``"model.backbone.lr"``.
        value: Value to assign.
    """
    keys = dotpath.split(".")
    node = d
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def cartesian_cfgs(cfg: DictConfig) -> list[DictConfig]:
    """Expand sweep axes in *cfg* into a cartesian product of single-value configs.

    Uses :func:`split_base_and_axes` to find ``_choices_`` sweep nodes and
    returns one config per combination.

    Args:
        cfg: OmegaConf config possibly containing ``_choices_`` sweep nodes.

    Returns:
        List of :class:`DictConfig` objects, one per combination.
        Returns ``[cfg]`` when no sweep axes are found.
    """
    base, axes = split_base_and_axes(cfg)

    if not axes:
        return [OmegaConf.create(base)]

    axis_paths = list(axes.keys())
    axis_vals = [axes[p] for p in axis_paths]

    configs = []
    for combo in product(*axis_vals):
        instance = deepcopy(base)
        for path, value in zip(axis_paths, combo):
            _set_by_dotpath(instance, path, value)
        configs.append(OmegaConf.create(instance))

    return configs


def build_config_grid(cfg: DictConfig) -> list[DictConfig]:
    """Expand list-valued fields in *cfg* into a cartesian product of configs.

    Unlike :func:`cartesian_cfgs` (which uses ``_choices_`` markers), this
    function treats *any* list-valued leaf as a sweep axis.

    Args:
        cfg: OmegaConf config object where list-valued leaves define sweep ranges.

    Returns:
        List of :class:`DictConfig` objects, one per combination.
        Returns ``[cfg]`` when no list-valued leaves are found.
    """

    def _find_list_leaves(node: DictConfig, prefix: str = "") -> dict:
        leaves = {}
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, DictConfig):
                leaves.update(_find_list_leaves(v, key))
            elif isinstance(v, list):
                leaves[key] = v
        return leaves

    grid_params = _find_list_leaves(cfg)

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
