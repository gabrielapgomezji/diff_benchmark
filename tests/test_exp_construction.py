"""
Tests to evaluate a correct experiment construction.
"""
from omegaconf import DictConfig, OmegaConf

from datetime import datetime
from unittest.mock import Mock
import pytest

import diff_benchmark.cli.utils as config_utils
import diff_benchmark.utils.run_id as run_id_utils
from diff_benchmark.utils.run_id import is_cached

# ================================================
# ---------- TESTS FOR cartesian_cfgs() ----------
# ================================================
def test_cartesian_cfgs_returns_single_config_when_no_axes(monkeypatch):
    """A configuration without sweep axes should produce one equivalent config."""
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "hcp"},
            "model": {"name": "ridge"},
        }
    )

    base = OmegaConf.to_container(cfg, resolve=True)

    monkeypatch.setattr(
        config_utils,
        "split_base_and_axes",
        lambda _: (base, {}),
    )

    result = config_utils.cartesian_cfgs(cfg)

    assert len(result) == 1
    assert isinstance(result[0], DictConfig)
    assert OmegaConf.to_container(result[0], resolve=True) == base

    # The function should return a newly created config, not the original object.
    assert result[0] is not cfg


def test_cartesian_cfgs_expands_single_axis(monkeypatch):
    """Every value of one sweep axis should produce one configuration."""
    cfg = OmegaConf.create({})

    base = {
        "dataset": {"name": "hcp"},
        "model": {"name": None},
    }
    axes = {
        "model.name": ["ridge", "random_forest", "mlp"],
    }

    monkeypatch.setattr(
        config_utils,
        "split_base_and_axes",
        lambda _: (base, axes),
    )

    result = config_utils.cartesian_cfgs(cfg)

    assert len(result) == 3
    assert [config.model.name for config in result] == [
        "ridge",
        "random_forest",
        "mlp",
    ]

    assert all(config.dataset.name == "hcp" for config in result)


def test_cartesian_cfgs_builds_full_cartesian_product(monkeypatch):
    """Several axes should produce every possible combination."""
    cfg = OmegaConf.create({})

    base = {
        "dataset": {"name": None},
        "model": {"name": None},
        "random_state": 42,
    }
    axes = {
        "dataset.name": ["hcp", "camcan"],
        "model.name": ["ridge", "mlp"],
    }

    monkeypatch.setattr(
        config_utils,
        "split_base_and_axes",
        lambda _: (base, axes),
    )

    result = config_utils.cartesian_cfgs(cfg)

    combinations = [
        (config.dataset.name, config.model.name)
        for config in result
    ]

    assert combinations == [
        ("hcp", "ridge"),
        ("hcp", "mlp"),
        ("camcan", "ridge"),
        ("camcan", "mlp"),
    ]

    assert len(result) == 4
    assert all(config.random_state == 42 for config in result)


def test_cartesian_cfgs_supports_nested_dotpaths(monkeypatch):
    """Sweep values should be inserted at the correct nested location."""
    cfg = OmegaConf.create({})

    base = {
        "data": {
            "data_partition": {
                "n_splits": None,
            }
        },
        "model": {
            "backbone": {
                "hidden_size": None,
            }
        },
    }
    axes = {
        "data.data_partition.n_splits": [2, 5],
        "model.backbone.hidden_size": [32, 64],
    }

    monkeypatch.setattr(
        config_utils,
        "split_base_and_axes",
        lambda _: (base, axes),
    )

    result = config_utils.cartesian_cfgs(cfg)

    values = [
        (
            config.data.data_partition.n_splits,
            config.model.backbone.hidden_size,
        )
        for config in result
    ]

    assert values == [
        (2, 32),
        (2, 64),
        (5, 32),
        (5, 64),
    ]


def test_cartesian_cfgs_does_not_modify_base_config(monkeypatch):
    """Expanding configurations should not mutate the shared base dictionary."""
    cfg = OmegaConf.create({})

    base = {
        "dataset": {"name": "hcp"},
        "model": {
            "name": None,
            "parameters": {"alpha": 1.0},
        },
    }
    expected_base = {
        "dataset": {"name": "hcp"},
        "model": {
            "name": None,
            "parameters": {"alpha": 1.0},
        },
    }
    axes = {
        "model.name": ["ridge", "lasso"],
    }

    monkeypatch.setattr(
        config_utils,
        "split_base_and_axes",
        lambda _: (base, axes),
    )

    config_utils.cartesian_cfgs(cfg)

    assert base == expected_base


def test_cartesian_cfgs_returns_independent_configs(monkeypatch):
    """Modifying one generated configuration must not affect the others."""
    cfg = OmegaConf.create({})

    base = {
        "model": {
            "name": None,
            "parameters": {
                "alpha": 1.0,
            },
        }
    }
    axes = {
        "model.name": ["ridge", "lasso"],
    }

    monkeypatch.setattr(
        config_utils,
        "split_base_and_axes",
        lambda _: (base, axes),
    )

    result = config_utils.cartesian_cfgs(cfg)

    result[0].model.parameters.alpha = 100.0

    assert result[0].model.parameters.alpha == 100.0
    assert result[1].model.parameters.alpha == 1.0


def test_cartesian_cfgs_returns_empty_list_for_empty_axis(monkeypatch):
    """An axis without possible values currently produces no configurations."""
    cfg = OmegaConf.create({})

    base = {
        "dataset": {"name": None},
    }
    axes = {
        "dataset.name": [],
    }

    monkeypatch.setattr(
        config_utils,
        "split_base_and_axes",
        lambda _: (base, axes),
    )

    result = config_utils.cartesian_cfgs(cfg)

    assert result == []

# ========================================
# ---------- TESTS FOR run_id() ----------
# ========================================
def test_make_run_id_combines_prefix_and_short_hash(monkeypatch):
    """The run ID should contain the readable prefix and first eight hash characters."""
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "hcp"},
            "model": {"name": "ridge"},
        }
    )

    fingerprint_mock = Mock(
        return_value="12345678abcdef1234567890"
    )
    prefix_mock = Mock(
        return_value="hcp_ridge"
    )

    monkeypatch.setattr(
        run_id_utils,
        "config_fingerprint",
        fingerprint_mock,
    )
    monkeypatch.setattr(
        run_id_utils,
        "_build_readable_prefix",
        prefix_mock,
    )

    run_id, experiment_hash = run_id_utils.make_run_id(cfg)

    assert run_id == "hcp_ridge_12345678"
    assert experiment_hash == "12345678"

    fingerprint_mock.assert_called_once_with(cfg)
    prefix_mock.assert_called_once_with(cfg)


def test_make_run_id_is_deterministic_without_force(monkeypatch):
    """Repeated calls should return the same ID when force is False."""
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "hcp"},
            "model": {"name": "ridge"},
        }
    )

    monkeypatch.setattr(
        run_id_utils,
        "config_fingerprint",
        lambda _: "abcdef1234567890",
    )
    monkeypatch.setattr(
        run_id_utils,
        "_build_readable_prefix",
        lambda _: "hcp_ridge",
    )

    first_run_id, first_hash = run_id_utils.make_run_id(
        cfg,
        force=False,
    )
    second_run_id, second_hash = run_id_utils.make_run_id(
        cfg,
        force=False,
    )

    assert first_run_id == second_run_id
    assert first_hash == second_hash
    assert first_run_id == "hcp_ridge_abcdef12"
    assert first_hash == "abcdef12"


def test_make_run_id_appends_timestamp_when_forced(monkeypatch):
    """force=True should append the current timestamp."""
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "hcp"},
            "model": {"name": "ridge"},
        }
    )

    monkeypatch.setattr(
        run_id_utils,
        "config_fingerprint",
        lambda _: "12345678abcdef",
    )
    monkeypatch.setattr(
        run_id_utils,
        "_build_readable_prefix",
        lambda _: "hcp_ridge",
    )

    fixed_datetime = datetime(
        year=2026,
        month=7,
        day=23,
        hour=14,
        minute=35,
        second=12,
    )

    datetime_mock = Mock()
    datetime_mock.now.return_value = fixed_datetime

    monkeypatch.setattr(
        run_id_utils,
        "datetime",
        datetime_mock,
    )

    run_id, experiment_hash = run_id_utils.make_run_id(
        cfg,
        force=True,
    )

    assert run_id == "hcp_ridge_12345678_20260723-143512"
    assert experiment_hash == "12345678"

    datetime_mock.now.assert_called_once_with()


def test_make_run_id_does_not_add_timestamp_by_default(monkeypatch):
    """The default behavior should not read the clock or append a timestamp."""
    cfg = OmegaConf.create({})

    monkeypatch.setattr(
        run_id_utils,
        "config_fingerprint",
        lambda _: "deadbeef12345678",
    )
    monkeypatch.setattr(
        run_id_utils,
        "_build_readable_prefix",
        lambda _: "test_model",
    )

    datetime_mock = Mock()
    monkeypatch.setattr(
        run_id_utils,
        "datetime",
        datetime_mock,
    )

    run_id, experiment_hash = run_id_utils.make_run_id(cfg)

    assert run_id == "test_model_deadbeef"
    assert experiment_hash == "deadbeef"

    datetime_mock.now.assert_not_called()


def test_make_run_id_passes_original_config_to_helpers(monkeypatch):
    """Both helper functions should receive the original configuration."""
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "camcan"},
            "model": {"name": "random_forest"},
        }
    )

    fingerprint_mock = Mock(return_value="aaaaaaaa99999999")
    prefix_mock = Mock(return_value="camcan_random_forest")

    monkeypatch.setattr(
        run_id_utils,
        "config_fingerprint",
        fingerprint_mock,
    )
    monkeypatch.setattr(
        run_id_utils,
        "_build_readable_prefix",
        prefix_mock,
    )

    run_id_utils.make_run_id(cfg)

    fingerprint_mock.assert_called_once_with(cfg)
    prefix_mock.assert_called_once_with(cfg)
    

# ===========================================
# ---------- TESTS FOR is_cached() ----------
# ===========================================
@pytest.mark.parametrize(
    "run_id",
    [
        "2dcnn_1f5f8fac",
        "mlp_a1b2c3d4",
    ],
)
def test_is_cached_returns_true_for_successful_experiment(
    cached_experiments_root,
    run_id,
):
    assert is_cached(run_id, cached_experiments_root)


def test_is_cached_returns_false_for_unknown_run(
    cached_experiments_root,
):
    assert not is_cached(
        "nonexistent_run_id",
        cached_experiments_root,
    )


def test_is_cached_returns_false_when_experiment_directory_is_missing(
    tmp_path,
):
    assert not is_cached("any_run_id", tmp_path)


def test_is_cached_returns_false_when_metadata_is_missing(tmp_path):
    run_id = "ridge_12345678"
    metrics_dir = (
        tmp_path
        / f"exp_{run_id}"
        / "metrics"
    )
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "summary.parquet").touch()

    assert not is_cached(run_id, tmp_path)


def test_is_cached_returns_false_when_metrics_are_missing(tmp_path):
    run_id = "ridge_12345678"
    experiment_dir = tmp_path / f"exp_{run_id}"
    experiment_dir.mkdir(parents=True)

    OmegaConf.save(
        {"status": "success"},
        experiment_dir / "metadata.yaml",
    )

    assert not is_cached(run_id, tmp_path)


@pytest.mark.parametrize(
    "status",
    [
        "running",
        "partial",
        "crashed",
        "failed",
        None,
    ],
)
def test_is_cached_returns_false_for_non_success_status(
    tmp_path,
    status,
):
    run_id = "ridge_12345678"
    experiment_dir = tmp_path / f"exp_{run_id}"
    metrics_dir = experiment_dir / "metrics"

    metrics_dir.mkdir(parents=True)

    metadata = {}
    if status is not None:
        metadata["status"] = status

    OmegaConf.save(
        metadata,
        experiment_dir / "metadata.yaml",
    )
    (metrics_dir / "summary.parquet").touch()

    assert not is_cached(run_id, tmp_path)


def test_is_cached_requires_expected_summary_filename(tmp_path):
    """A different Parquet filename should not count as a complete cache."""
    run_id = "ridge_12345678"
    experiment_dir = tmp_path / f"exp_{run_id}"
    metrics_dir = experiment_dir / "metrics"

    metrics_dir.mkdir(parents=True)

    OmegaConf.save(
        {"status": "success"},
        experiment_dir / "metadata.yaml",
    )

    # Wrong filename.
    (metrics_dir / "fold_metrics.parquet").touch()

    assert not is_cached(run_id, tmp_path)