import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf


@pytest.fixture
def synthetic_brain_csv(tmp_path):
    """Fixture to generate a brain features CSV with subject_id + 1000 random columns."""
    np.random.seed(42)
    subjects = np.random.choice(
        [f"{i:04d}" for i in range(1001, 1021)], size=10, replace=False
    ).tolist()
    n_features = 1000

    data = np.random.randn(len(subjects), n_features)  # random Gaussian features
    df = pd.DataFrame(data, columns=[f"{i}" for i in range(n_features)])
    df.insert(
        0, "subject_id", np.random.choice(subjects, size=len(subjects), replace=False)
    )

    csv_path = tmp_path / "synthetic_brain.csv"
    df.to_csv(csv_path, index=False)

    return csv_path  # returns path to CSV for tests


@pytest.fixture
def synthetic_demographics_csv(tmp_path):
    """Fixture to generate demographics CSV with Subject + behavioral columns."""
    subjects = [f"{i:04d}" for i in range(1001, 1021)]

    df = pd.DataFrame(
        {
            "Subject": subjects,
            "Age_in_Yrs": np.random.randint(
                18, 70, size=len(subjects)
            ),  # random ages 18-70
            "Gender": np.random.choice(["M", "F"], size=len(subjects)),  # random gender
            "IQ": np.random.normal(100, 15, size=len(subjects)),  # IQ normal dist
            "ReactionTime": np.random.rand(len(subjects)) * 2.0,  # between 0 and 2s
            "PicVocab_Unadj": np.random.randint(0, 10, size=len(subjects)),
            "ReadEng_Unadj": np.random.randint(0, 10, size=len(subjects)),
            "DDisc_AUC_40K": np.random.randint(18, 70, size=len(subjects)),
            "Endurance_Unadj": np.random.randint(18, 70, size=len(subjects)),
            "Relational_Task_Acc": np.random.randint(0, 10, size=len(subjects)),
            "VSPLOT_TC": np.random.randint(18, 70, size=len(subjects)),
            "LifeSatisf_Unadj": np.random.randint(0, 10, size=len(subjects)),
            "PMAT24_A_CR": np.random.randint(18, 70, size=len(subjects)),
            "Taste_Unadj": np.random.randint(0, 10, size=len(subjects)),
            "WM_Task_Acc": np.random.randint(20, 100, size=len(subjects)),
            "CardSort_Unadj": np.random.randint(0, 10, size=len(subjects)),
            "ListSort_Unadj": np.random.randint(0, 10, size=len(subjects)),
            "Language_Task_Story_Avg_Difficulty_Level": np.random.randint(
                18, 70, size=len(subjects)
            ),
            "PercStress_Unadj": np.random.randint(20, 100, size=len(subjects)),
        }
    )

    csv_path = tmp_path / "synthetic_demographics.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def sample_all_results(tmp_path: Path):
    """Fixture that creates a fake all_results.json file with multiple models."""
    all_results = [
        {
            "model_name": "2dcnn",
            "pipeline": {
                "batch_size": 128,
                "n_splits": 5,
                "random_state": 42,
                "run_id": "2dcnn_1f5f8fac",
            },
            "results": {
                "train_average_score": 0.85,
                "test_average_score": 0.80,
            },
            "history": {"loss": [0.9, 0.6, 0.3]},
        },
        {
            "model_name": "mlp",
            "pipeline": {
                "batch_size": 64,
                "n_splits": 5,
                "random_state": 42,
                "run_id": "mlp_a1b2c3d4",
            },
            "results": {
                "train_average_score": 0.90,
                "test_average_score": 0.82,
            },
            "history": {"loss": [1.0, 0.7, 0.4]},
        },
    ]

    results_file = tmp_path / "all_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f)

    return tmp_path  # return directory containing the file

@pytest.fixture
def cached_experiments_root(tmp_path: Path) -> Path:
    """Create fake experiment directories for cache tests."""

    successful_runs = [
        "2dcnn_1f5f8fac",
        "mlp_a1b2c3d4",
    ]

    for run_id in successful_runs:
        experiment_dir = tmp_path / f"exp_{run_id}"
        metrics_dir = experiment_dir / "metrics"

        metrics_dir.mkdir(parents=True)

        OmegaConf.save(
            {"status": "success"},
            experiment_dir / "metadata.yaml",
        )

        # is_cached only checks that this file exists, so it does not
        # need to contain a valid Parquet table for this unit test.
        (metrics_dir / "summary.parquet").touch()

    return tmp_path