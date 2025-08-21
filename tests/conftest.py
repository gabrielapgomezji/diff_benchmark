import numpy as np
import pandas as pd
import pytest


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
