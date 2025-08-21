import pandas as pd


def test_X_y_alignment(synthetic_brain_csv, synthetic_demographics_csv):
    # Load both CSVs
    brain_df = pd.read_csv(synthetic_brain_csv)
    demo_df = pd.read_csv(synthetic_demographics_csv)

    # Ensure subject columns exist
    assert "subject_id" in brain_df.columns
    assert "Subject" in demo_df.columns

    # --- Alignment step ---
    common_subjects = set(brain_df["subject_id"].astype(str)) & set(
        demo_df["Subject"].astype(str)
    )
    assert (
        len(common_subjects) > 0
    ), "No overlapping subjects between brain and demographics"

    brain_filtered = brain_df[
        brain_df["subject_id"].astype(str).isin(common_subjects)
    ].copy()
    demo_filtered = demo_df[demo_df["Subject"].astype(str).isin(common_subjects)].copy()

    # Align order by sorting
    brain_filtered = brain_filtered.sort_values("subject_id").reset_index(drop=True)
    demo_filtered = demo_filtered.sort_values("Subject").reset_index(drop=True)

    # --- Tests ---
    # 1. Same length
    assert len(brain_filtered) == len(demo_filtered), "X and y lengths mismatch"

    # 2. Same subjects in same order
    assert list(brain_filtered["subject_id"].astype(str)) == list(
        demo_filtered["Subject"].astype(str)
    ), "Subjects not aligned between X and y"

    # 3. Brain features exclude ID column
    X = brain_filtered.drop(columns=["subject_id"])
    y = demo_filtered[["Age_in_Yrs"]]  # for example, target = Age

    assert len(X) == len(y), "Final X and y lengths mismatch"
