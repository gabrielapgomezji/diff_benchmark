"""
Test script for data_distribution CLI
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diff_benchmark.cli.data_distribution import (
    compute_variable_statistics,
    generate_summary_report,
    plot_variable_distribution,
)


def test_compute_variable_statistics():
    """Test statistics computation for numeric and categorical variables."""
    print("Testing compute_variable_statistics...")

    # Test numeric variable
    numeric_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name="test_numeric")
    numeric_stats = compute_variable_statistics(numeric_series)

    assert numeric_stats["type"] == "numeric"
    assert numeric_stats["count"] == 10
    assert numeric_stats["mean"] == 5.5
    assert "std" in numeric_stats
    assert "min" in numeric_stats
    assert "max" in numeric_stats
    print("  ✓ Numeric variable statistics computed correctly")

    # Test categorical variable
    categorical_series = pd.Series(
        ["A", "B", "A", "C", "B", "A", "A"], name="test_categorical"
    )
    categorical_stats = compute_variable_statistics(categorical_series)

    assert categorical_stats["type"] == "categorical"
    assert categorical_stats["count"] == 7
    assert categorical_stats["n_unique"] == 3
    assert categorical_stats["top_value"] == "A"
    assert categorical_stats["top_frequency"] == 4
    print("  ✓ Categorical variable statistics computed correctly")

    print("✓ All statistics tests passed\n")


def test_plot_variable_distribution():
    """Test plot generation for numeric and categorical variables."""
    print("Testing plot_variable_distribution...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test numeric plot
        numeric_series = pd.Series(np.random.randn(100), name="test_numeric")
        numeric_plot_path = tmpdir / "numeric_test.png"
        plot_variable_distribution(numeric_series, numeric_plot_path, is_target=False)

        assert numeric_plot_path.exists()
        assert numeric_plot_path.stat().st_size > 0
        print("  ✓ Numeric variable plot generated")

        # Test categorical plot
        categorical_series = pd.Series(
            np.random.choice(["A", "B", "C"], size=100), name="test_categorical"
        )
        categorical_plot_path = tmpdir / "categorical_test.png"
        plot_variable_distribution(
            categorical_series, categorical_plot_path, is_target=True
        )

        assert categorical_plot_path.exists()
        assert categorical_plot_path.stat().st_size > 0
        print("  ✓ Categorical variable plot generated")

    print("✓ All plot tests passed\n")


def test_generate_summary_report():
    """Test summary report generation."""
    print("Testing generate_summary_report...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create sample dataframe
        df = pd.DataFrame(
            {
                "Subject": [f"SUB{i:03d}" for i in range(50)],
                "Age": np.random.randint(20, 80, size=50),
                "IQ": np.random.normal(100, 15, size=50),
                "Gender": np.random.choice(["M", "F"], size=50),
                "Group": np.random.choice(["Control", "Patient"], size=50),
            }
        )

        target_columns = ["Age", "Gender"]

        summary_df = generate_summary_report(df, target_columns, tmpdir)

        # Check that files were created
        assert (tmpdir / "summary_detailed.json").exists()
        assert (tmpdir / "summary_numeric.parquet").exists()
        assert (tmpdir / "summary_numeric.csv").exists()
        assert (tmpdir / "summary_categorical.parquet").exists()
        assert (tmpdir / "summary_categorical.csv").exists()
        print("  ✓ All summary files created")

        # Check summary dataframe content
        assert len(summary_df) == 4  # Age, IQ, Gender, Group (excluding Subject)
        assert summary_df is not None
        print("  ✓ Summary dataframe has correct structure")

        # Load and verify numeric summary
        numeric_df = pd.read_parquet(tmpdir / "summary_numeric.parquet")
        assert "Age" in numeric_df["variable"].values
        assert "IQ" in numeric_df["variable"].values
        age_row = numeric_df[numeric_df["variable"] == "Age"].iloc[0]
        assert age_row["is_target"] == True
        iq_row = numeric_df[numeric_df["variable"] == "IQ"].iloc[0]
        assert iq_row["is_target"] == False
        print("  ✓ Numeric summary has correct target flags")

        # Load and verify categorical summary
        categorical_df = pd.read_parquet(tmpdir / "summary_categorical.parquet")
        assert "Gender" in categorical_df["variable"].values
        assert "Group" in categorical_df["variable"].values
        gender_row = categorical_df[categorical_df["variable"] == "Gender"].iloc[0]
        assert gender_row["is_target"] == True
        print("  ✓ Categorical summary has correct target flags")

    print("✓ All summary report tests passed\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("RUNNING DATA DISTRIBUTION CLI TESTS")
    print("=" * 60 + "\n")

    try:
        test_compute_variable_statistics()
        test_plot_variable_distribution()
        test_generate_summary_report()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
