from diff_benchmark.analysis.save_results import is_cached


def test_is_cached_hit(sample_all_results):
    """Check that is_cached returns True for existing run_id."""
    output_dir = sample_all_results
    assert is_cached("2dcnn_1f5f8fac", output_dir)
    assert is_cached("mlp_a1b2c3d4", output_dir)


def test_is_cached_miss(sample_all_results):
    """Check that is_cached returns False for non-existing run_id."""
    output_dir = sample_all_results
    assert not is_cached("nonexistent_run_id", output_dir)


def test_is_cached_no_file(tmp_path):
    """Check that is_cached returns False if results file does not exist."""
    output_dir = tmp_path
    assert not is_cached("any_run_id", output_dir)
