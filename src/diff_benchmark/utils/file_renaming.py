from pathlib import Path

from joblib import Parallel, delayed


def rename_files_in_parallel(
    base_path: str, old_file_name: str, new_file_name: str, n_jobs: int = -1
):
    """
    Renames files in parallel within a specified directory structure.
    This function searches for a specific file in the 'processed' subdirectory of each subject directory
    under the given base path and renames it if found. The renaming operation is performed in parallel
    using the specified number of jobs.
    Parameters:
        base_path (str): The path to the base directory containing subject directories.
        old_file_name (str): The name of the file to be renamed.
        new_file_name (str): The new name for the file.
        n_jobs (int, optional): The number of jobs to run in parallel. Default is -1, which uses all available cores.
    Returns:
        List[str]: A list of messages indicating the result of each renaming operation.
    """
    base_dir = Path(base_path)
    subject_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

    def rename_file(subject_dir: Path):
        processed_dir = subject_dir / "processed"
        old_file = processed_dir / old_file_name
        new_file = processed_dir / new_file_name

        if not processed_dir.exists():
            return f"Skipped (no 'processed' folder): {processed_dir}"

        if old_file.exists():
            old_file.rename(new_file)
            return f"Renamed: {old_file} → {new_file}"
        else:
            return f"Skipped (not found): {old_file}"

    # Parallel processing
    _ = Parallel(n_jobs=n_jobs)(
        delayed(rename_file)(subject_dir) for subject_dir in subject_dirs
    )
