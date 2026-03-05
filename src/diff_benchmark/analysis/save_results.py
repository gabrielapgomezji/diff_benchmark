import json
from pathlib import Path

from filelock import FileLock

from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


def is_cached(
    run_id: str, output_dir: Path, results_filename: str = "all_results.json"
) -> bool:
    """Check whether results for *run_id* are already stored on disk.

    Args:
        run_id: The unique run identifier to look up.
        output_dir: Directory that contains the results file.
        results_filename: Name of the JSON results file.

    Returns:
        ``True`` if a matching entry is found in the results file.
    """
    out_path = output_dir / results_filename
    if not out_path.exists():
        return False
    with open(out_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    return any(
        result.get("pipeline", {}).get("run_id") == run_id for result in all_results
    )


def _make_json_serializable(obj) -> object:
    """Recursively convert *obj* to a JSON-serializable form.

    Non-serializable objects are replaced with a ``"<non-serializable: …>"``
    string so the overall structure can still be written to JSON.

    Args:
        obj: Any Python object.

    Returns:
        A JSON-safe equivalent of *obj*.
    """
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return f"<non-serializable: {obj.__class__.__name__}>"


def save_model_results(
    summary: dict,
    output_dir: Path,
    results_filename: str = "all_results.json",
) -> None:
    """Append *summary* to the aggregated results JSON file.

    The write is protected by a :class:`FileLock` so multiple processes can
    safely append results concurrently.

    Args:
        summary: Dict containing model results and metadata.  Must include
            ``summary["config"]["runtime"]["run_id"]`` and
            ``summary["model_name"]`` for the log message.
        output_dir: Root directory for experiment outputs.
        results_filename: Name of the results JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / results_filename

    lock = FileLock(f"{out_path}.lock")
    with lock:
        all_results = []
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)

        all_results.append(_make_json_serializable(summary))

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    run_id = summary.get("config", {}).get("runtime", {}).get("run_id", "?")
    logger.info(f"Saved results for {summary.get('model_name')} (run_id={run_id})")
