from pathlib import Path

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.core.subdivide_octahedron import create_unit_sphere

from diff_benchmark.preprocessing.models_configuration import get_model_configs
from diff_benchmark.preprocessing.utils_brain import (
    compute_data,
    create_subgraph,
    extract_data,
    save_output,
)


def preprocess_subject(raw_data_path: str, subject_id: str, save_path: str):
    """
    Preprocess a given raw data file for a specific subject.

    Args:
        raw_data_path (str): Path to the raw data file.
        subject_id (str): Subject ID for processing.
        save_path (str): Path to the folder where to save the data.

    Returns:
        None
    """

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Preprocessing parameters
    normalize_input = False  # Set to True if normalization is required
    sphere = create_unit_sphere(7)

    # ---------- LOAD RAW DATA ----------
    data = extract_data(raw_data_path=Path(raw_data_path))
    gtab_original = gradient_table(bvals=data["bvals"], bvecs=data["bvecs"])

    # ---------- CREATE MESH GRAPH ----------
    graph_ins = create_subgraph(data)

    # ---------- CREATE SPHERE ----------
    gtab0 = gradient_table(bvals=np.repeat(1, 1), bvecs=np.array([[1, 0, 0]]))

    # ---------- MULTI-BVALUE PROCESS ----------
    bvals_to_compute = [1000, 2000, 3000]
    model_configs = get_model_configs(gtab_original)

    # Process and save results for each model
    for name, model in model_configs.items():
        # Compute the data
        all_results = compute_data(
            data, bvals_to_compute, sphere, model, gtab0, graph_ins, normalize_input
        )
        # Save the output
        save_output(all_results, save_path, name, sphere, data, subject_id)
