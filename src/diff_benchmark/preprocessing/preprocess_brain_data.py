from pathlib import Path

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.core.subdivide_octahedron import create_unit_sphere

from diff_benchmark.preprocessing.base_brain_data import BrainDataPreprocessor
from diff_benchmark.preprocessing.models_configuration import get_model_configs
from diff_benchmark.preprocessing.utils_brain import (
    compute_data,
    create_subgraph,
    extract_data,
    save_output,
)


class DefaultBrainPreprocessor(BrainDataPreprocessor):
    """
    DefaultBrainPreprocessor is a class that extends the BrainDataPreprocessor
    to handle the preprocessing of brain data for individual subjects.
    Methods:
        preprocess_subject(raw_data_path: Path, subject_id: str, save_path: Path):
            Preprocesses the brain data for a given subject by extracting data,
            computing results using specified models, and saving the output to
            the designated path.
        save_subject_info(subject_id: str):
            Saves information related to the specified subject. This is a placeholder
            method that currently only prints a message.
        save_dataset_info():
            Saves information related to the entire dataset. This is a placeholder
            method that currently only prints a message.
    """

    def preprocess_subject(self, raw_data_path: Path, subject_id: str, save_path: Path):
        save_path.mkdir(parents=True, exist_ok=True)

        normalize_input = False
        sphere = create_unit_sphere(6)

        data = extract_data(raw_data_path=raw_data_path)
        gtab_original = gradient_table(bvals=data["bvals"], bvecs=data["bvecs"])
        graph_ins = create_subgraph(data)
        gtab0 = gradient_table(bvals=np.repeat(1, 1), bvecs=np.array([[1, 0, 0]]))
        bvals_to_compute = [1000, 2000, 3000]
        model_configs = get_model_configs(gtab_original)

        for name, model in model_configs.items():
            output_file = save_path / f"{name}_all_bvals.h5"
            if output_file.exists():
                print(
                    f"Skipping {subject_id} model {name} because output already exists."
                )
                continue

            all_results = compute_data(
                data, bvals_to_compute, sphere, model, gtab0, graph_ins, normalize_input
            )
            save_output(all_results, save_path, name, sphere, data, subject_id)

    # def preprocess_dataset(self):
    #     base_path = Path(self.config["results_path"])
    #     subjects = [p.name for p in base_path.iterdir() if p.is_dir()]
    #     for sub in subjects:
    #         raw_path = base_path / sub / "raw_surface_data.h5"
    #         save_path = Path(self.config["results_path_2"]) / sub / "processed"
    #         self.preprocess_subject(raw_path, sub, save_path)

    def save_subject_info(self, subject_id: str):
        print(f"[{subject_id}] Saving subject info... (placeholder)")

    def save_dataset_info(self):
        print("Saving dataset info... (placeholder)")
