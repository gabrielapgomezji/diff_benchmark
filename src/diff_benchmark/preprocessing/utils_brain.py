from dataclasses import dataclass
from typing import Any

import h5py
import networkx as nx
import numpy as np
from dipy.core.gradients import gradient_table
from scipy.linalg import LinAlgError
from tqdm import tqdm


def extract_data(raw_data_path: str):
    """Extracts data from an HDF5 file containing brain surface information.
    Args:
        raw_data_path (str): Path to the HDF5 file containing the raw data.
    Returns:
        dict: A dictionary containing the following keys:
            - "dwi_signal" (numpy.ndarray): Diffusion-weighted imaging signal with shape [n_vertices, n_directions].
            - "labels" (numpy.ndarray): Surface labels with shape [n_vertices].
            - "vertex_indices" (numpy.ndarray): 1D list of vertex indices.
            - "coords" (numpy.ndarray): Surface coordinates with shape [n_vertices, 3].
            - "faces" (numpy.ndarray): Surface faces with shape [n_faces, 3].
            - "bvals" (numpy.ndarray): Original b-values.
            - "bvecs" (numpy.ndarray): Original b-vectors.
    """
    with h5py.File(raw_data_path, "r") as f:
        data = {
            "dwi_signal": f["left_dwi_surface"][:],  # shape: [n_vertices, n_directions]
            "labels": f["surface_labels"][:],  # shape: [n_vertices]
            "vertex_indices": f["nodes_left"][:],  # 1D list of vertex indices
            "coords": f["surface_coordinates"][:],  # shape: [n_vertices, 3]
            "faces": f["surface_faces"][:],  # shape: [n_faces, 3]
            "bvals": f["bvals"][:],  # original bvals
            "bvecs": f["bvecs"][:],  # original bvecs
        }
    return data


def create_subgraph(data: dict):
    """
    Creates a subgraph from the given data.

    Args:
        data (dict): A dictionary containing "faces" and "vertex_indices".

    Returns:
        networkx.Graph: The subgraph containing only the specified vertex indices.
    """
    faces = np.array(data["faces"]).T
    edge_index = np.concatenate([faces[:2], faces[1:], faces[::2]], axis=1)
    edge_index = np.unique(edge_index, axis=1)

    g_graph = nx.Graph()
    g_graph.add_edges_from(edge_index.T)

    graph = g_graph.subgraph(data["vertex_indices"])
    return graph


def load_processed_file(file_path: str):
    """
    Loads and concatenates attenuation data across all b-values from an HDF5 file.

    Args:
        file_path (Path or str): Path to the HDF5 file.

    Returns:
        np.ndarray: Concatenated array of shape (n_total_vertices, n_directions).
        dict: Metadata including per-bval vertex counts and bval keys.
    """
    all_data = []
    metadata = {"bval_keys": [], "vertices_per_bval": []}

    with h5py.File(file_path, "r") as f:
        metadata["sphere_vertices"] = f["sphere_vertices"][()]
        metadata["sphere_faces"] = (
            f["sphere_faces"][()] if "sphere_faces" in f else None
        )
        metadata["sphere_edges"] = (
            f["sphere_edges"][()] if "sphere_edges" in f else None
        )
        for bval_key in f:
            if not bval_key.startswith("bval_"):
                continue  # Skip non-bval groups

            metadata["bval_keys"].append(bval_key)
            bval_group = f[bval_key]

            vertex_data = []
            for vertex_key in bval_group:
                attenuation = bval_group[vertex_key]["attenuation"][()]
                vertex_data.append(attenuation)

            vertex_data = np.array(vertex_data)  # Shape: (n_vertices, n_directions)
            metadata["vertices_per_bval"].append(vertex_data)
            all_data.append(vertex_data)

    if all_data:
        concatenated = np.stack(all_data, axis=1)
    else:
        concatenated = np.empty((0, 0))

    return concatenated, metadata


def normalize(data):
    """
    Normalize the input data by subtracting the mean and dividing by the standard deviation.
    Parameters:
        data (numpy.ndarray): Input array to be normalized.
    Returns:
        numpy.ndarray: Normalized array where each element is transformed as (element - mean) / std.
    """

    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data


@dataclass
class ComputationConfig:
    bvals_to_compute: list
    sphere: Any
    model: Any
    gtab0: Any
    graph_ins: nx.Graph
    normalize_input: bool


def _fit_signal_with_fallback(model, signal, vertex, data, graph_ins, normalize_input):
    """Try fitting signal, fall back to neighbor averaging if SVD fails."""
    try:
        return model.fit(signal)
    except LinAlgError:
        print(f"Vertex {vertex} - SVD did not converge. Using neighbor average.")
        neighbors = list(nx.neighbors(graph_ins, vertex))
        if not neighbors:
            print(f"Vertex {vertex} has no neighbors. Skipping.")
            return None

        neighbor_signals = np.array([data["dwi_signal"][n] for n in neighbors])
        avg_signal = np.mean(neighbor_signals, axis=0)
        if normalize_input:
            avg_signal = normalize(avg_signal)
        try:
            return model.fit(avg_signal)
        except LinAlgError:
            print(f"Vertex {vertex} - Averaged neighbor signal also failed. Skipping.")
            return None


def compute_data(
    data, bvals_to_compute, sphere, model, gtab0, graph_ins, normalize_input
):
    """
    Computes the attenuation data for each b-value and vertex.

    Args:
        data (dict): Input data containing dwi_signal, vertex_indices, labels, etc.
        bvals_to_compute (list): List of b-values to compute.
        sphere (object): Sphere object with vertices and edges.
        model (object): Model used for fitting the signal.
        gtab0 (object): Gradient table for b0.
        graph_ins (networkx.Graph): Graph of the surface mesh.
        normalize_input (bool): Whether to normalize the input signal.

    Returns:
        dict: Computed results for all b-values.
    """
    all_results = {}
    for bval in bvals_to_compute:
        gtab_sphere = gradient_table(
            bvals=np.repeat(bval, len(sphere.vertices)), bvecs=sphere.vertices
        )

        subject_spheres = []

        print(f"Computing bval {bval}")

        for _, vertex in enumerate(tqdm(data["vertex_indices"], desc=f"B={bval}")):
            signal = (
                normalize(data["dwi_signal"][vertex])
                if normalize_input
                else data["dwi_signal"][vertex]
            )

            # try:
            #     fit = model.fit(signal)
            # except LinAlgError:
            #     print(
            #         f"Vertex {vertex} - SVD did not converge. Using neighbor average."
            #     )

            #     neighbors = list(nx.neighbors(graph_ins, vertex))
            #     if not neighbors:
            #         print(f"Vertex {vertex} has no neighbors. Skipping.")
            #         continue

            #     neighbor_signals = np.array([data["dwi_signal"][n] for n in neighbors])
            #     avg_signal = np.mean(neighbor_signals, axis=0)

            #     if normalize_input:
            #         avg_signal = normalize(avg_signal)

            #     try:
            #         fit = model.fit(avg_signal)
            #     except LinAlgError:
            #         print(
            #             f"Vertex {vertex} - Averaged neighbor signal also failed. Skipping."
            #         )
            #         continue
            fit = _fit_signal_with_fallback(
                model, signal, vertex, data, graph_ins, normalize_input
            )
            if fit is None:
                continue

            b0_val = fit.predict(gtab0)
            attenuation = fit.predict(gtab_sphere) / b0_val

            vertex_data = {
                "vertex": vertex,
                "attenuation": attenuation.astype(np.float32),
                "neighbors": list(nx.neighbors(graph_ins, vertex)),
                "label": data["labels"][vertex],
            }
            subject_spheres.append(vertex_data)

        all_results[str(bval)] = subject_spheres

    return all_results


# def save_output(all_results, save_path, name, sphere, data, sub):
#     """
#     Saves the computed results to an HDF5 file.

#     Args:
#         all_results (dict): Computed results for all b-values.
#         save_path (Path): Path to save the output file.
#         name (str): Name of the output file.
#         sphere (object): Sphere object with vertices and edges.
#         data (dict): Input data containing bvals and bvecs.
#         sub (str): Subject identifier.
#     """
#     out_file = save_path / f"{name}_all_bvals.h5"
#     os.makedirs(save_path, exist_ok=True)

#     with h5py.File(out_file, "w") as f:
#         # Save sphere geometry once
#         f.create_dataset("sphere_vertices", data=sphere.vertices)
#         f.create_dataset("sphere_faces", data=sphere.faces)
#         f.create_dataset("sphere_edges", data=sphere.edges)

#         # Save original bvals and bvecs
#         f.create_dataset("bvals", data=data["bvals"])
#         f.create_dataset("bvecs", data=data["bvecs"])

#         # Metadata
#         meta = f.create_group("meta")
#         meta.attrs["subject"] = sub
#         meta.attrs["surface"] = "MSMAll.32k_fs_LR"
#         meta.attrs["hemisphere"] = "left"
#         meta.attrs["model"] = "MAPMRI"
#         meta.attrs["interpolation"] = "linear"

#         # Save per-bvalue results
#         for bval_str, vertices in all_results.items():
#             grp = f.create_group(f"bval_{bval_str}")
#             for i, vdata in enumerate(vertices):
#                 vgrp = grp.create_group(f"vertex_{i}")
#                 vgrp.create_dataset("attenuation", data=vdata["attenuation"])
#                 vgrp.create_dataset("surface_vertex", data=vdata["vertex"])
#                 vgrp.create_dataset(
#                     "mesh_neighbors", data=np.array(vdata["neighbors"], dtype=np.int32)
#                 )
#                 vgrp.create_dataset("deen_insula_label", data=vdata["label"])

#     print(f"Saved full output with all b-values to:\n{out_file}")
