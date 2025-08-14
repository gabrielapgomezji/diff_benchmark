import os
from pathlib import Path

import h5py
import networkx as nx
import nibabel as nib
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

    G = nx.Graph()
    G.add_edges_from(edge_index.T)

    graph = G.subgraph(data["vertex_indices"])
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

            try:
                fit = model.fit(signal)
            except LinAlgError:
                print(
                    f"Vertex {vertex} - SVD did not converge. Using neighbor average."
                )

                neighbors = list(nx.neighbors(graph_ins, vertex))
                if not neighbors:
                    print(f"Vertex {vertex} has no neighbors. Skipping.")
                    continue

                neighbor_signals = np.array([data["dwi_signal"][n] for n in neighbors])
                avg_signal = np.mean(neighbor_signals, axis=0)

                if normalize_input:
                    avg_signal = normalize(avg_signal)

                try:
                    fit = model.fit(avg_signal)
                except LinAlgError:
                    print(
                        f"Vertex {vertex} - Averaged neighbor signal also failed. Skipping."
                    )
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


def save_output(all_results, save_path, name, sphere, data, sub):
    """
    Saves the computed results to an HDF5 file.

    Args:
        all_results (dict): Computed results for all b-values.
        save_path (Path): Path to save the output file.
        name (str): Name of the output file.
        sphere (object): Sphere object with vertices and edges.
        data (dict): Input data containing bvals and bvecs.
        sub (str): Subject identifier.
    """
    out_file = save_path / f"{name}_all_bvals.h5"
    os.makedirs(save_path, exist_ok=True)

    with h5py.File(out_file, "w") as f:
        # Save sphere geometry once
        f.create_dataset("sphere_vertices", data=sphere.vertices)
        f.create_dataset("sphere_faces", data=sphere.faces)
        f.create_dataset("sphere_edges", data=sphere.edges)

        # Save original bvals and bvecs
        f.create_dataset("bvals", data=data["bvals"])
        f.create_dataset("bvecs", data=data["bvecs"])

        # Metadata
        meta = f.create_group("meta")
        meta.attrs["subject"] = sub
        meta.attrs["surface"] = "MSMAll.32k_fs_LR"
        meta.attrs["hemisphere"] = "left"
        meta.attrs["model"] = "MAPMRI"
        meta.attrs["interpolation"] = "linear"

        # Save per-bvalue results
        for bval_str, vertices in all_results.items():
            grp = f.create_group(f"bval_{bval_str}")
            for i, vdata in enumerate(vertices):
                vgrp = grp.create_group(f"vertex_{i}")
                vgrp.create_dataset("attenuation", data=vdata["attenuation"])
                vgrp.create_dataset("surface_vertex", data=vdata["vertex"])
                vgrp.create_dataset(
                    "mesh_neighbors", data=np.array(vdata["neighbors"], dtype=np.int32)
                )
                vgrp.create_dataset("deen_insula_label", data=vdata["label"])

    print(f"Saved full output with all b-values to:\n{out_file}")


def dti_measure(model_fit, measure_list: list):
    # === Fit DTI model only if any DTI metrics requested ===
    if "FA" in measure_list:
        fa = model_fit.fa
        return fa

    if "MD" in measure_list:
        md = model_fit.md
        return md

    if "AD" in measure_list:
        ad = model_fit.ad
        return ad

    if "RD" in measure_list:
        rd = model_fit.rd
        return rd


def mapmri_measure(model_fit, measure_list: list):
    # === Fit MAP-MRI model only if needed ===
    if "RTOP" in measure_list:
        rtop = model_fit.rtop()
        return rtop

    if "RTAP" in measure_list:
        rtap = model_fit.rtap()
        return rtap

    if "RTPP" in measure_list:
        rtpp = model_fit.rtpp()
        return rtpp


def dti_measure2(model_fit, affine, measure_list: list, save_path: Path):
    # === Fit DTI model only if any DTI metrics requested ===
    if "FA" in measure_list:
        fa = model_fit.fa
        nib.save(
            nib.Nifti1Image(fa.astype(np.float32), affine), save_path / "FA.nii.gz"
        )

    if "MD" in measure_list:
        md = model_fit.md
        nib.save(
            nib.Nifti1Image(md.astype(np.float32), affine), save_path / "MD.nii.gz"
        )

    if "AD" in measure_list:
        ad = model_fit.ad
        nib.save(
            nib.Nifti1Image(ad.astype(np.float32), affine), save_path / "AD.nii.gz"
        )

    if "RD" in measure_list:
        rd = model_fit.rd
        nib.save(
            nib.Nifti1Image(rd.astype(np.float32), affine), save_path / "RD.nii.gz"
        )


def mapmri_measure2(model_fit, affine, measure_list: list, save_path: Path):
    # === Fit MAP-MRI model only if needed ===
    if "RTOP" in measure_list:
        rtop = model_fit.rtop()
        rtop = np.nan_to_num(rtop, nan=0.0, posinf=0.0, neginf=0.0)
        nib.save(
            nib.Nifti1Image(rtop.astype(np.float32), affine), save_path / "RTOP.nii.gz"
        )

    if "RTAP" in measure_list:
        rtap = model_fit.rtap()
        rtap = np.nan_to_num(rtap, nan=0.0, posinf=0.0, neginf=0.0)
        nib.save(
            nib.Nifti1Image(rtap.astype(np.float32), affine), save_path / "RTAP.nii.gz"
        )

    if "RTPP" in measure_list:
        rtpp = model_fit.rtpp()
        rtpp = np.nan_to_num(rtpp, nan=0.0, posinf=0.0, neginf=0.0)
        nib.save(
            nib.Nifti1Image(rtpp.astype(np.float32), affine), save_path / "RTPP.nii.gz"
        )
