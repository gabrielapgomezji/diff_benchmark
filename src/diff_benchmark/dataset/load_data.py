from pathlib import Path

import h5py
import numpy as np


def load_embeddings_and_power_from_h5(h5_path):
    """
    Load embeddings, power, and metadata from an HDF5 file.

    Returns:
        embeddings (dict): Dictionary mapping b-value (int) to embedding array
        power (np.ndarray): Array of shape [num_nodes, 3]
        metadata (dict): Dictionary of metadata attributes
    """
    embeddings = {}
    metadata = {}

    with h5py.File(h5_path, "r") as f:
        # Load embeddings
        emb_grp = f["embeddings"]
        for key in emb_grp:
            bval = int(key[1:])  # from "b1000" to 1000
            embeddings[bval] = emb_grp[key][:]

        # Load power
        power = f["power"][:]

        # Load metadata
        meta_grp = f["metadata"]
        metadata = {k: meta_grp.attrs[k] for k in meta_grp.attrs}

    return embeddings, power, metadata


def load_attenuation_from_h5(h5_path: Path):
    """
    Load attenuation data and associated metadata from the custom HDF5 file.

    Returns:
        attenuation_data (dict): Dictionary mapping b-value (int) to an array of shape [num_vertices, num_directions]
        metadata (dict): Contains additional arrays such as surface_vertex, neighbors, deen_insula_label
                         and global attributes like subject, hemisphere, surface, model, interpolation
    """
    attenuation_data = {}
    metadata = {}

    with h5py.File(h5_path, "r") as f:
        # Load global metadata
        meta_grp = f["meta"]
        for key in meta_grp.attrs:
            metadata[key] = meta_grp.attrs[key]

        # Load bvals and bvecs
        metadata["bvals"] = f["bvals"][:]
        metadata["bvecs"] = f["bvecs"][:]

        # Sphere geometry
        metadata["sphere_vertices"] = f["sphere_vertices"][:]
        metadata["sphere_faces"] = f["sphere_faces"][:]
        metadata["sphere_edges"] = f["sphere_edges"][:]

        # Load attenuation data per b-value
        for bval_grp in f:
            if not bval_grp.startswith("bval_"):
                continue

            bval = int(bval_grp.split("_")[1])
            vertex_group = f[bval_grp]
            num_vertices = len(vertex_group)
            sample_vertex = next(iter(vertex_group.values()))
            num_directions = sample_vertex["attenuation"].shape[0]

            attenuation_array = np.zeros((num_vertices, num_directions))
            surface_vertices = np.zeros(num_vertices, dtype=np.int32)
            labels = np.zeros(num_vertices, dtype=np.int32)
            neighbors = []

            for i, (_, vgrp) in enumerate(vertex_group.items()):
                attenuation_array[i] = vgrp["attenuation"][:]
                surface_vertices[i] = vgrp["surface_vertex"][()]
                labels[i] = vgrp["deen_insula_label"][()]
                neighbors.append(vgrp["mesh_neighbors"][:])

            attenuation_data[bval] = attenuation_array
            metadata[f"surface_vertices_b{bval}"] = surface_vertices
            metadata[f"labels_b{bval}"] = labels
            metadata[f"neighbors_b{bval}"] = neighbors

    return attenuation_data, metadata
