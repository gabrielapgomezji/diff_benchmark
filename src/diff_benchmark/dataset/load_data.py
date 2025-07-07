import h5py


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


# def load_input():
#     embeddings, power, metadata = load_embeddings_and_power_from_h5("path/to/h5file.h5")
#     imput_data = embeddings
#     return input_data
