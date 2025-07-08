import numpy as np


def is_valid_embedding(embeddings):
    """Return True if embeddings are not all NaNs."""
    for array in embeddings.values():
        if not np.isnan(array).all():
            return True
    return False
