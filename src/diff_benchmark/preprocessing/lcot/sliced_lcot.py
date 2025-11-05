import numpy as np
import torch
from ot.utils import get_coordinate_circle

from .lcot import (
    dist_emb_circle_paired,
    dist_emb_circle_pairwise,
    linear_circular_embedding,
    linear_circular_ot,
)


def get_projections_sphere(d, n_projections, seed=None):
    """
    Sample great circles, i.e. matrices on the Stiefel manifold of size d x 2.

    Inputs:
    - d: int - dimension
    - n_projections: int - number of projections

    Output:
    - ndarray of shape (n_projections, d, 2)
    """

    rng = np.random.default_rng(seed)

    Z = torch.from_numpy(rng.normal(size=(n_projections, d, 2)))
    projections, _ = torch.linalg.qr(Z)
    return projections


def projection_sphere_to_circle(x, projections):
    """
    Projection of x on the circle.

    Inputs:
    - x: ndarray, shape (...,n_samples, dim) - samples on the sphere
    - projections: shape (n_projections, dim, 2), optional - Projection matrix

    Outputs:
    - return projections on S^1, shape (n_projections, ..., n_samples)
    """
    # n, d = x.shape
    n_projections = projections.shape[0]

    # Projection on S^1
    # Projection on plane
    Xp = torch.einsum("ikj, ...k -> i...j", projections, x)

    # Projection on sphere
    Xp = Xp / torch.sqrt(torch.sum(Xp**2, -1, keepdims=True))

    tuple_size = tuple([n_projections] + list(x.shape[:-1]))

    # Get coordinates on [0,1[
    Xp_coords = torch.reshape(
        get_coordinate_circle(torch.reshape(Xp, (-1, 2))), tuple_size
    )

    return Xp_coords


def sliced_lcot(
    X_s, X_t, a=None, b=None, n_projections=50, projections=None, seed=None
):
    """
    Sliced LCOT from [1]

    Inputs:
    - X_s: ndarray, shape (n_samples_a, dim) - samples in the source domain
    - X_t: ndarray, shape (n_samples_b, dim) - samples in the target domain
    - a : ndarray, shape (n_samples_a,), optional - samples weights in the source domain
    - b : ndarray, shape (n_samples_b,), optional - samples weights in the target domain
    - n_projections : int, optional - Number of projections used for the Monte-Carlo approximation
    - projections: shape (n_projections, dim, 2), optional - Projection matrix (n_projections and seed are not used in this case)
    - seed: int or RandomState or None, optional - Seed used for random number generator
    - log: bool, optional - if True, sliced_wasserstein_sphere returns the projections used and their associated EMD.

    Outputs:
    - return cost

    [1] Liu, X., Bai, Y., Martín, R. D., Shi, K., Shahbazi, A., Landman, B. A., Chang, C., & Kolouri, S. Linear Spherical Sliced Optimal Transport: A Fast Metric for Comparing Spherical Data. ICLR 2025.
    """
    d = X_s.shape[-1]

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(
                X_s.shape[1], X_t.shape[1]
            )
        )

    if torch.any(torch.abs(torch.sum(X_s**2, axis=-1) - 1) > 10 ** (-4)):
        raise ValueError("X_s is not on the sphere.")
    if torch.any(torch.abs(torch.sum(X_t**2, axis=-1) - 1) > 10 ** (-4)):
        raise ValueError("X_t is not on the sphere.")

    if projections is None:
        projections = get_projections_sphere(d, n_projections, seed=seed)
        projections = projections.type(X_s.dtype).to(X_s.device)

    Xps_coords = projection_sphere_to_circle(X_s, projections)

    Xpt_coords = projection_sphere_to_circle(X_t, projections=projections)

    projected_lcot = linear_circular_ot(
        Xps_coords.T, Xpt_coords.T, u_weights=a, v_weights=b
    )
    res = torch.mean(projected_lcot)  # ** (1 / 2)

    return res


def embedding_slcot(
    x, weights=None, n_projections=50, projections=None, seed=None, ts=None
):
    """
    Embedding from Sliced LCOT [1]

    Inputs:
    - x: ndarray, shape (n_samples_a, dim) - samples to embed
    - weights : ndarray, shape (n_samples_a,), optional - samples weights in the source domain
    - n_projections : int, optional - Number of projections used for the Monte-Carlo approximation
    - projections: shape (n_projections, dim, 2), optional - Projection matrix (n_projections and seed are not used in this case)
    - seed: int or RandomState or None, optional - Seed used for random number generator
    - ts: points in [0,1] where to evaluate the embedding

    Outputs:
    - return ndarray, embedding of shape (len(ts), n_projections,)

    [1] Liu, X., Bai, Y., Martín, R. D., Shi, K., Shahbazi, A., Landman, B. A., Chang, C., & Kolouri, S. Linear Spherical Sliced Optimal Transport: A Fast Metric for Comparing Spherical Data. ICLR 2025.

    """
    d = x.shape[-1]
    device = x.device
    dtype = x.dtype

    if ts is None:
        ts = torch.linspace(0, 1, 101, dtype=dtype, device=device)[:-1]

    if projections is None:
        projections = get_projections_sphere(d, n_projections, seed=seed)
        projections = projections.type(dtype).to(device)

    x_circle = projection_sphere_to_circle(x, projections)
    emb_circle = linear_circular_embedding(ts, x_circle.T, weights)
    return emb_circle


def embedding_slcot_batch(
    x, weights=None, n_projections=50, projections=None, seed=None, ts=None
):
    """
    Embedding from Sliced LCOT [1]

    Inputs:
    - x: ndarray, shape (..., n_samples_a, dim) - samples to embed
    - weights : ndarray, shape (..., n_samples_a,), optional - samples weights in the source domain
    - n_projections : int, optional - Number of projections used for the Monte-Carlo approximation
    - projections: shape (n_projections, dim, 2), optional - Projection matrix (n_projections and seed are not used in this case)
    - seed: int or RandomState or None, optional - Seed used for random number generator
    - ts: points in [0,1] where to evaluate the embedding

    Outputs:
    - return ndarray, embedding of shape (len(ts), n_projections, ...)

    [1] Liu, X., Bai, Y., Martín, R. D., Shi, K., Shahbazi, A., Landman, B. A., Chang, C., & Kolouri, S. Linear Spherical Sliced Optimal Transport: A Fast Metric for Comparing Spherical Data. ICLR 2025.

    """
    n, d = x.shape[-2], x.shape[-1]
    device = x.device
    dtype = x.dtype

    if ts is None:
        ts = torch.linspace(0, 1, 101, dtype=dtype, device=device)[:-1]

    if projections is None:
        projections = get_projections_sphere(d, n_projections, seed=seed)
        projections = projections.type(dtype).to(device)

    x_circle = projection_sphere_to_circle(x, projections)

    tuple_size = tuple([len(ts)] + list(x_circle.shape[:-1]))

    if weights is not None:
        weights = torch.repeat_interleave(weights[None], x_circle.shape[0], dim=0)
        weights = torch.movedim(weights, -1, 0).reshape(n, -1)

    emb_circle = linear_circular_embedding(
        ts, torch.movedim(x_circle, -1, 0).reshape(n, -1), weights
    )

    return torch.reshape(emb_circle, tuple_size)


class LSSOT:
    def __init__(
        self,
        d,
        n_projections,
        num_ts=100,
        random_state=42,
        device="cpu",
        dtype=torch.float,
    ):
        self.projections = (
            get_projections_sphere(d, n_projections, seed=random_state)
            .type(dtype)
            .to(device)
        )

        self.ts = torch.linspace(0, 1, num_ts + 1, dtype=dtype, device=device)[:-1]
        self.dtype = dtype
        self.device = device

    def lssot(self, Xs, Xt, u_weights=None, v_weights=None):
        return sliced_lcot(
            Xs, Xt, a=u_weights, b=v_weights, projections=self.projections
        )

    def get_features(self, x, weights=None):
        return embedding_slcot(
            x.type(self.dtype).to(self.projections.device),
            (
                weights.type(self.dtype).to(self.projections.device)
                if weights is not None
                else None
            ),
            projections=self.projections,
            ts=self.ts,
        )


class EmbeddingCircle:
    def __init__(
        self,
        d,
        n_projections,
        num_ts=100,
        random_state=None,
        device="cpu",
        dtype=torch.float,
    ):
        self.projections = (
            get_projections_sphere(d, n_projections, seed=random_state)
            .type(dtype)
            .to(device)
        )

        self.ts = torch.linspace(0, 1, num_ts + 1, dtype=dtype, device=device)[:-1]
        self.dtype = dtype
        self.device = device

    def get_features(self, x, weights=None):
        """Compute embedding of x on the circle.

        Args:
            x (torch.tensor): coordinates on the sphere (d, n_samples)
            weights (torch.tensor, optional): values of each point (n_batch, n_samples). Defaults to None.

        Returns:
            embeddings: embedding of x on the circle (n_batch, len(ts), n_projections)
        """
        if weights is not None and x.ndim != weights.ndim + 1:
            x = x.unsqueeze(0).expand(weights.shape[0], -1, -1)

        # shape (len(ts), n_projections, n)
        embedding = embedding_slcot_batch(
            x, weights, projections=self.projections, ts=self.ts
        )
        # reshape in shape (n, len(ts), n_projections)
        return torch.movedim(embedding, -1, 0)

    def get_dist(self, Xs, Xt, u_weights=None, v_weights=None):
        """
        Xs of shape (k, d), Xt of shape (k, d)
        """
        # shape (len(ts), n_projections)
        embedding_Xs = self.get_features(Xs, u_weights)
        embedding_Xt = self.get_features(Xt, v_weights)

        dist = dist_emb_circle_paired(
            embedding_Xs.reshape(-1), embedding_Xt.reshape(-1)
        )
        return dist

    def get_dist_batch(self, Xs, Xt, u_weights=None, v_weights=None):
        """
        Should work for:
        - Xs (n, k, d), Xt (m, k, d), u_weights (n, k), v_weights (m, k)
        - Xs (n, k, d), Xt (m, k, d), u_weights None, v_weights None
        - Xs (k, d), Xt (k, d), u_weights (n, k), v_weights (m, k)

        Output of shape (n, m)
        """
        if u_weights is not None and Xs.ndim != u_weights.ndim + 1:
            Xs = Xs.unsqueeze(0).expand(u_weights.shape[0], -1, -1)

        if v_weights is not None and Xt.ndim != v_weights.ndim + 1:
            Xt = Xt.unsqueeze(0).expand(v_weights.shape[0], -1, -1)

        n = Xs.shape[0]
        m = Xt.shape[0]

        # shape (n, len(ts), n_projections)
        embedding_Xs = self.get_features(Xs, u_weights)
        embedding_Xt = self.get_features(Xt, v_weights)

        # shape (n, m)
        dist = dist_emb_circle_pairwise(
            embedding_Xs.reshape(n, -1), embedding_Xt.reshape(m, -1)
        )
        return dist


class EmbeddingCircleWeights:
    def __init__(
        self,
        d,
        n_projections,
        x_coords,  # coordinates on the sphere (n_samples, d)
        num_ts=100,
        random_state=None,
        device="cpu",
        dtype=torch.float,
    ):
        self.n_projections = n_projections

        self.projections = (
            get_projections_sphere(d, n_projections, seed=random_state)
            .type(dtype)
            .to(device)
        )

        self.x_coords = x_coords  # shape (n_samples, d)

        # shape (n_samples, n_projections)
        self.x_circle = projection_sphere_to_circle(x_coords, self.projections).T

        u_values, u_sorter = torch.sort(self.x_circle, dim=0)
        self.u_values = u_values  # shape (n_samples, n_projections)
        self.u_sorter = u_sorter  # shape (n_samples, n_projections)

        self.ts = torch.linspace(0, 1, num_ts + 1, dtype=dtype, device=device)[:-1]
        self.dtype = dtype
        self.device = device

    def get_features(self, weights):
        """Compute embedding of x on the circle.

        Args:
            weights (torch.tensor, optional): values of each point (n_batch, n_samples)

        Returns:
            embeddings: embedding of x on the circle (n_batch, len(ts) * n_projections)
        """
        n_distr = weights.shape[0]
        n = self.x_coords.shape[-2]

        # shape (n_distrs, n_samples, n_projections)
        u_values = self.u_values.unsqueeze(0).expand(weights.shape[0], -1, -1)
        u_sorter = self.u_sorter.unsqueeze(0).expand(weights.shape[0], -1, -1)

        # # shape (n_samples, n_distrs, n_projections)
        u_values = torch.movedim(u_values, 1, 0)
        u_sorter = torch.movedim(u_sorter, 1, 0)

        weights = torch.repeat_interleave(weights[None], self.n_projections, dim=0)
        weights = torch.movedim(weights, -1, 0)
        weights = torch.transpose(weights, 1, 2)
        weights = torch.gather(weights, 0, u_sorter)

        # shape (len(ts), n_projections, n_distr)
        embedding = linear_circular_embedding(
            self.ts,
            u_values.reshape(n, -1),
            weights.reshape(n, -1),
            requires_sort=False,
        )

        # reshape in shape (n_distr, n_projections * len(ts))
        return torch.movedim(embedding, -1, 0).reshape(n_distr, -1)

    def get_dist(self, u_weights=None, v_weights=None):
        """
        Xs of shape (k, d), Xt of shape (k, d)
        """
        # shape (len(ts), n_projections)
        embedding_Xs = self.get_features(u_weights)
        embedding_Xt = self.get_features(v_weights)

        dist = dist_emb_circle_paired(
            embedding_Xs.reshape(-1), embedding_Xt.reshape(-1)
        )
        return dist

    def get_dist_batch(self, u_weights=None, v_weights=None):
        """
        u_weights: shape (n, k)
        v_weights: shape (m, k)

        Output of shape (n, m)
        """
        n = u_weights.shape[0]
        m = v_weights.shape[0]

        # shape (n, len(ts), n_projections)
        embedding_Xs = self.get_features(u_weights)
        embedding_Xt = self.get_features(v_weights)

        # shape (n, m)
        dist = dist_emb_circle_pairwise(
            embedding_Xs.reshape(n, -1), embedding_Xt.reshape(m, -1)
        )

        return dist
