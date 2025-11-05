import torch
from ot.lp.solver_1d import quantile_function
from torch.nn.functional import pad


def dist_emb_circle_paired(emb_u, emb_v):
    """
    Compute the distance between two embeddings on the circle.

    Inputs:
    - emb_u: shape (..., n_unif * n_projs) - embedding of the first distribution
    - emb_v: shape (..., n_unif * n_projs) - embedding of the second distribution

    Outputs:
    - dist: shape (...,) - distance between the two embeddings
    """

    diff = torch.abs(emb_u - emb_v)
    dist_uv = torch.minimum(diff, 1 - diff)
    return torch.mean(dist_uv**2, axis=-1)


def dist_emb_circle_pairwise(emb_u, emb_v):
    """
    Compute the pairwise distance between two embeddings on the circle.

    Inputs:
    - emb_u: shape (n, d) - embedding of the first distribution
    - emb_v: shape (m, d) - embedding of the second distribution

    Outputs:
    - dist: shape (m, n) - pairwise distance between the two embeddings
    """
    dist_uv = torch.minimum(
        torch.abs(emb_u[:, None] - emb_v[None, :]),
        1 - torch.abs(emb_u[:, None] - emb_v[None, :]),
    )
    return torch.mean(dist_uv**2, axis=-1)


def linear_circular_embedding(x, u_values, u_weights=None, requires_sort=True):
    """
    Inputs:
    - x: shape (m,), points where we evaluate the embedding
    - u_values: shape (n, ...) (coordinates on [0,1[, n: number of samples)
    - u_weights: shape (n, ...)

    Output:
    - embedding of shape (m, ...)
    """
    n = u_values.shape[0]
    device = u_values.device
    dtype = u_values.dtype
    u_values = u_values % 1

    if len(u_values.shape) == 1:
        u_values = torch.reshape(u_values, (n, 1))

    if u_weights is None:
        u_weights = torch.full(u_values.shape, 1.0 / n, dtype=dtype, device=device)
    elif u_weights.ndim != u_values.ndim:
        u_weights = torch.repeat_interleave(
            u_weights[..., None], u_values.shape[-1], dim=-1
        )

    if requires_sort:
        u_values, u_sorter = torch.sort(u_values, dim=0)
        u_weights = torch.gather(u_weights, 0, u_sorter)

    u_cdf = torch.cumsum(u_weights, 0)

    pad_width = [(1, 0), (0, 0)]
    how_pad = tuple(element for tupl in pad_width[::-1] for element in tupl)
    u_cdf = pad(u_cdf, how_pad, value=0)

    # shape (m, ...)
    q_s = x[:, None] - torch.sum(u_values * u_weights, axis=0)[None] + 0.5

    u_quantiles = quantile_function(q_s % 1, u_cdf, u_values)
    return (u_quantiles - x[:, None]) % 1


def linear_circular_ot(u_values, v_values, u_weights=None, v_weights=None):
    """
    LCOT from [1]

    Inputs:
    - u_values: shape (n, ...) - samples in the source domain (coordinates on [0,1[)
    - v_values: shape (m, ...) - samples in the target domain (coordinates on [0,1[)
    - u_weights: shape (n, ...), optional - weights of the first empirical distribution, if None then uniform weights are used
    - v_weights, shape (m, ...), optional - weights of the second empirical distribution, if None then uniform weights are used

    Outputs:
    - return batchs LCOT

    [1] Martin, R. D., Medri, I., Bai, Y., Liu, X., Yan, K., Rohde, G. K., & Kolouri, S. LCOT: Linear Circular Optimal Transport. ICLR 2024.
    """
    n = u_values.shape[0]
    device = u_values.device
    dtype = u_values.dtype
    u_values = u_values % 1

    if len(u_values.shape) == 1:
        u_values = torch.reshape(u_values, (n, 1))

    unif_s1 = torch.linspace(0, 1, 101, device=device, dtype=dtype)[:-1]

    emb_u = linear_circular_embedding(unif_s1, u_values, u_weights)
    emb_v = linear_circular_embedding(unif_s1, v_values, v_weights)

    dist_uv = torch.minimum(torch.abs(emb_u - emb_v), 1 - torch.abs(emb_u - emb_v))
    return torch.mean(dist_uv**2, axis=0)
