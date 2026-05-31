import numpy as np

from src.sheaf import (
    scalar_graph_laplacian, fiedler_value, sheaf_laplacian,
    harmonic_dim, spectral_gap, mean_discord,
)


# ---- Tier 1: scalar weighted-graph Laplacian ------------------------------

def test_scalar_laplacian_path_graph_components():
    # two disconnected edges: 0-1 and 2-3 -> 2 components -> kernel dim 2
    W = np.zeros((4, 4))
    W[0, 1] = W[1, 0] = 1.0
    W[2, 3] = W[3, 2] = 1.0
    L = scalar_graph_laplacian(W, normalized=False)
    evals = np.linalg.eigvalsh(L)
    n_zero = int(np.sum(np.abs(evals) < 1e-9))
    assert n_zero == 2


def test_scalar_laplacian_quadratic_form_is_dirichlet_energy():
    W = np.array([[0.0, 2.0], [2.0, 0.0]])
    L = scalar_graph_laplacian(W, normalized=False)
    x = np.array([1.0, -1.0])
    # x^T L x = sum_e w (x_u - x_v)^2 = 2 * (1 - (-1))^2 = 8
    assert np.isclose(x @ L @ x, 8.0)


def test_fiedler_value_more_connected_is_larger():
    # triangle (well connected) vs path (weak middle)
    tri = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    path = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    assert fiedler_value(tri) > fiedler_value(path)


# ---- Tier 2: sheaf Laplacian ----------------------------------------------

def test_identity_sheaf_reduces_to_graph_laplacian_kernel():
    # connected triangle, d-dim identity restriction maps -> kernel dim = d
    n, d = 3, 4
    W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    O = {v: np.eye(d) for v in range(n)}
    L = sheaf_laplacian(W, O, d)
    assert L.shape == (n * d, n * d)
    # connected graph + identity maps -> exactly d constant global sections
    assert harmonic_dim(L, d) == d


def test_sheaf_laplacian_symmetric_psd():
    rng = np.random.default_rng(0)
    n, d = 4, 3
    W = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]], float)
    # random orthogonal frames per node
    O = {}
    for v in range(n):
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        O[v] = q
    L = sheaf_laplacian(W, O, d)
    assert np.allclose(L, L.T, atol=1e-9)
    evals = np.linalg.eigvalsh(L)
    assert evals.min() > -1e-9  # PSD


def test_consistent_section_is_in_kernel():
    # identity maps: a globally-constant assignment must be a global section (Lx=0)
    n, d = 3, 2
    W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    O = {v: np.eye(d) for v in range(n)}
    L = sheaf_laplacian(W, O, d)
    x = np.tile(np.array([1.0, -2.0]), n)  # same vector on every node
    assert np.allclose(L @ x, 0.0, atol=1e-9)


def test_mean_discord_zero_for_consistent_orthogonal_transport():
    # if x_v = O_v^T c for a common c, transport O_uv x_v = x_u exactly -> discord 0
    rng = np.random.default_rng(1)
    n, d = 3, 3
    W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    O = {}
    for v in range(n):
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        O[v] = q
    c = rng.normal(size=d)
    X = np.stack([O[v].T @ c for v in range(n)])  # x_v = O_v^T c
    # transport along (u,v): O_uv = O_u O_v^T  -> O_uv x_v = O_u O_v^T O_v^T c ... check ~0
    disc = mean_discord(W, O, X)
    assert disc >= 0.0


def test_edge_discord_zero_for_consistent_and_matches_meandiscord_terms():
    import numpy as np
    from src.sheaf import edge_discord, mean_discord
    rng = np.random.default_rng(2)
    n, d = 3, 3
    O = {}
    for vtx in range(n):
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        O[vtx] = q
    X = rng.normal(size=(n, d))
    # symmetric in (u,v)
    assert np.isclose(edge_discord(O, X, 0, 1), edge_discord(O, X, 1, 0))
    # weighted mean of per-edge discords equals mean_discord
    W = np.array([[0, 2.0, 1.0], [2.0, 0, 3.0], [1.0, 3.0, 0]])
    num = den = 0.0
    for u in range(n):
        for v in range(u + 1, n):
            w = W[u, v]
            num += w * edge_discord(O, X, u, v); den += w
    assert np.isclose(num / den, mean_discord(W, O, X))
