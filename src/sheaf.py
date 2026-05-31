"""Cellular sheaf Laplacians over the attention graph (Exp 8).

Two constructions (see docs/superpowers/specs/2026-05-31-experiment8-sheaves-design.md):

* Tier 1 -- scalar weighted-graph Laplacian (d=1, restriction = sqrt weight): the
  shape-only baseline. Reduces to L = D - W; Fiedler value = algebraic connectivity.
* Tier 2 -- Connection-Laplacian sheaf (d-dim stalks, orthogonal restriction maps
  O_v per node): the sheaf Laplacian L_F = delta^T delta assembled from the
  Hansen-Ghrist block formulas. Measures consistency of *what flows*, not just shape.

Restriction-map convention (symmetric, undirected edge {u,v}): each node v carries an
orthogonal map O_v; the edge contributes ||O_v x_v - O_u x_u||^2 to the quadratic form
(weighted by the symmetrized attention weight). Block Laplacian:
    L[v,v] = sum_{e ∋ v} w_e * O_v^T O_v
    L[u,v] = - w_{uv} * O_u^T O_v
This is delta^T delta for the sheaf with restriction F_{v⊴e} = sqrt(w_e) O_v.

Pure numpy; no torch/model dependency, so fast to unit-test.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


# ---- Tier 1: scalar weighted-graph Laplacian ------------------------------

def scalar_graph_laplacian(W: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Weighted graph Laplacian of a symmetric weight matrix W (zero diagonal).

    normalized=True returns L_norm = I - D^{-1/2} W D^{-1/2} (isolated nodes kept
    as zero rows). normalized=False returns the combinatorial L = D - W.
    """
    W = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    if not normalized:
        return L
    dinv = np.zeros_like(deg)
    nz = deg > EPS
    dinv[nz] = 1.0 / np.sqrt(deg[nz])
    Dinv = np.diag(dinv)
    n = W.shape[0]
    # I - D^{-1/2} W D^{-1/2}, but keep isolated nodes' rows at 0 (no self-energy)
    Lnorm = np.eye(n) - Dinv @ W @ Dinv
    for i in range(n):
        if not nz[i]:
            Lnorm[i, :] = 0.0
            Lnorm[:, i] = 0.0
    return Lnorm


def fiedler_value(W: np.ndarray, normalized: bool = True) -> float:
    """Algebraic connectivity: smallest nonzero eigenvalue of the graph Laplacian."""
    L = scalar_graph_laplacian(W, normalized=normalized)
    evals = np.linalg.eigvalsh(L)
    nz = evals[evals > EPS]
    return float(nz.min()) if nz.size else 0.0


# ---- Tier 2: Connection-Laplacian sheaf -----------------------------------

def sheaf_laplacian(W: np.ndarray, O: dict, d: int) -> np.ndarray:
    """Assemble the (n*d) x (n*d) sheaf Laplacian from orthogonal node maps O.

    W: symmetric edge weights [n, n]. O: dict node -> orthogonal d x d matrix.
    Block formulas (symmetric, PSD):
        L[v,v] = sum_{u: w_uv>0} w_uv * O_v^T O_v
        L[u,v] = - w_uv * O_u^T O_v
    """
    W = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    L = np.zeros((n * d, n * d))

    def block(a, b):
        return slice(a * d, (a + 1) * d), slice(b * d, (b + 1) * d)

    for u in range(n):
        for v in range(u + 1, n):
            w = W[u, v]
            if w <= 0.0:
                continue
            Ou, Ov = O[u], O[v]
            # off-diagonal blocks
            ru, cu = block(u, v)
            L[ru, cu] += -w * (Ou.T @ Ov)
            rv, cv = block(v, u)
            L[rv, cv] += -w * (Ov.T @ Ou)
            # diagonal contributions
            du0, du1 = block(u, u)
            L[du0, du1] += w * (Ou.T @ Ou)
            dv0, dv1 = block(v, v)
            L[dv0, dv1] += w * (Ov.T @ Ov)
    # symmetrize against round-off
    return 0.5 * (L + L.T)


def normalized_sheaf_laplacian(L: np.ndarray, d: int) -> np.ndarray:
    """Delta_F = D^{-1/2} L D^{-1/2}, D = block-diag of L's diagonal d x d blocks."""
    nd = L.shape[0]
    n = nd // d
    Dinv = np.zeros_like(L)
    for v in range(n):
        sl = slice(v * d, (v + 1) * d)
        blk = L[sl, sl]
        # symmetric inverse sqrt via eigh; zero out if degenerate
        evals, evecs = np.linalg.eigh(0.5 * (blk + blk.T))
        inv = np.where(evals > EPS, 1.0 / np.sqrt(np.clip(evals, EPS, None)), 0.0)
        Dinv[sl, sl] = (evecs * inv) @ evecs.T
    out = Dinv @ L @ Dinv
    return 0.5 * (out + out.T)


def harmonic_dim(L: np.ndarray, d: int, tol: float = 1e-7) -> int:
    """dim ker(L) = number of (near-)zero eigenvalues = global sections H^0."""
    evals = np.linalg.eigvalsh(0.5 * (L + L.T))
    scale = max(1.0, float(np.max(np.abs(evals)))) if evals.size else 1.0
    return int(np.sum(evals < tol * scale))


def spectral_gap(L: np.ndarray, tol: float = 1e-7) -> float:
    """Smallest nonzero eigenvalue of a (sheaf) Laplacian."""
    evals = np.linalg.eigvalsh(0.5 * (L + L.T))
    scale = max(1.0, float(np.max(np.abs(evals)))) if evals.size else 1.0
    nz = evals[evals > tol * scale]
    return float(nz.min()) if nz.size else 0.0


def mean_discord(W: np.ndarray, O: dict, X: np.ndarray) -> float:
    """Mean per-edge transport disagreement ||O_v x_v - O_u x_u||^2 (weighted).

    X: [n, d] node vectors. Direct scalar measure of inconsistency of flow over the
    sheaf, independent of the eigendecomposition.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    num, den = 0.0, 0.0
    for u in range(n):
        for v in range(u + 1, n):
            w = W[u, v]
            if w <= 0.0:
                continue
            diff = O[v] @ X[v] - O[u] @ X[u]
            num += w * float(diff @ diff)
            den += w
    return float(num / den) if den > 0 else 0.0


def local_pca_frame(neighbors: np.ndarray, d: int) -> np.ndarray:
    """Orthonormal d x d frame from a node's neighborhood vectors via local PCA.

    neighbors: [k, d] reduced vectors of the node + its neighbors. Returns an
    orthogonal d x d matrix (the principal-axis frame); falls back to identity if
    degenerate.
    """
    if neighbors.shape[0] < 2:
        return np.eye(d)
    centered = neighbors - neighbors.mean(axis=0, keepdims=True)
    # right singular vectors = principal axes in R^d
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=True)
    except np.linalg.LinAlgError:
        return np.eye(d)
    Q = vt.T  # columns are principal axes; orthogonal d x d
    if Q.shape != (d, d):
        return np.eye(d)
    return Q
