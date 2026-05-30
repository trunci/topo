"""Attention matrix -> flag complex -> H1 persistent homology features.

Pure functions only (no model/torch dependency) so this module is fast to test.
"""
from __future__ import annotations

import numpy as np
import gudhi


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Elementwise-max symmetrization of a (causal) attention matrix."""
    return np.maximum(A, A.T)


def sparsify(W: np.ndarray, top_k: int | None = None, weight_floor: float = 0.0) -> np.ndarray:
    """Zero the diagonal, drop weak edges, keep top_k per node, re-symmetrize.

    top_k is applied per row, which breaks symmetry, so we re-symmetrize with max
    afterward (an edge survives if it is in either endpoint's top_k).
    """
    W = W.astype(float).copy()
    np.fill_diagonal(W, 0.0)
    if weight_floor > 0.0:
        W[W < weight_floor] = 0.0
    if top_k is not None:
        n = W.shape[0]
        kept = np.zeros_like(W)
        for i in range(n):
            row = W[i]
            nz = np.flatnonzero(row)
            if nz.size == 0:
                continue
            k = min(top_k, nz.size)
            top = nz[np.argpartition(row[nz], -k)[-k:]]
            kept[i, top] = row[top]
        W = np.maximum(kept, kept.T)
    return W


def build_simplex_tree(W: np.ndarray) -> gudhi.SimplexTree:
    """Build the flag (clique) complex with filtration f(edge) = 1 - weight.

    Vertices enter at 0; strong edges (weight near 1) enter early (near 0).
    Expanded to dimension 2 so triangles can fill in and kill 1-cycles.
    """
    n = W.shape[0]
    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i], filtration=0.0)
    iu = np.triu_indices(n, k=1)
    for i, j in zip(*iu):
        w = W[i, j]
        if w > 0.0:
            st.insert([int(i), int(j)], filtration=float(1.0 - w))
    st.expansion(2)
    return st


def h1_intervals(st: gudhi.SimplexTree) -> np.ndarray:
    """Return the dimension-1 persistence intervals as an (m, 2) array.

    persistence_dim_max=True is REQUIRED: without it, H1 classes living in the
    complex's maximal dimension (e.g. an unfilled cycle) are silently dropped.
    """
    st.compute_persistence(persistence_dim_max=True)
    intervals = st.persistence_intervals_in_dimension(1)
    return np.asarray(intervals, dtype=float).reshape(-1, 2)


def betti1_at(intervals: np.ndarray, t: float) -> int:
    """Number of H1 classes alive at filtration value t."""
    if intervals.size == 0:
        return 0
    births, deaths = intervals[:, 0], intervals[:, 1]
    return int(np.sum((births <= t) & (t < deaths)))


def h1_features(
    A: np.ndarray,
    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
    top_k: int | None = 8,
    weight_floor: float = 0.0,
    symmetrized: bool = False,
) -> dict:
    """Full attention-matrix -> H1 feature dict.

    Set symmetrized=True when A is already a symmetric weight matrix (tests).
    """
    W = A.astype(float) if symmetrized else symmetrize(A)
    W = sparsify(W, top_k=top_k, weight_floor=weight_floor)
    intervals = h1_intervals(build_simplex_tree(W))
    finite = intervals[np.isfinite(intervals[:, 1])] if intervals.size else intervals
    persist = (finite[:, 1] - finite[:, 0]) if finite.size else np.array([])
    feats = {
        "n_cycles": int(intervals.shape[0]),
        "total_persistence": float(persist.sum()) if persist.size else 0.0,
        "max_persistence": float(persist.max()) if persist.size else 0.0,
    }
    for t in thresholds:
        feats[f"betti1_t{t}"] = betti1_at(intervals, t)
    return feats
