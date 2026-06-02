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


def edges_from_weights(W: np.ndarray) -> tuple[int, np.ndarray]:
    """Extract the upper-triangular nonzero edges of a (sparsified) weight matrix.

    Returns (n_nodes, edges) where edges is an (m, 3) array of [i, j, weight].
    This is the compact representation passed to workers instead of the dense
    matrix — after top-k sparsification it is ~170x smaller than W.
    """
    n = W.shape[0]
    iu = np.triu_indices(n, k=1)
    w = W[iu]
    nz = w > 0.0
    edges = np.column_stack([iu[0][nz], iu[1][nz], w[nz]]).astype(float)
    return n, edges


def build_simplex_tree_from_edges(n: int, edges: np.ndarray) -> gudhi.SimplexTree:
    """Build the flag complex from a precomputed (n_nodes, edge-list).

    Same filtration f(edge) = 1 - weight as build_simplex_tree, expanded to
    dimension 2. Splitting prep (sparsify -> edges, in the main process) from
    this step lets workers receive a tiny edge list instead of a dense matrix.
    """
    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i], filtration=0.0)
    for i, j, w in edges:
        st.insert([int(i), int(j)], filtration=float(1.0 - w))
    st.expansion(2)
    return st


def build_simplex_tree(W: np.ndarray) -> gudhi.SimplexTree:
    """Build the flag (clique) complex with filtration f(edge) = 1 - weight.

    Vertices enter at 0; strong edges (weight near 1) enter early (near 0).
    Expanded to dimension 2 so triangles can fill in and kill 1-cycles.
    """
    n, edges = edges_from_weights(W)
    return build_simplex_tree_from_edges(n, edges)


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


def _features_from_intervals(intervals: np.ndarray,
                             thresholds: tuple[float, ...]) -> dict:
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


def h1_features_from_edges(
    n: int, edges: np.ndarray,
    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> dict:
    """H1 feature dict from a precomputed (n_nodes, edge-list).

    Identical output to h1_features, but takes the compact edge representation
    so the dense weight matrix never has to be passed across a process boundary.
    """
    intervals = h1_intervals(build_simplex_tree_from_edges(n, edges))
    return _features_from_intervals(intervals, thresholds)


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
    return _features_from_intervals(intervals, thresholds)
