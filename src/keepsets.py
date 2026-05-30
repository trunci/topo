"""Budget-matched baseline keep-sets.

Each function takes a symmetrized (already sparsified) weight matrix W and an
integer budget k, and returns a set of exactly min(k, n_candidate_edges) kept
edges as frozenset{i, j}. Candidate edges are the upper-triangle entries with
W > 0. These baselines are matched to DMT's per-head edge count.
"""
from __future__ import annotations

import random

import numpy as np


def _candidate_edges(W: np.ndarray):
    n = W.shape[0]
    iu = np.triu_indices(n, k=1)
    return [(int(i), int(j)) for i, j in zip(*iu) if W[i, j] > 0.0]


def magnitude_keep(W: np.ndarray, k: int) -> set:
    """Keep the k highest-weight edges."""
    edges = _candidate_edges(W)
    edges.sort(key=lambda e: W[e[0], e[1]], reverse=True)
    return {frozenset(e) for e in edges[:k]}


def window_keep(W: np.ndarray, k: int) -> set:
    """Keep the k edges nearest the diagonal (smallest |i - j|).

    Ties broken by higher weight, then by (i, j) order.
    """
    edges = _candidate_edges(W)
    edges.sort(key=lambda e: (abs(e[0] - e[1]), -W[e[0], e[1]], e))
    return {frozenset(e) for e in edges[:k]}


def random_keep(W: np.ndarray, k: int, seed: int = 0) -> set:
    """Keep k edges sampled uniformly without replacement (deterministic by seed)."""
    edges = _candidate_edges(W)
    rng = random.Random(seed)
    if k >= len(edges):
        return {frozenset(e) for e in edges}
    return {frozenset(e) for e in rng.sample(edges, k)}
