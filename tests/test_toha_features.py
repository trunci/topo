"""Unit tests for the Exp 16 MTop-Div implementation (no model needed)."""
import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree

from src.toha_features import mtop_div


def _brute_reference(resp_rows, p):
    """Same quantity via scipy MST on the contracted graph (weights kept
    strictly positive so csgraph's zero-means-no-edge quirk can't bite)."""
    R = resp_rows.shape[0]
    d_to_p = 1.0 - resp_rows[:, :p].max(axis=1)
    block = resp_rows[:, p:p + R]
    d_rr = 1.0 - np.maximum(block, block.T)
    D = np.zeros((R + 1, R + 1))
    D[0, 1:] = d_to_p
    D[1:, 0] = d_to_p
    D[1:, 1:] = d_rr
    np.fill_diagonal(D, 0.0)
    shift = 10.0  # constant shift; MST over complete graph has exactly R edges
    D_pos = np.triu(D + shift, k=1)
    total = minimum_spanning_tree(D_pos).sum() - shift * R
    return total / R


def test_matches_scipy_reference_on_random_graphs():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p, R = int(rng.integers(3, 30)), int(rng.integers(1, 12))
        rows = rng.random((R, p + R))
        # causal mask: row i attends only to columns <= p + i
        for i in range(R):
            rows[i, p + i + 1:] = 0.0
        assert mtop_div(rows, p) == pytest.approx(_brute_reference(rows, p),
                                                  abs=1e-9)


def test_grounded_response_scores_low():
    """Response tokens attending hard onto the prompt attach cheaply."""
    p, R = 10, 5
    grounded = np.full((R, p + R), 1e-4)
    grounded[:, 0] = 0.95            # every response token locks onto prompt
    detached = np.full((R, p + R), 1e-4)
    detached[:, 0] = 0.05            # barely touches the prompt
    assert mtop_div(grounded, p) < mtop_div(detached, p)


def test_empty_response():
    assert mtop_div(np.zeros((0, 7)), 7) == 0.0


def test_normalization_by_response_length():
    """Two identical detached tokens ~ same per-token cost as one."""
    p = 8
    one = np.full((1, p + 1), 1e-4)
    two = np.full((2, p + 2), 1e-4)
    assert mtop_div(two, p) == pytest.approx(mtop_div(one, p), rel=1e-3)
