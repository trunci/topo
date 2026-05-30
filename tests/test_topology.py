import numpy as np
from src.topology import (
    symmetrize, sparsify, build_simplex_tree, h1_intervals, betti1_at, h1_features,
)


def _weight_matrix(n, edges):
    W = np.zeros((n, n), dtype=float)
    for i, j in edges:
        W[i, j] = 1.0
        W[j, i] = 1.0
    return W


def test_square_has_one_cycle():
    W = _weight_matrix(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    st = build_simplex_tree(W)
    intervals = h1_intervals(st)
    assert betti1_at(intervals, 0.5) == 1


def test_filled_triangle_has_no_cycle():
    # All three edges present -> expansion(2) fills the 2-simplex -> no H1.
    W = _weight_matrix(3, [(0, 1), (1, 2), (2, 0)])
    st = build_simplex_tree(W)
    intervals = h1_intervals(st)
    assert betti1_at(intervals, 0.5) == 0


def test_two_disjoint_squares_have_two_cycles():
    W = _weight_matrix(8, [(0, 1), (1, 2), (2, 3), (3, 0),
                           (4, 5), (5, 6), (6, 7), (7, 4)])
    st = build_simplex_tree(W)
    intervals = h1_intervals(st)
    assert betti1_at(intervals, 0.5) == 2


def test_path_has_no_cycle():
    W = _weight_matrix(4, [(0, 1), (1, 2), (2, 3)])
    st = build_simplex_tree(W)
    intervals = h1_intervals(st)
    assert betti1_at(intervals, 0.5) == 0


def test_symmetrize_takes_elementwise_max():
    A = np.array([[0.0, 0.7], [0.2, 0.0]])
    W = symmetrize(A)
    assert W[0, 1] == 0.7 and W[1, 0] == 0.7


def test_sparsify_keeps_top_k_per_node_symmetric():
    A = np.array([
        [0.0, 0.9, 0.1, 0.05],
        [0.9, 0.0, 0.8, 0.2],
        [0.1, 0.8, 0.0, 0.7],
        [0.05, 0.2, 0.7, 0.0],
    ])
    W = sparsify(A, top_k=1)
    assert np.allclose(W, W.T)
    assert W[0, 1] > 0  # node 0's strongest edge survives


def test_h1_features_on_square_reports_one_cycle():
    W = _weight_matrix(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    feats = h1_features(W, thresholds=(0.5,), top_k=8)
    assert feats["n_cycles"] == 1
    assert feats["betti1_t0.5"] == 1
    assert feats["max_persistence"] >= 0.0
