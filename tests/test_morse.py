import numpy as np
from src.morse import morse_keep


def _wm(n, edges):
    W = np.zeros((n, n), dtype=float)
    for i, j in edges:
        W[i, j] = 1.0
        W[j, i] = 1.0
    return W


def test_four_cycle_keeps_all_edges():
    # An unfilled 4-cycle has a 1-cycle (H1); all 4 edges are needed to preserve it.
    W = _wm(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    keep = morse_keep(W, top_k=8, symmetrized=True)
    assert keep == {frozenset({0, 1}), frozenset({1, 2}),
                    frozenset({2, 3}), frozenset({3, 0})}


def test_filled_triangle_keeps_two_edges():
    # A filled triangle (3 mutually-connected nodes) is contractible: a spanning
    # tree of 2 edges preserves connectivity; the third edge is redundant.
    W = _wm(3, [(0, 1), (1, 2), (2, 0)])
    keep = morse_keep(W, top_k=8, symmetrized=True)
    assert len(keep) == 2


def test_two_disjoint_edges_kept():
    W = _wm(4, [(0, 1), (2, 3)])
    keep = morse_keep(W, top_k=8, symmetrized=True)
    assert keep == {frozenset({0, 1}), frozenset({2, 3})}


def test_path_keeps_all_edges():
    W = _wm(4, [(0, 1), (1, 2), (2, 3)])
    keep = morse_keep(W, top_k=8, symmetrized=True)
    assert keep == {frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})}


def test_min_budget_frac_tops_up_keepset():
    # A filled triangle naturally keeps 2 of 3 edges (~0.67). With min_budget_frac=1.0
    # the keep-set must be topped up to all 3 edges.
    W = _wm(3, [(0, 1), (1, 2), (2, 0)])
    keep = morse_keep(W, top_k=8, symmetrized=True, min_budget_frac=1.0)
    assert len(keep) == 3


def test_returns_frozensets():
    W = _wm(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    keep = morse_keep(W, top_k=8, symmetrized=True)
    assert all(isinstance(e, frozenset) and len(e) == 2 for e in keep)
