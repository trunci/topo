import numpy as np
from src.keepsets import magnitude_keep, random_keep, window_keep


def _wm(n, weights):
    """weights: dict (i,j)->w, symmetric."""
    W = np.zeros((n, n), dtype=float)
    for (i, j), w in weights.items():
        W[i, j] = w
        W[j, i] = w
    return W


def test_magnitude_keep_picks_top_k_by_weight():
    W = _wm(4, {(0, 1): 0.9, (1, 2): 0.8, (2, 3): 0.7, (0, 3): 0.1})
    keep = magnitude_keep(W, 2)
    assert keep == {frozenset({0, 1}), frozenset({1, 2})}


def test_magnitude_keep_caps_at_available_edges():
    W = _wm(4, {(0, 1): 0.9, (1, 2): 0.8})
    keep = magnitude_keep(W, 10)
    assert keep == {frozenset({0, 1}), frozenset({1, 2})}


def test_window_keep_picks_smallest_distance():
    W = _wm(4, {(0, 1): 0.1, (1, 2): 0.1, (2, 3): 0.1, (0, 3): 0.9})
    # |i-j|: (0,1)=1 (1,2)=1 (2,3)=1 (0,3)=3 -> top-3 by smallest distance excludes (0,3)
    keep = window_keep(W, 3)
    assert keep == {frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})}


def test_random_keep_is_deterministic_under_seed():
    W = _wm(5, {(0, 1): 0.5, (1, 2): 0.5, (2, 3): 0.5, (3, 4): 0.5, (0, 4): 0.5})
    a = random_keep(W, 3, seed=42)
    b = random_keep(W, 3, seed=42)
    assert a == b
    assert len(a) == 3


def test_random_keep_changes_with_seed():
    W = _wm(6, {(i, j): 0.5 for i in range(6) for j in range(i + 1, 6)})
    a = random_keep(W, 5, seed=1)
    b = random_keep(W, 5, seed=2)
    # extremely unlikely to be identical across seeds for this many candidates
    assert a != b


def test_all_return_exactly_k_edges():
    W = _wm(5, {(0, 1): 0.9, (1, 2): 0.8, (2, 3): 0.7, (3, 4): 0.6, (0, 4): 0.5})
    for fn in (magnitude_keep, window_keep):
        assert len(fn(W, 3)) == 3
    assert len(random_keep(W, 3, seed=0)) == 3
