import numpy as np

from src.compute_stats_exp15 import head_selected_auc


def _perhead(n, n_heads, signal_heads, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    y = (s + 0.3 * rng.normal(size=n) > 0).astype(int)
    ph = rng.normal(size=(n, n_heads))
    for j in signal_heads:
        ph[:, j] = s + 0.2 * rng.normal(size=n)
    return ph, y


def test_head_selection_finds_signal_heads():
    ph, y = _perhead(300, 64, signal_heads=[5, 17, 40], seed=1)
    aucs = head_selected_auc(ph, y, k_heads=8)
    assert np.mean(aucs) > 0.8


def test_head_selection_stays_near_chance_on_noise():
    ph, y = _perhead(300, 64, signal_heads=[], seed=2)
    aucs = head_selected_auc(ph, y, k_heads=8)
    # no leakage: selecting noise heads on train must not score on test
    assert np.mean(aucs) < 0.62


def test_handles_constant_heads():
    ph, y = _perhead(120, 16, signal_heads=[3], seed=3)
    ph[:, 0] = 1.0   # zero-variance head must not crash roc_auc_score
    aucs = head_selected_auc(ph, y, k_heads=4)
    assert len(aucs) == 5
