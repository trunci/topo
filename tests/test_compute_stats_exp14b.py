import numpy as np
import pandas as pd

from src.compute_stats_exp14b import (length_correlations, residualize,
                                      analyze, TOPO)


def _frame(n, regime, seed=0):
    """regime controls what the topology features actually measure:
    'pure-length'  — topo is a deterministic function of seq_len, label random
    'hidden-signal'— topo = big length term + small real failure signal
    """
    rng = np.random.default_rng(seed)
    seq_len = rng.integers(200, 3800, size=n).astype(float)
    z = (seq_len - seq_len.mean()) / seq_len.std()
    signal = rng.normal(size=n)
    y = (signal + 0.3 * rng.normal(size=n) > 0).astype(int)

    if regime == "pure-length":
        base = z
    else:  # hidden-signal: length dominates variance, signal is the residual
        base = 5.0 * z + signal

    df = pd.DataFrame({
        "is_correct": y if regime == "hidden-signal"
                      else rng.integers(0, 2, size=n),
        "confidence_margin": rng.normal(size=n),
        "seq_len": seq_len,
    })
    for i, c in enumerate(TOPO):
        df[c] = base + 0.01 * rng.normal(size=n)
    return df


def test_length_correlations_flag_pure_length_proxy():
    df = _frame(300, "pure-length", seed=1)
    corr = length_correlations(df)
    for c in TOPO:
        assert abs(corr["features"][c]["rho"]) > 0.95
    assert abs(corr["len_vs_correct"]["rho"]) < 0.15
    assert corr["length_dominated"] is True


def test_residualize_removes_length_component():
    df = _frame(300, "pure-length", seed=2)
    R = residualize(df, TOPO)
    from scipy.stats import spearmanr
    for c in TOPO:
        rho, _ = spearmanr(R[c], df["seq_len"])
        assert abs(rho) < 0.2


def test_residualization_recovers_hidden_signal():
    df = _frame(400, "hidden-signal", seed=3)
    res = analyze(df)
    raw = res["auc"]["topology_raw"]["mean_auc"]
    rec = res["auc"]["topology_resid"]["mean_auc"]
    # length swamps the raw feature; residualizing should reveal the signal
    assert rec > raw + 0.1
    assert rec > 0.8


def test_pure_length_stays_null_after_residualization():
    df = _frame(400, "pure-length", seed=4)
    res = analyze(df)
    assert res["auc"]["topology_resid"]["mean_auc"] < 0.62
    assert res["delta_resid_topo_vs_conf"]["adds"] is False
    assert "NULL-STANDS" in res["verdict"]
