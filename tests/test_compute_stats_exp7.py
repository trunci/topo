import numpy as np
import pandas as pd

from src.compute_stats_exp7 import _verdict


def _synth(n, mode, seed=0):
    """Build a features frame where correctness depends on a chosen signal.

    mode='topology': is_correct driven by topo features (confidence uninformative)
    mode='confidence': is_correct driven by confidence only (topology noise)
    mode='balanced_noise': is_correct random (nothing predicts)
    """
    rng = np.random.default_rng(seed)
    conf = rng.normal(size=n)
    tmean = rng.normal(size=n)
    tmax = rng.normal(size=n)
    tfrac = rng.normal(size=n)
    if mode == "topology":
        logit = 2.0 * tmean + 1.0 * tmax
    elif mode == "confidence":
        logit = 2.5 * conf
    else:
        logit = np.zeros(n)
    p = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({
        "is_correct": y,
        "confidence_margin": conf,
        "topo_mean_persist": tmean,
        "topo_max_persist": tmax,
        "topo_frac_nontrivial": tfrac,
        "ctrl_attn_distance": rng.normal(size=n),
        "ctrl_offdiag_mass": rng.normal(size=n),
        "ctrl_attn_entropy": rng.normal(size=n),
        "hop": rng.integers(1, 4, size=n),
    })


def test_green_when_topology_drives_correctness():
    v = _verdict(_synth(400, "topology", seed=1))
    assert v["topology_beats_chance"] is True
    assert v["verdict"] == "GREEN"


def test_red_or_partial_when_topology_is_noise():
    # topology is pure noise; correctness driven by confidence -> not GREEN
    v = _verdict(_synth(400, "confidence", seed=2))
    assert v["verdict"] != "GREEN"


def test_underpowered_when_base_rate_extreme():
    df = _synth(200, "balanced_noise", seed=3)
    df["is_correct"] = 1  # everyone correct -> no variance
    v = _verdict(df)
    assert v["underpowered"] is True
    assert v["verdict"] == "UNDERPOWERED"
