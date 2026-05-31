import numpy as np
import pandas as pd

from src.compute_stats_exp8 import _frame1, _frame2


def _frame(n, mode, seed=0):
    rng = np.random.default_rng(seed)
    hop = rng.integers(1, 4, size=n).astype(float)
    conf = rng.normal(size=n)
    # sheaf features: in 'sheaf' mode they encode hop / correctness
    base = rng.normal(size=(n, 5))
    if mode in ("sheaf_disc", "sheaf_pred"):
        base[:, 3] += 0.8 * hop  # discord tracks hop
    df = pd.DataFrame({
        "hop": hop,
        "is_correct": (rng.uniform(size=n) < 0.5).astype(int),
        "confidence_margin": conf,
        "sheaf_t1_fiedler_mean": base[:, 0],
        "sheaf_harmonic_dim_mean": base[:, 1],
        "sheaf_spectral_gap_mean": base[:, 2],
        "sheaf_discord_mean": base[:, 3],
        "sheaf_discord_max": base[:, 4],
        "h1_mean_persist": rng.normal(size=n),
        "h1_max_persist": rng.normal(size=n),
        "h1_frac_nontrivial": rng.normal(size=n),
        "ctrl_attn_distance": rng.normal(size=n),
        "ctrl_offdiag_mass": rng.normal(size=n),
        "ctrl_attn_entropy": rng.normal(size=n),
    })
    if mode == "sheaf_pred":
        logit = 2.0 * base[:, 3]
        df["is_correct"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def test_frame1_green_when_sheaf_predicts_hop():
    v = _frame1(_frame(400, "sheaf_disc", seed=1))
    assert v["delta_r2"] >= 0.02
    assert v["verdict"] == "GREEN"


def test_frame1_red_when_sheaf_is_noise():
    v = _frame1(_frame(400, "noise", seed=2))
    assert v["verdict"] == "RED"


def test_frame2_green_when_sheaf_drives_correctness():
    v = _frame2(_frame(400, "sheaf_pred", seed=3))
    assert v["sheaf_beats_chance"] is True
    assert v["verdict"] == "GREEN"


def test_frame2_not_green_when_sheaf_noise():
    v = _frame2(_frame(400, "noise", seed=4))
    assert v["verdict"] != "GREEN"
