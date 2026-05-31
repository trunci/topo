import numpy as np
import pandas as pd

from src.compute_stats_exp9a import _nested, _drop_zero_var


def _frame(n, carrier, seed=0):
    """carrier in {'flow','shape','none'} controls what predicts hop."""
    rng = np.random.default_rng(seed)
    hop = rng.integers(1, 4, size=n).astype(float)
    fiedler = rng.normal(size=n)
    discord = rng.normal(size=n)
    if carrier == "flow":
        discord += 1.2 * hop          # discord carries the signal
    elif carrier == "shape":
        fiedler += 1.2 * hop          # fiedler (shape) carries the signal
    return pd.DataFrame({
        "hop": hop,
        "h1_mean_persist": rng.normal(size=n),
        "h1_max_persist": rng.normal(size=n),
        "h1_frac_nontrivial": rng.normal(size=n),
        "ctrl_attn_distance": rng.normal(size=n),
        "ctrl_offdiag_mass": rng.normal(size=n),
        "ctrl_attn_entropy": rng.normal(size=n),
        "sheaf_t1_fiedler_mean": fiedler,
        "sheaf_t1_fiedler_max": fiedler + 0.01 * rng.normal(size=n),
        "sheaf_discord_mean": discord,
        "sheaf_discord_max": discord + 0.01 * rng.normal(size=n),
        "sheaf_spectral_gap_mean": rng.normal(size=n),
        "sheaf_harmonic_dim_mean": np.full(n, 4.0),  # degenerate, must be dropped
    })


def test_flow_adds_when_discord_carries():
    df = _frame(400, "flow", seed=1)
    res = _nested(df, ["h1_mean_persist", "h1_max_persist", "h1_frac_nontrivial",
                       "ctrl_attn_distance", "ctrl_offdiag_mass", "ctrl_attn_entropy",
                       "sheaf_t1_fiedler_mean", "sheaf_t1_fiedler_max"],
                  ["sheaf_discord_mean", "sheaf_discord_max", "sheaf_spectral_gap_mean"])
    assert res["adds"] is True
    assert res["delta_r2"] >= 0.02


def test_flow_does_not_add_when_shape_carries():
    df = _frame(400, "shape", seed=2)
    # fiedler is in the baseline, so adding flow should NOT help
    res = _nested(df, ["h1_mean_persist", "h1_max_persist", "h1_frac_nontrivial",
                       "ctrl_attn_distance", "ctrl_offdiag_mass", "ctrl_attn_entropy",
                       "sheaf_t1_fiedler_mean", "sheaf_t1_fiedler_max"],
                  ["sheaf_discord_mean", "sheaf_discord_max", "sheaf_spectral_gap_mean"])
    assert res["adds"] is False


def test_drop_zero_var_removes_degenerate():
    df = _frame(50, "none", seed=3)
    kept, dropped = _drop_zero_var(df, ["sheaf_discord_mean", "sheaf_harmonic_dim_mean"])
    assert "sheaf_harmonic_dim_mean" in dropped
    assert "sheaf_discord_mean" in kept
