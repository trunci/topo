"""Phase A / A1 -- carrier decomposition for the sheaf-flow signal (Exp 9a).

THE DECISIVE GATE for the follow-up study
(docs/proposals/2026-05-31-sheaf-flow-followup-proposal.md): Exp 8 Frame 1 found a
sheaf "block" adds beyond H1 at predicting reasoning hop-count. But the block mixed
*shape* features (Fiedler value = scalar graph-Laplacian connectivity; harmonic dim,
which was degenerate) with the only true *flow* feature (discord). This module asks:
is the added signal carried by FLOW (discord) or by SHAPE (Fiedler) in disguise?

Runs on the EXISTING results/exp8_features.parquet -- no new model run needed.

Definitions (predicting hop):
  SHAPE baseline = H1 persistence + first-order controls + Fiedler (mean,max).
  FLOW           = discord (mean,max) + sheaf spectral gap.
  (harmonic_dim is dropped: zero variance in Exp 8, pre-committed.)

Tests:
  * delta_r2(FLOW | SHAPE)        -- does flow add beyond ALL shape features?
  * delta_r2(Fiedler | H1+ctrl)   -- how much of Exp 8's original block was just Fiedler?
  * partial Spearman(discord_mean, hop | SHAPE).

Verdict (pre-registered):
  CONFIRMED-FLOW    -- flow adds beyond shape (delta_r2 >= 0.02 AND nested-F p < 0.05).
  SHAPE-IN-DISGUISE -- flow does NOT add beyond shape, but Fiedler added beyond H1.
  NULL              -- neither survives.

Single source of truth: writes results/exp9a_stats.json. No hand-typed numbers.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats as ss
import statsmodels.api as sm

ALPHA = 0.05
DELTA_R2_FLOOR = 0.02

H1 = ["h1_mean_persist", "h1_max_persist", "h1_frac_nontrivial"]
CTRL = ["ctrl_attn_distance", "ctrl_offdiag_mass", "ctrl_attn_entropy"]
FIEDLER = ["sheaf_t1_fiedler_mean", "sheaf_t1_fiedler_max"]
FLOW = ["sheaf_discord_mean", "sheaf_discord_max", "sheaf_spectral_gap_mean"]


def _drop_zero_var(df, cols):
    """Keep only columns with nonzero, finite variance (pre-committed degeneracy guard)."""
    kept, dropped = [], []
    for c in cols:
        v = df[c].var()
        if c in df.columns and np.isfinite(v) and v > 1e-12:
            kept.append(c)
        else:
            dropped.append(c)
    return kept, dropped


def _ols(df, cols, target="hop"):
    X = sm.add_constant(df[cols].to_numpy(dtype=float))
    y = df[target].to_numpy(dtype=float)
    return sm.OLS(y, X).fit()


def _nested(df, base_cols, add_cols, target="hop"):
    base_cols, _ = _drop_zero_var(df, base_cols)
    add_cols, dropped_add = _drop_zero_var(df, add_cols)
    full_cols = base_cols + add_cols
    m_base = _ols(df, base_cols, target)
    m_full = _ols(df, full_cols, target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_stat, f_p, _ = m_full.compare_f_test(m_base)
    return {
        "baseline_cols": base_cols,
        "added_cols": add_cols,
        "dropped_zero_var": dropped_add,
        "baseline_r2": float(m_base.rsquared),
        "full_r2": float(m_full.rsquared),
        "delta_r2": float(m_full.rsquared - m_base.rsquared),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_p),
        "adds": bool((m_full.rsquared - m_base.rsquared) >= DELTA_R2_FLOOR and f_p < ALPHA),
    }


def _partial_spearman(df, feat, target, controls):
    """Spearman(feat, target) after linearly residualizing both on controls."""
    controls, _ = _drop_zero_var(df, controls)
    Xc = sm.add_constant(df[controls].to_numpy(dtype=float))
    rf = df[feat].to_numpy(dtype=float) - sm.OLS(df[feat].to_numpy(dtype=float), Xc).fit().fittedvalues
    rt = df[target].to_numpy(dtype=float) - sm.OLS(df[target].to_numpy(dtype=float), Xc).fit().fittedvalues
    rho, p = ss.spearmanr(rf, rt)
    return {"rho": float(rho), "p": float(p)}


def compute_stats(features_path="results/exp8_features.parquet",
                  out_path="results/exp9a_stats.json", target="hop"):
    df = pd.read_parquet(features_path)

    flow_vs_shape = _nested(df, H1 + CTRL + FIEDLER, FLOW, target)
    fiedler_vs_h1 = _nested(df, H1 + CTRL, FIEDLER, target)
    discord_partial = _partial_spearman(df, "sheaf_discord_mean", target,
                                        H1 + CTRL + FIEDLER)

    flow_adds = flow_vs_shape["adds"]
    fiedler_adds = fiedler_vs_h1["adds"]
    if flow_adds:
        verdict = "CONFIRMED-FLOW"
    elif fiedler_adds:
        verdict = "SHAPE-IN-DISGUISE"
    else:
        verdict = "NULL"

    stats = {
        "phase": "A1 carrier decomposition",
        "source": features_path,
        "target": target,
        "n_items": int(len(df)),
        "alpha": ALPHA,
        "delta_r2_floor": DELTA_R2_FLOOR,
        "flow_beyond_shape": flow_vs_shape,
        "fiedler_beyond_h1": fiedler_vs_h1,
        "discord_partial_spearman": discord_partial,
        "verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    compute_stats()
