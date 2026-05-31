"""Experiment 3: residual-information test -> results/exp3_stats.json (truth).

Question: does H1 persistence add ANY predictive power for induction score beyond
first-order attention statistics (distance, off-diagonal mass, entropy)?

Pre-registered (see docs/.../experiment3-residual-info-design.md):
  baseline OLS: induction ~ attn_distance + offdiag_mass + attn_entropy
  full OLS    : baseline + h1_persistence
  delta_r2    = full_r2 - baseline_r2
  nested F-test (statsmodels anova_lm) for adding h1_persistence -> f_pvalue
  partial Spearman(h1, induction | controls) via rank residuals
Verdict: GREEN iff delta_r2 >= 0.02 AND f_pvalue < 0.05; RED otherwise.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from scipy.stats import spearmanr, rankdata

DELTA_R2_FLOOR = 0.02
ALPHA = 0.05
CONTROLS = ["attn_distance", "offdiag_mass", "attn_entropy"]


def _partial_spearman(x, y, controls):
    """Spearman partial correlation of x, y controlling for `controls` columns.

    Rank-transform everything, regress rank(x) and rank(y) on the (rank) controls
    with an intercept, then Pearson-correlate the residuals (== partial Spearman).
    """
    rx = rankdata(x)
    ry = rankdata(y)
    rc = np.column_stack([rankdata(c) for c in controls])
    Z = sm.add_constant(rc)
    res_x = sm.OLS(rx, Z).fit().resid
    res_y = sm.OLS(ry, Z).fit().resid
    rho, p = spearmanr(res_x, res_y)  # residuals are continuous; equivalent to pearson
    return float(rho), float(p)


def compute_stats(parquet_path: str, out_path: str) -> dict:
    df = pd.read_parquet(parquet_path)
    y = df["induction_score"].to_numpy()
    h1 = df["h1_persistence"].to_numpy()

    Xb = sm.add_constant(df[CONTROLS].to_numpy())
    Xf = sm.add_constant(df[CONTROLS + ["h1_persistence"]].to_numpy())

    base = sm.OLS(y, Xb).fit()
    full = sm.OLS(y, Xf).fit()

    baseline_r2 = float(base.rsquared)
    full_r2 = float(full.rsquared)
    delta_r2 = full_r2 - baseline_r2

    # Nested-model F-test for the added h1 term.
    aov = anova_lm(base, full)
    f_stat = float(aov["F"].iloc[-1])
    f_pvalue = float(aov["Pr(>F)"].iloc[-1])

    # h1 coefficient is the LAST predictor in the full model (after the const).
    h1_coef = float(full.params[-1])
    h1_coef_p = float(full.pvalues[-1])

    partial_rho, partial_p = _partial_spearman(
        h1, y, [df[c].to_numpy() for c in CONTROLS]
    )
    raw_rho, raw_p = spearmanr(h1, y)

    adds_power = bool(delta_r2 >= DELTA_R2_FLOOR and f_pvalue < ALPHA)
    if adds_power:
        verdict = ("GREEN: H1 persistence adds significant predictive power for "
                   "induction beyond first-order attention statistics.")
    else:
        verdict = ("RED: H1 persistence is redundant with first-order attention "
                   "statistics for predicting induction (no residual predictive "
                   "power).")

    stats = {
        "n_heads": int(len(df)),
        "controls": CONTROLS,
        "delta_r2_floor": DELTA_R2_FLOOR,
        "alpha": ALPHA,
        "baseline_r2": baseline_r2,
        "baseline_adj_r2": float(base.rsquared_adj),
        "full_r2": full_r2,
        "full_adj_r2": float(full.rsquared_adj),
        "delta_r2": float(delta_r2),
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
        "h1_coef": h1_coef,
        "h1_coef_p": h1_coef_p,
        "partial_spearman_rho": partial_rho,
        "partial_spearman_p": partial_p,
        "raw_spearman_rho": float(raw_rho),
        "raw_spearman_p": float(raw_p),
        "h1_adds_predictive_power": adds_power,
        "verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    compute_stats("results/exp3.parquet", "results/exp3_stats.json")
    print("WROTE results/exp3_stats.json")
