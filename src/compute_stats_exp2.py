"""Compute Experiment-2 statistics from the parquet into a JSON (source of truth).

Pre-registered (see spec):
  1. discrimination: Spearman(induction, H1 persistence) > 0 and p < 0.05
  2. separation: top-k induction heads have higher H1 persistence (Mann-Whitney)
  3. beats baseline: |rho(persistence,induction)| >= |rho(distance,induction)| - 0.05
Verdict: GREEN if 1&2 and not dominated by distance; YELLOW if 1&2 but distance
as good/better; RED if 1 or 2 fails.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

TOP_K = 10
ALPHA = 0.05
BASELINE_SLACK = 0.05


def compute_stats(parquet_path: str, out_path: str) -> dict:
    df = pd.read_parquet(parquet_path)
    ind = df["induction_score"].to_numpy()
    h1 = df["h1_persistence"].to_numpy()
    dist = df["attn_distance"].to_numpy()

    rho, p = spearmanr(ind, h1)
    drho, dp = spearmanr(ind, dist)

    order = np.argsort(ind)[::-1]
    top_idx = order[:TOP_K]
    rest_idx = order[TOP_K:]
    top_h1 = h1[top_idx]
    rest_h1 = h1[rest_idx]
    if len(rest_h1) > 0 and len(top_h1) > 0:
        u_stat, mw_p = mannwhitneyu(top_h1, rest_h1, alternative="greater")
    else:
        u_stat, mw_p = float("nan"), float("nan")

    crit1 = bool(rho > 0 and p < ALPHA)
    crit2 = bool(mw_p < ALPHA)
    not_dominated = bool(abs(rho) >= abs(drho) - BASELINE_SLACK)

    if crit1 and crit2 and not_dominated:
        verdict = "GREEN: topology corresponds to induction heads beyond a trivial distance baseline."
    elif crit1 and crit2:
        verdict = "YELLOW: topology correlates with induction but a distance baseline does as well or better."
    else:
        verdict = "RED: no significant topology-induction correspondence."

    stats = {
        "n_heads": int(len(df)),
        "top_k": TOP_K,
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(mw_p),
        "top_mean_h1": float(np.mean(top_h1)),
        "rest_mean_h1": float(np.mean(rest_h1)),
        "baseline_distance_rho": float(drho),
        "baseline_distance_p": float(dp),
        "topology_not_dominated_by_distance": not_dominated,
        "top_induction_heads": [
            f"L{int(df.iloc[i]['layer'])}H{int(df.iloc[i]['head'])} "
            f"ind={ind[i]:.3f} h1={h1[i]:.3f}" for i in top_idx
        ],
        "verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    compute_stats("results/exp2.parquet", "results/exp2_stats.json")
    print("WROTE results/exp2_stats.json")
