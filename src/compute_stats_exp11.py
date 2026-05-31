"""Authoritative statistics for Experiment 11 (clean IOI causal readout) -> exp11_stats.json.

Readout = IO-token log-probability at END (Exp 10 used the IO-S margin, which the
name-mover edge confounds). damage = logprob(unmasked) - logprob(condition); positive
= the ablated edges were necessary for recalling the IO name.

Reported:
  * instrument check: median damage of `gt_io` (ablate EXACTLY the END->IO edge). The
    readout is sound iff this is clearly positive and beats random (one-sided Wilcoxon).
  * topology (cycle, sheaf) and magnitude vs random and vs each other.

Verdict (pre-registered):
  INSTRUMENT-BROKEN -- gt_io damage not > random (the readout still can't detect the
                       known-necessary edge; no further claim is licensed).
  Else (instrument sound):
    TOPO-CAUSAL  -- a topological condition's damage > random (p<alpha) AND > 0.
    MAG-ONLY     -- only magnitude beats random.
    NULL-CAUSAL  -- nothing (incl. magnitude) beats random.

No hand-typed numbers downstream.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ALPHA = 0.05
CONDS = ["gt_io", "cycle", "sheaf", "magnitude", "random"]


def _w(a, b=None, mu=0.0):
    a = np.asarray(a, float)
    d = (a - mu) if b is None else (a - np.asarray(b, float))
    d = d[~np.isnan(d)]
    if d.size == 0 or not np.any(d != 0):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(wilcoxon(d, alternative="greater").pvalue)
        except ValueError:
            return None


def compute_stats(abl_path="results/exp11_ablation.parquet",
                  out_path="results/exp11_stats.json"):
    df = pd.read_parquet(abl_path)
    piv = df.pivot_table(index="prompt", columns="condition", values="io_logprob")
    dmg = {c: (piv["unmasked"] - piv[c]).to_numpy(float) for c in CONDS}
    res = {
        "median_damage": {c: float(np.median(v)) for c, v in dmg.items()},
        "n": int(len(piv)),
    }
    # instrument check
    res["gt_io_damage_positive"] = bool(np.median(dmg["gt_io"]) > 0)
    res["gt_io_vs_random_p"] = _w(dmg["gt_io"], dmg["random"])
    instrument_sound = (res["gt_io_damage_positive"]
                        and res["gt_io_vs_random_p"] is not None
                        and res["gt_io_vs_random_p"] < ALPHA)
    res["instrument_sound"] = bool(instrument_sound)
    # condition comparisons
    for sal in ["cycle", "sheaf", "magnitude"]:
        res[f"{sal}_damage_positive"] = bool(np.median(dmg[sal]) > 0)
        res[f"{sal}_vs_random_p"] = _w(dmg[sal], dmg["random"])
    for sal in ["cycle", "sheaf"]:
        res[f"{sal}_vs_magnitude_p"] = _w(dmg[sal], dmg["magnitude"])

    def beats_random(sal):
        p = res[f"{sal}_vs_random_p"]
        return res[f"{sal}_damage_positive"] and p is not None and p < ALPHA

    if not instrument_sound:
        verdict = "INSTRUMENT-BROKEN"
    elif beats_random("cycle") or beats_random("sheaf"):
        verdict = "TOPO-CAUSAL"
    elif beats_random("magnitude"):
        verdict = "MAG-ONLY"
    else:
        verdict = "NULL-CAUSAL"

    stats = {"alpha": ALPHA, "task": "IOI", "model": "gpt2",
             "readout": "io_logprob_at_end", "part_b_causal_clean": res,
             "verdict": verdict}
    json.dump(stats, open(out_path, "w"), indent=2)
    return stats


if __name__ == "__main__":
    compute_stats()
