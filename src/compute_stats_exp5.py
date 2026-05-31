"""Authoritative statistics for Experiment 5 -> results/exp5_stats.json.

Single source of truth. Computes:

* E2 (cycle inspection): per-head fraction of critical cycle edges with
  |gap - S| <= 2, for induction vs non-induction heads; one-sided Mann-Whitney
  (induction > non-induction). GREEN if p < 0.05, else RED. Raw gap histograms
  included for inspection.
* E1 (causal ablation): damage(cond) = loss(cond) - loss(unmasked), paired
  across sequences; one-sided Wilcoxon signed-rank (greater) for cycle vs
  magnitude and cycle vs random, with median differences. GREEN if cycle beats
  BOTH (p<0.05, positive median diff); PARTIAL if cycle beats random only; RED
  if cycle does not beat random. Sanity flags whether median cycle and
  magnitude damages are > 0.

The verdict rules are pre-registered (see the design spec) and fixed here.
No numbers are hand-typed elsewhere; the write-up reads this JSON.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats as ss

GAP_TOL = 2          # |gap - S| <= GAP_TOL counts as "at the induction gap"
ALPHA = 0.05


def _e2_head_fractions(cycles_df, seq_len):
    """Per-head fraction of cycle edges with |gap - S| <= GAP_TOL.

    A "head" here is a (layer, head) identity pooled across sequences. Returns
    (induction_fractions, noninduction_fractions) as lists of floats.
    """
    fracs = {0: [], 1: []}
    for is_ind in (1, 0):
        sub = cycles_df[cycles_df["is_induction"] == is_ind]
        for (_, _), g in sub.groupby(["layer", "head"]):
            gaps = g["gap"].values
            if len(gaps) == 0:
                continue
            at_gap = np.abs(gaps - seq_len) <= GAP_TOL
            fracs[is_ind].append(float(np.mean(at_gap)))
    return fracs[1], fracs[0]


def _gap_histogram(cycles_df, is_ind):
    sub = cycles_df[cycles_df["is_induction"] == is_ind]
    if len(sub) == 0:
        return {}
    vc = sub["gap"].value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in vc.items()}


def _e2(cycles_df, seq_len):
    ind_frac, nonind_frac = _e2_head_fractions(cycles_df, seq_len)
    result = {
        "metric": f"fraction of cycle edges with |gap - S| <= {GAP_TOL} (S={seq_len})",
        "n_induction_heads": len(ind_frac),
        "n_noninduction_heads": len(nonind_frac),
        "induction_fractions": [float(x) for x in ind_frac],
        "noninduction_fractions": [float(x) for x in nonind_frac],
        "induction_mean_fraction": float(np.mean(ind_frac)) if ind_frac else None,
        "noninduction_mean_fraction": (float(np.mean(nonind_frac))
                                       if nonind_frac else None),
        "gap_histogram_induction": _gap_histogram(cycles_df, 1),
        "gap_histogram_noninduction": _gap_histogram(cycles_df, 0),
    }
    # one-sided Mann-Whitney: induction fractions > non-induction fractions
    if ind_frac and nonind_frac and (set(ind_frac) | set(nonind_frac)) != {0.0} \
            and not (len(set(ind_frac)) == 1 and len(set(nonind_frac)) == 1
                     and ind_frac[0] == nonind_frac[0]):
        try:
            mw = ss.mannwhitneyu(ind_frac, nonind_frac, alternative="greater")
            p = float(mw.pvalue)
        except ValueError:
            p = None
    else:
        # identical or degenerate distributions -> no concentration difference
        p = None
    result["mann_whitney_p"] = p
    result["verdict"] = "GREEN" if (p is not None and p < ALPHA) else "RED"
    return result


def _paired_damage(abl_df):
    """Return dict cond -> array of damages aligned by seq, plus unmasked array."""
    pivot = abl_df.pivot_table(index="seq", columns="condition",
                               values="second_copy_loss")
    pivot = pivot.sort_index()
    unmasked = pivot["unmasked"].values
    damage = {c: (pivot[c].values - unmasked)
              for c in ("cycle", "random", "magnitude")}
    return damage


def _wilcoxon_greater(a, b):
    """One-sided Wilcoxon signed-rank: median(a - b) > 0. Returns (p, median_diff).

    Returns p=None if the test is undefined (all differences zero / too few).
    """
    diff = np.asarray(a) - np.asarray(b)
    median_diff = float(np.median(diff))
    nz = diff[diff != 0]
    if nz.size == 0:
        return None, median_diff
    try:
        res = ss.wilcoxon(a, b, alternative="greater")
        return float(res.pvalue), median_diff
    except ValueError:
        return None, median_diff


def _e1(abl_df):
    damage = _paired_damage(abl_df)
    med = {c: float(np.median(d)) for c, d in damage.items()}

    p_mag, mdiff_mag = _wilcoxon_greater(damage["cycle"], damage["magnitude"])
    p_rnd, mdiff_rnd = _wilcoxon_greater(damage["cycle"], damage["random"])

    beats_mag = (p_mag is not None and p_mag < ALPHA and mdiff_mag > 0)
    beats_rnd = (p_rnd is not None and p_rnd < ALPHA and mdiff_rnd > 0)

    if beats_mag and beats_rnd:
        verdict = "GREEN"
    elif beats_rnd:
        verdict = "PARTIAL"
    else:
        verdict = "RED"

    result = {
        "n_sequences": int(abl_df["seq"].nunique()),
        "median_damage": med,
        "cycle_vs_magnitude": {"p_value": p_mag, "median_diff": mdiff_mag},
        "cycle_vs_random": {"p_value": p_rnd, "median_diff": mdiff_rnd},
        "sanity": {
            "cycle_damage_positive": bool(med["cycle"] > 0),
            "magnitude_damage_positive": bool(med["magnitude"] > 0),
            "note": ("ablation/readout too weak to interpret if cycle and "
                     "magnitude damages are ~0"),
        },
        "verdict": verdict,
    }
    return result


def compute_stats(cycles_path="results/exp5_cycles.parquet",
                  ablation_path="results/exp5_ablation.parquet",
                  out_path="results/exp5_stats.json",
                  seq_len=25):
    cycles_df = pd.read_parquet(cycles_path)
    abl_df = pd.read_parquet(ablation_path)
    result = {
        "model": (str(abl_df["model"].iloc[0]) if len(abl_df) else "gpt2"),
        "seq_len": int(seq_len),
        "gap_tolerance": GAP_TOL,
        "alpha": ALPHA,
        "E2": _e2(cycles_df, seq_len),
        "E1": _e1(abl_df),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    res = compute_stats()
    print(json.dumps(res, indent=2))
