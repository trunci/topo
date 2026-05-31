"""Authoritative statistics for Experiment 9 -> results/exp9_stats.json.

Part A (attribution): per saliency, mean ROC-AUC at recovering ground-truth induction
  copy edges, tested (one-sided Wilcoxon) (i) > 0.5 chance and (ii) paired > magnitude.
Part B (causal): damage(cond) = loss(cond) - loss(unmasked), paired; topology-salient
  (cycle, sheaf) vs random and vs magnitude (one-sided Wilcoxon).

Verdicts pre-registered in the design spec. No hand-typed numbers downstream.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ALPHA = 0.05
TOPO_SALIENCIES = ["cycle_participation", "sheaf_discord"]


def _w_greater(a, b=None, mu=0.0):
    """One-sided Wilcoxon: median(a-b) > 0 (or median(a)-mu > 0). p or None."""
    a = np.asarray(a, dtype=float)
    if b is None:
        d = a - mu
    else:
        d = a - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0 or not np.any(d != 0):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(wilcoxon(d, alternative="greater").pvalue)
        except ValueError:
            return None


def _part_a(attr_df):
    out = {}
    piv = attr_df.pivot_table(index=["model", "seq", "layer", "head"],
                              columns="saliency", values="auc")
    mag = piv["magnitude"].to_numpy(dtype=float)
    res = {}
    for sal in ["magnitude"] + TOPO_SALIENCIES:
        col = piv[sal].to_numpy(dtype=float)
        res[sal] = {
            "mean_auc": float(np.nanmean(col)),
            "mean_p_at_k": float(attr_df[attr_df.saliency == sal]["p_at_k"].mean()),
            "p_vs_chance": _w_greater(col, mu=0.5),
        }
    for sal in TOPO_SALIENCIES:
        col = piv[sal].to_numpy(dtype=float)
        res[sal]["p_vs_magnitude"] = _w_greater(col, mag)
    out["per_saliency"] = res
    out["n_observations"] = int(len(piv))

    # verdict: best topo saliency
    def beats_chance(s): return (res[s]["p_vs_chance"] is not None
                                 and res[s]["p_vs_chance"] < ALPHA
                                 and res[s]["mean_auc"] > 0.5)
    def beats_mag(s): return (res[s]["p_vs_magnitude"] is not None
                              and res[s]["p_vs_magnitude"] < ALPHA)
    any_chance = any(beats_chance(s) for s in TOPO_SALIENCIES)
    any_mag = any(beats_chance(s) and beats_mag(s) for s in TOPO_SALIENCIES)
    out["verdict"] = "GREEN" if any_mag else ("PARTIAL" if any_chance else "RED")
    return out


def _part_b(abl_df):
    piv = abl_df.pivot_table(index=["model", "seq"], columns="condition",
                             values="second_copy_loss")
    dmg = {c: (piv[c] - piv["unmasked"]).to_numpy(dtype=float)
           for c in ["cycle", "sheaf", "magnitude", "random"]}
    res = {"median_damage": {c: float(np.median(v)) for c, v in dmg.items()},
           "n": int(len(piv))}
    for sal in ["cycle", "sheaf"]:
        res[f"{sal}_vs_random_p"] = _w_greater(dmg[sal], dmg["random"])
        res[f"{sal}_vs_magnitude_p"] = _w_greater(dmg[sal], dmg["magnitude"])
        res[f"{sal}_damage_positive"] = bool(np.median(dmg[sal]) > 0)

    def green(sal):
        pr = res[f"{sal}_vs_random_p"]
        return (res[f"{sal}_damage_positive"] and pr is not None and pr < ALPHA)
    any_green = green("cycle") or green("sheaf")
    any_pos = res["cycle_damage_positive"] or res["sheaf_damage_positive"]
    res["verdict"] = "GREEN" if any_green else ("PARTIAL" if any_pos else "RED")
    return res


def compute_stats(attr_path="results/exp9_attribution.parquet",
                  abl_path="results/exp9_ablation.parquet",
                  out_path="results/exp9_stats.json"):
    a = _part_a(pd.read_parquet(attr_path))
    b = _part_b(pd.read_parquet(abl_path))
    headline = ("GREEN" if a["verdict"] == "GREEN" and b["verdict"] == "GREEN"
                else ("RED" if a["verdict"] == "RED" and b["verdict"] == "RED"
                      else "PARTIAL"))
    stats = {"alpha": ALPHA, "part_a_attribution": a, "part_b_causal": b,
             "headline_verdict": headline}
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    compute_stats()
