"""Authoritative statistics for Experiment 10 (IOI attribution) -> exp10_stats.json.

Part A: per saliency, mean ROC-AUC + p@1 at recovering the ground-truth name-mover
  edge (END->IO); one-sided Wilcoxon (i) >0.5 chance and (ii) paired > magnitude.
Part B: damage = margin(unmasked) - margin(cond) (positive = ablation hurt IO behavior);
  topology-salient (cycle, sheaf) vs random and vs magnitude.
Verdicts pre-registered in the design spec. No hand-typed numbers downstream.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ALPHA = 0.05
TOPO = ["cycle_participation", "sheaf_discord"]


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


def _part_a(df):
    piv = df.pivot_table(index=["prompt", "layer", "head"], columns="saliency", values="auc")
    mag = piv["magnitude"].to_numpy(float)
    res = {}
    for sal in ["magnitude"] + TOPO:
        col = piv[sal].to_numpy(float)
        res[sal] = {"mean_auc": float(np.nanmean(col)),
                    "mean_p_at_1": float(df[df.saliency == sal]["p_at_1"].mean()),
                    "p_vs_chance": _w(col, mu=0.5)}
    for sal in TOPO:
        res[sal]["p_vs_magnitude"] = _w(piv[sal].to_numpy(float), mag)

    def bc(s): return (res[s]["p_vs_chance"] is not None and res[s]["p_vs_chance"] < ALPHA
                       and res[s]["mean_auc"] > 0.5)
    def bm(s): return (res[s]["p_vs_magnitude"] is not None and res[s]["p_vs_magnitude"] < ALPHA)
    verdict = ("GREEN" if any(bc(s) and bm(s) for s in TOPO)
               else "PARTIAL" if any(bc(s) for s in TOPO) else "RED")
    return {"per_saliency": res, "n_observations": int(len(piv)), "verdict": verdict}


def _part_b(df):
    piv = df.pivot_table(index="prompt", columns="condition", values="io_margin")
    dmg = {c: (piv["unmasked"] - piv[c]).to_numpy(float)
           for c in ["cycle", "sheaf", "magnitude", "random"]}
    res = {"median_damage": {c: float(np.median(v)) for c, v in dmg.items()}, "n": int(len(piv))}
    for sal in ["cycle", "sheaf"]:
        res[f"{sal}_vs_random_p"] = _w(dmg[sal], dmg["random"])
        res[f"{sal}_vs_magnitude_p"] = _w(dmg[sal], dmg["magnitude"])
        res[f"{sal}_damage_positive"] = bool(np.median(dmg[sal]) > 0)

    def green(sal):
        pr = res[f"{sal}_vs_random_p"]
        return res[f"{sal}_damage_positive"] and pr is not None and pr < ALPHA
    any_green = green("cycle") or green("sheaf")
    any_pos = res["cycle_damage_positive"] or res["sheaf_damage_positive"]
    res["verdict"] = "GREEN" if any_green else ("PARTIAL" if any_pos else "RED")
    return res


def compute_stats(attr_path="results/exp10_attribution.parquet",
                  abl_path="results/exp10_ablation.parquet",
                  out_path="results/exp10_stats.json"):
    a = _part_a(pd.read_parquet(attr_path))
    b = _part_b(pd.read_parquet(abl_path))
    headline = ("GREEN" if a["verdict"] == "GREEN" and b["verdict"] == "GREEN"
                else "RED" if a["verdict"] == "RED" and b["verdict"] == "RED" else "PARTIAL")
    stats = {"alpha": ALPHA, "task": "IOI", "model": "gpt2",
             "part_a_attribution": a, "part_b_causal": b, "headline_verdict": headline}
    json.dump(stats, open(out_path, "w"), indent=2)
    return stats


if __name__ == "__main__":
    compute_stats()
