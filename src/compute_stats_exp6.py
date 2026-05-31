"""Compute Experiment 6 statistics from the ablation parquet.

Single source of truth: results/exp6_ablation.parquet. This module reads it,
computes the pre-registered per-model statistics, and writes
results/exp6_stats.json. No numbers are hand-typed; FINDINGS_exp6.md is generated
from the JSON downstream.

Pre-registered per-model verdict (see the design doc):
* GREEN        -- cycle damage positive AND cycle > random (one-sided Wilcoxon,
                  p < 0.05): topology-selected edges are causally functional.
* RED          -- cycle damage positive but not > random: edges matter, but
                  topology selection is no better than random.
* INCONCLUSIVE -- cycle damage NOT positive: the ablation/readout is still too
                  weak to interpret (should not happen with the directed
                  edge-exact instrument, but reported honestly if it does).
cycle_vs_magnitude is reported as a specificity sub-question, not part of the gate.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ALPHA = 0.05


def _model_stats(sub_df):
    """Per-model damage stats from rows of one model.

    sub_df has columns seq, condition, second_copy_loss, k_ablated.
    Damage(cond) = loss(cond) - loss(unmasked), paired across sequences.
    """
    wide = sub_df.pivot_table(index="seq", columns="condition",
                              values="second_copy_loss")
    damage = {c: (wide[c] - wide["unmasked"]).values
              for c in ["cycle", "magnitude", "random"]}
    out = {
        "n_sequences": int(len(wide)),
        "k_ablated_mean": float(sub_df.groupby("seq")["k_ablated"].first().mean()),
        "median_damage": {c: float(np.median(damage[c])) for c in damage},
        "mean_damage": {c: float(np.mean(damage[c])) for c in damage},
    }
    for ctrl in ["magnitude", "random"]:
        d = damage["cycle"] - damage[ctrl]
        try:
            _, p = wilcoxon(d, alternative="greater")
        except ValueError:  # all-zero differences
            p = float("nan")
        out[f"cycle_vs_{ctrl}"] = {"p_value": float(p),
                                   "median_diff": float(np.median(d))}

    cyc_pos = out["median_damage"]["cycle"] > 0
    beats_random = (not np.isnan(out["cycle_vs_random"]["p_value"])
                    and out["cycle_vs_random"]["p_value"] < ALPHA)
    out["cycle_damage_positive"] = bool(cyc_pos)
    out["beats_random"] = bool(beats_random)
    if not cyc_pos:
        out["verdict"] = "INCONCLUSIVE"
    elif beats_random:
        out["verdict"] = "GREEN"
    else:
        out["verdict"] = "RED"
    return out


def compute_stats(ablation_path="results/exp6_ablation.parquet",
                  out_path="results/exp6_stats.json", seq_len=25, gap_tol=2):
    df = pd.read_parquet(ablation_path)
    models = {}
    for model_name, sub in df.groupby("model"):
        models[str(model_name)] = _model_stats(sub)
    stats = {
        "seq_len": seq_len,
        "gap_tolerance": gap_tol,
        "alpha": ALPHA,
        "models": models,
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    compute_stats()
