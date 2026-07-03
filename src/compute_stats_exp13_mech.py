"""Persist the Exp 13 mechanistic-decomposition scalars -> results/exp13_mech_stats.json.

Recomputes, from the three seed parquets, the four numbers that WRITEUP.md §
"Mechanistic decomposition" and write_paper.py previously inherited as inline
constants (0.989 / 0.991 / 0.925 / 0.000):

* hop-5 pooled (3 seeds, n=180) topology-only 5-fold CV AUC
* hop-5 pooled controls-only 5-fold CV AUC
* Pearson r(topo_mean_persist, ctrl_attn_entropy) on the hop-5 pooled slice
* delta-AUC of controls+topology minus controls (median over folds, one-sided
  Wilcoxon), i.e. what topology adds beyond first-order controls at 0.5B

Uses the identical probe machinery as compute_stats_exp13 (same folds, seed,
scaler, classifier) so the persisted values are the same analysis, now
machine-checked instead of hand-inherited.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

from src.compute_stats_exp13 import CTRL, TOPO, _fit_block, _summary

SEED_PARQUETS = [
    "results/exp13_features.parquet",     # seed 0
    "results/exp13_s1_features.parquet",  # seed 1
    "results/exp13_s2_features.parquet",  # seed 2
]


def compute_stats(out_path="results/exp13_mech_stats.json"):
    df = pd.concat([pd.read_parquet(p) for p in SEED_PARQUETS], ignore_index=True)
    hop5 = df[df["hop"] == 5].reset_index(drop=True)

    m_topo = _fit_block(hop5, TOPO)
    m_ctrl = _fit_block(hop5, CTRL)
    m_full = _fit_block(hop5, CTRL + TOPO)

    d = np.array(m_full) - np.array(m_ctrl)
    delta_med = float(np.median(d))
    delta_p = None
    if np.any(d != 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                delta_p = float(wilcoxon(m_full, m_ctrl,
                                         alternative="greater").pvalue)
            except ValueError:
                delta_p = None

    r, r_p = pearsonr(hop5["topo_mean_persist"], hop5["ctrl_attn_entropy"])

    stats = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "slice": "hop5, pooled seeds 0-2",
        "n_items": int(len(hop5)),
        "base_rate_correct": float(hop5["is_correct"].mean()),
        "auc": {
            "topology_only": _summary(m_topo),
            "controls_only": _summary(m_ctrl),
            "controls_plus_topology": _summary(m_full),
        },
        "delta_auc_topo_beyond_controls": {
            "median": delta_med, "wilcoxon_p": delta_p,
        },
        "pearson_topo_mean_persist_vs_attn_entropy": {
            "r": float(r), "p": float(r_p),
        },
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    s = compute_stats()
    print(f"n={s['n_items']}  topo AUC {s['auc']['topology_only']['mean_auc']:.3f}  "
          f"ctrl AUC {s['auc']['controls_only']['mean_auc']:.3f}  "
          f"r(persist,entropy) {s['pearson_topo_mean_persist_vs_attn_entropy']['r']:.3f}  "
          f"dAUC {s['delta_auc_topo_beyond_controls']['median']:.3f} "
          f"(p={s['delta_auc_topo_beyond_controls']['wilcoxon_p']})")
