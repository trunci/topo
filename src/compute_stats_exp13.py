"""Authoritative statistics for Experiment 13 -> results/exp13_stats.json.

Single source of truth: results/exp13_features.parquet. Fits nested logistic-
regression models predicting is_correct, scored by stratified 5-fold CV ROC-AUC.

Slices reported:
* all_items     -- hop=4 + hop=5 combined
* hop4_only     -- hop=4 items
* hop5_only     -- hop=5 items

Same verdict rule as Exp 7:
* GREEN        -- M_topo AUC > 0.5 AND delta-AUC(M1 - M0) > 0 with paired
                  Wilcoxon p < 0.05
* PARTIAL      -- M_topo beats chance but does not add beyond confidence
* RED          -- M_topo does not beat chance
* UNDERPOWERED -- base rate outside [0.15, 0.85]

Pre-condition: at least one of hop4/hop5 should land in [0.35, 0.70]; if both
are UNDERPOWERED (base_rate > 0.85), the task was still too easy.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ALPHA = 0.05
N_SPLITS = 5
SEED = 0

TOPO = ["topo_mean_persist", "topo_max_persist", "topo_frac_nontrivial"]
CTRL = ["ctrl_attn_distance", "ctrl_offdiag_mass", "ctrl_attn_entropy"]
CONF = ["confidence_margin"]


def _cv_auc(X, y, seed=SEED):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(scaler.transform(X[tr]), y[tr])
            p = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        if len(np.unique(y[te])) < 2:
            continue
        aucs.append(float(roc_auc_score(y[te], p)))
    return aucs


def _fit_block(df, cols, seed=SEED):
    X = df[cols].to_numpy(dtype=float)
    y = df["is_correct"].to_numpy(dtype=int)
    return _cv_auc(X, y, seed=seed)


def _summary(folds):
    if not folds:
        return {"mean_auc": None, "std_auc": None, "n_folds": 0, "folds": []}
    return {"mean_auc": float(np.mean(folds)), "std_auc": float(np.std(folds)),
            "n_folds": len(folds), "folds": [float(x) for x in folds]}


def _verdict(slice_df):
    base_rate = float(slice_df["is_correct"].mean())
    n = int(len(slice_df))
    underpowered = (base_rate > 0.85 or base_rate < 0.15)

    if underpowered:
        empty = _summary([])
        return {
            "n_items": n,
            "base_rate_correct": base_rate,
            "majority_class_acc": float(max(base_rate, 1 - base_rate)),
            "underpowered": True,
            "auc": {
                "confidence_only": empty,
                "confidence_plus_topology": empty,
                "topology_only": empty,
                "confidence_plus_controls": empty,
            },
            "delta_auc_full_minus_baseline": {"median": None, "wilcoxon_p": None},
            "topology_beats_chance": False,
            "topology_adds_beyond_confidence": False,
            "verdict": "UNDERPOWERED",
        }

    m0 = _fit_block(slice_df, CONF)
    m1 = _fit_block(slice_df, CONF + TOPO)
    m_topo = _fit_block(slice_df, TOPO)
    m_ctrl = _fit_block(slice_df, CONF + CTRL)

    delta_p, delta_med = None, None
    if m0 and m1 and len(m0) == len(m1):
        d = np.array(m1) - np.array(m0)
        delta_med = float(np.median(d))
        if np.any(d != 0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    delta_p = float(wilcoxon(m1, m0, alternative="greater").pvalue)
                except ValueError:
                    delta_p = None

    topo_beats_chance = bool(m_topo and np.mean(m_topo) > 0.5)
    adds_beyond_conf = bool(delta_p is not None and delta_p < ALPHA
                            and delta_med is not None and delta_med > 0)

    if not topo_beats_chance:
        verdict = "RED"
    elif adds_beyond_conf:
        verdict = "GREEN"
    else:
        verdict = "PARTIAL"

    return {
        "n_items": n,
        "base_rate_correct": base_rate,
        "majority_class_acc": float(max(base_rate, 1 - base_rate)),
        "underpowered": underpowered,
        "auc": {
            "confidence_only": _summary(m0),
            "confidence_plus_topology": _summary(m1),
            "topology_only": _summary(m_topo),
            "confidence_plus_controls": _summary(m_ctrl),
        },
        "delta_auc_full_minus_baseline": {
            "median": delta_med, "wilcoxon_p": delta_p,
        },
        "topology_beats_chance": topo_beats_chance,
        "topology_adds_beyond_confidence": adds_beyond_conf,
        "verdict": verdict,
    }


def compute_stats(features_path="results/exp13_features.parquet",
                  out_path="results/exp13_stats.json"):
    df = pd.read_parquet(features_path)
    hop4 = df[df["hop"] == 4]
    hop5 = df[df["hop"] == 5]

    # pre-condition: flag if both hops still above the underpowered floor
    precondition_met = any(
        0.15 <= float(s["is_correct"].mean()) <= 0.85
        for s in [hop4, hop5] if len(s) > 0
    )

    stats = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "alpha": ALPHA,
        "n_splits": N_SPLITS,
        "hops_tested": [4, 5],
        "overall_accuracy": float(df["is_correct"].mean()),
        "acc_by_hop": {str(int(h)): float(g["is_correct"].mean())
                       for h, g in df.groupby("hop")},
        "precondition_met": precondition_met,
        "all_items": _verdict(df),
        "hop4_only": _verdict(hop4) if len(hop4) > 0 else None,
        "hop5_only": _verdict(hop5) if len(hop5) > 0 else None,
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    compute_stats()
