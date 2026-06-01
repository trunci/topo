"""Statistics for Exp 13 model-generalization run (Qwen2.5-1.5B-Instruct).

Saves results/exp13_1p5b_stats.json with per-hop and combined verdicts,
plus the key additional test: does topology add beyond first-order controls?
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

from src.compute_stats_exp13 import _verdict

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
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(sc.transform(X[tr]), y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        if len(np.unique(y[te])) >= 2:
            aucs.append(float(roc_auc_score(y[te], p)))
    return aucs


def _topo_beyond_ctrl(df):
    """Extra test: does topology add beyond first-order controls?"""
    y = df["is_correct"].to_numpy(int)
    mc = _cv_auc(df[CONF + CTRL].to_numpy(float), y)
    mf = _cv_auc(df[CONF + CTRL + TOPO].to_numpy(float), y)
    if not mc or not mf or len(mc) != len(mf):
        return {"delta_median": None, "wilcoxon_p": None, "adds": False}
    d = np.array(mf) - np.array(mc)
    delta_med = float(np.median(d))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(wilcoxon(mf, mc, alternative="greater").pvalue)
    except ValueError:
        p = None
    return {
        "ctrl_auc": float(np.mean(mc)),
        "full_auc": float(np.mean(mf)),
        "delta_median": delta_med,
        "wilcoxon_p": p,
        "adds": bool(p is not None and p < ALPHA and delta_med > 0),
    }


def compute_stats(features_path="results/exp13_1p5b_features.parquet",
                  out_path="results/exp13_1p5b_stats.json"):
    df = pd.read_parquet(features_path)

    slices = {"hop3": df[df["hop"] == 3],
              "hop4": df[df["hop"] == 4],
              "hop5": df[df["hop"] == 5]}

    stats = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "hops_tested": [3, 4, 5],
        "overall_accuracy": float(df["is_correct"].mean()),
        "acc_by_hop": {str(int(h)): float(g["is_correct"].mean())
                       for h, g in df.groupby("hop")},
        "all_items": _verdict(df),
        **{f"{k}_only": _verdict(v) for k, v in slices.items() if len(v) > 0},
        "topo_beyond_controls_all": _topo_beyond_ctrl(df),
    }

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[1p5b] overall acc={stats['overall_accuracy']:.3f}")
    for k, v in slices.items():
        vd = stats[f"{k}_only"]
        print(f"  {k}: acc={v['is_correct'].mean():.3f} verdict={vd['verdict']} "
              f"topo={vd['auc']['topology_only']['mean_auc']:.3f} "
              f"conf={vd['auc']['confidence_only']['mean_auc']:.3f}")
    tb = stats["topo_beyond_controls_all"]
    print(f"  topo beyond controls: delta={tb['delta_median']:.3f} "
          f"p={tb['wilcoxon_p']:.4f} adds={tb['adds']}")
    return stats


if __name__ == "__main__":
    compute_stats()
