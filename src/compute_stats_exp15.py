"""Statistics for Experiment 15 (gold-only HotpotQA) -> results/exp15_stats.json.

Primary analysis: identical to Exp 14 (_verdict_block — pooled features,
5-fold CV, same pre-registered GREEN/PARTIAL/RED rule), so the gold-only vs
full-context comparison is apples-to-apples. Plus the length diagnostics from
Exp 14b, to verify the manipulation actually tightened length variance.

Secondary (exploratory, possible because Exp 15 persists per-head features):
fold-internal head selection — within each CV train split, rank heads by
single-head AUC of per-head H1 persistence and keep the top K_HEADS; fit on
those columns only; evaluate on the held-out fold. No leakage: selection sees
only training labels. This is the cheapest TOHA-flavoured featurization test.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.compute_stats_exp14 import (_verdict_block, _summary,
                                     N_SPLITS, SEED, TOHA_AUROC)
from src.compute_stats_exp14b import length_correlations

K_HEADS = 16


def head_selected_auc(ph: np.ndarray, y: np.ndarray, k_heads: int = K_HEADS,
                      seed: int = SEED) -> list[float]:
    """CV AUC with per-fold head selection (top-k single-head train AUC)."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(ph, y):
        head_scores = []
        for j in range(ph.shape[1]):
            col = ph[tr, j]
            if np.std(col) == 0:
                head_scores.append(0.5)
                continue
            a = roc_auc_score(y[tr], col)
            head_scores.append(max(a, 1 - a))   # direction-agnostic
        top = np.argsort(head_scores)[::-1][:k_heads]
        sc = StandardScaler().fit(ph[tr][:, top])
        clf = LogisticRegression(max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(sc.transform(ph[tr][:, top]), y[tr])
            p = clf.predict_proba(sc.transform(ph[te][:, top]))[:, 1]
        if len(np.unique(y[te])) >= 2:
            aucs.append(float(roc_auc_score(y[te], p)))
    return aucs


def compute_stats(features_path="results/exp15_features.parquet",
                  exp14_stats_path="results/exp14_stats.json",
                  out_path="results/exp15_stats.json"):
    df = pd.read_parquet(features_path)

    stats = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "dataset": "HotpotQA bridge (validation), GOLD-ONLY contexts",
        "n_items": int(len(df)),
        "overall_accuracy": float(df["is_correct"].mean()),
        "mean_seq_len": float(df["seq_len"].mean()),
        "seq_len_stats": {
            "min": int(df["seq_len"].min()), "median": float(df["seq_len"].median()),
            "max": int(df["seq_len"].max()), "std": float(df["seq_len"].std()),
        },
        "toha_reference_auroc": TOHA_AUROC,
        "length_correlations": length_correlations(df),
        "all_items": _verdict_block(df),
    }

    # secondary: fold-internal head selection over per-head H1 persistence
    if "ph_tot" in df.columns:
        ph = np.stack(df["ph_tot"].to_numpy())
        y = df["is_correct"].to_numpy(int)
        folds = head_selected_auc(ph, y)
        stats["head_selected_topology"] = {**_summary(folds), "k_heads": K_HEADS,
                                           "note": "exploratory, fold-internal selection"}

    # side-by-side with Exp 14 (full contexts, same items/model/pipeline)
    try:
        e14 = json.load(open(exp14_stats_path))
        stats["exp14_comparison"] = {
            "exp14_acc": e14["overall_accuracy"],
            "exp14_mean_seq_len": e14["mean_seq_len"],
            "exp14_topology_auc": e14["all_items"]["auc"]["topology_only"]["mean_auc"],
            "exp14_confidence_auc": e14["all_items"]["auc"]["confidence_only"]["mean_auc"],
            "exp14_verdict": e14["all_items"]["verdict"],
        }
    except FileNotFoundError:
        pass

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    v = stats["all_items"]
    print("\n[exp15] ── RESULTS (gold-only contexts) ──────────────────────")
    print(f"  n={stats['n_items']}  acc={stats['overall_accuracy']:.3f}  "
          f"seq_len {stats['seq_len_stats']['min']}–{stats['seq_len_stats']['max']} "
          f"(std {stats['seq_len_stats']['std']:.0f})")
    lc = stats["length_correlations"]
    print(f"  max |rho(topo, len)| = {lc['max_topo_len_rho']:.3f}  "
          f"length_dominated={lc['length_dominated']}")
    if v["verdict"] != "UNDERPOWERED":
        print(f"  topology only:   {v['auc']['topology_only']['mean_auc']:.3f}")
        print(f"  confidence only: {v['auc']['confidence_only']['mean_auc']:.3f}")
        print(f"  topo adds beyond conf: {v['delta_topo_vs_conf']['adds']} "
              f"(Δ={v['delta_topo_vs_conf']['median']:.3f}, "
              f"p={v['delta_topo_vs_conf']['wilcoxon_p']:.4f})")
        print(f"  topo adds beyond ctrl: {v['delta_topo_vs_ctrl']['adds']}")
    if "head_selected_topology" in stats:
        print(f"  head-selected topo (exploratory): "
              f"{stats['head_selected_topology']['mean_auc']:.3f}")
    if "exp14_comparison" in stats:
        c = stats["exp14_comparison"]
        print(f"  vs Exp 14 full-context: acc {c['exp14_acc']:.3f}, "
              f"topo {c['exp14_topology_auc']:.3f}, conf {c['exp14_confidence_auc']:.3f}")
    print(f"  verdict: {v['verdict']}")
    return stats


if __name__ == "__main__":
    compute_stats()
