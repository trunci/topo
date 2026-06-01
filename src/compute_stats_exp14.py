"""Statistics for Experiment 14 -> results/exp14_stats.json.

Same analysis framework as Exp 13, plus the key competitive test:
does topology add beyond first-order controls (the dimension where
TOHA is benchmarked but the field doesn't test internally)?

TOHA baseline for comparison: AUROC 0.71 on HotpotQA/Mistral-7B.
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
CTRL = ["ctrl_attn_distance", "ctrl_offdiag_mass", "ctrl_attn_entropy",
        "ctrl_kl_from_uniform"]
CONF = ["confidence_margin"]
TOHA_AUROC = 0.71    # TOHA on HotpotQA/Mistral-7B (Bazarova et al., ACL 2026)
KL_PROBE_AUROC = 0.79  # KL-divergence probe (arXiv 2605.05025), avg over models


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


def _summary(folds):
    if not folds:
        return {"mean_auc": None, "std_auc": None, "n_folds": 0}
    return {"mean_auc": float(np.mean(folds)), "std_auc": float(np.std(folds)),
            "n_folds": len(folds)}


def _verdict_block(df):
    acc = float(df["is_correct"].mean())
    n = int(len(df))
    if acc > 0.85 or acc < 0.15:
        return {"n": n, "acc": acc, "verdict": "UNDERPOWERED",
                "auc": {}, "topo_beyond_ctrl": {}}

    y = df["is_correct"].to_numpy(int)
    X = lambda c: df[c].to_numpy(float)

    m0   = _cv_auc(X(CONF), y)
    m1   = _cv_auc(X(CONF + TOPO), y)
    mt   = _cv_auc(X(TOPO), y)
    mc   = _cv_auc(X(CONF + CTRL), y)
    mf   = _cv_auc(X(CONF + CTRL + TOPO), y)

    def _delta(a, b):
        if not a or not b or len(a) != len(b): return None, None
        d = np.array(a) - np.array(b)
        med = float(np.median(d))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = float(wilcoxon(a, b, alternative="greater").pvalue)
        except ValueError:
            p = None
        return med, p

    d01_med, d01_p = _delta(m1, m0)
    dfc_med, dfc_p = _delta(mf, mc)

    topo_beats = bool(mt and np.mean(mt) > 0.5)
    adds_conf  = bool(d01_p is not None and d01_p < ALPHA and d01_med and d01_med > 0)
    adds_ctrl  = bool(dfc_p is not None and dfc_p < ALPHA and dfc_med and dfc_med > 0)

    if not topo_beats:       verdict = "RED"
    elif adds_conf:          verdict = "GREEN"
    else:                    verdict = "PARTIAL"

    return {
        "n": n, "acc": acc, "verdict": verdict,
        "auc": {
            "confidence_only":            _summary(m0),
            "confidence_plus_topology":   _summary(m1),
            "topology_only":              _summary(mt),
            "confidence_plus_controls":   _summary(mc),
            "confidence_plus_ctrl_topo":  _summary(mf),
        },
        "delta_topo_vs_conf": {"median": d01_med, "wilcoxon_p": d01_p,
                               "adds": adds_conf},
        "delta_topo_vs_ctrl": {"median": dfc_med, "wilcoxon_p": dfc_p,
                               "adds": adds_ctrl},
        "topology_beats_chance": topo_beats,
    }


def compute_stats(features_path="results/exp14_features.parquet",
                  out_path="results/exp14_stats.json"):
    df = pd.read_parquet(features_path)

    stats = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "dataset": "HotpotQA bridge (validation)",
        "n_items": int(len(df)),
        "overall_accuracy": float(df["is_correct"].mean()),
        "acc_by_level": {k: float(v) for k, v in
                         df.groupby("level")["is_correct"].mean().items()},
        "mean_seq_len": float(df["seq_len"].mean()),
        "toha_reference_auroc": TOHA_AUROC,
        "all_items": _verdict_block(df),
    }

    # level slices
    for level in ["easy", "medium", "hard"]:
        sl = df[df["level"] == level]
        if len(sl) >= 20:
            stats[f"{level}_items"] = _verdict_block(sl)

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    v = stats["all_items"]
    print(f"\n[exp14] ── RESULTS ─────────────────────────────────────────")
    print(f"  n={stats['n_items']}  acc={stats['overall_accuracy']:.3f}  "
          f"mean_seq_len={stats['mean_seq_len']:.0f}")
    print(f"  acc by level: {stats['acc_by_level']}")
    if v["verdict"] != "UNDERPOWERED":
        topo_auc = v["auc"]["topology_only"]["mean_auc"]
        conf_auc = v["auc"]["confidence_only"]["mean_auc"]
        ctrl_auc = v["auc"]["confidence_plus_controls"]["mean_auc"]
        print(f"\n  ── Overall AUCs ──")
        print(f"  confidence only:   {conf_auc:.3f}")
        print(f"  topology only:     {topo_auc:.3f}")
        print(f"  controls only:     {ctrl_auc:.3f}")
        print(f"  TOHA reference:    {TOHA_AUROC:.3f}  (same model, paper-reported)")
        print(f"  KL-probe ref:      {KL_PROBE_AUROC:.3f}  (arXiv 2605.05025)")
        print(f"\n  topology vs TOHA:  {'BEATS' if topo_auc > TOHA_AUROC else 'BELOW'} "
              f"({topo_auc:.3f} vs {TOHA_AUROC:.3f})")
        print(f"  topology vs KL:    {'BEATS' if topo_auc > KL_PROBE_AUROC else 'BELOW'} "
              f"({topo_auc:.3f} vs {KL_PROBE_AUROC:.3f})")
        print(f"\n  topo adds beyond conf: {v['delta_topo_vs_conf']['adds']} "
              f"(Δ={v['delta_topo_vs_conf']['median']:.3f}, "
              f"p={v['delta_topo_vs_conf']['wilcoxon_p']:.4f})")
        print(f"  topo adds beyond ctrl: {v['delta_topo_vs_ctrl']['adds']} "
              f"(Δ={v['delta_topo_vs_ctrl']['median']:.3f}, "
              f"p={v['delta_topo_vs_ctrl']['wilcoxon_p']:.4f})")
        print(f"\n  ── Regime-conditional (by level) ──")
        for lv in ["easy", "medium", "hard"]:
            if f"{lv}_items" in stats:
                lv_v = stats[f"{lv}_items"]
                if lv_v["verdict"] != "UNDERPOWERED":
                    ta = lv_v["auc"]["topology_only"]["mean_auc"]
                    ca = lv_v["auc"]["confidence_only"]["mean_auc"]
                    print(f"  {lv:8s}: acc={lv_v['acc']:.3f}  topo={ta:.3f}  "
                          f"conf={ca:.3f}  verdict={lv_v['verdict']}")
                else:
                    print(f"  {lv:8s}: UNDERPOWERED (acc={lv_v['acc']:.3f})")
        print(f"  verdict: {v['verdict']}")
    else:
        print(f"  UNDERPOWERED (acc={v['acc']:.3f})")
    return stats


if __name__ == "__main__":
    compute_stats()
