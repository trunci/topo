"""Exp 14b: length-confound diagnostic on the Exp 14 features -> exp14b_stats.json.

Exp 14 (Result Q) found H1 topology at chance (AUC 0.494) on Mistral-7B/HotpotQA.
This re-analysis of the SAME on-disk features asks WHY, testing the hypothesis
that the pooled global H1 features are dominated by sequence length — HotpotQA
prompts span ~200-3800 tokens — on a dataset where length carries no information
about correctness. (Exp 13's synthetic items were template-constant in length,
which is exactly the condition this diagnostic says made the GREEN possible.)

Pre-registered rules:
- LENGTH-DOMINATED if max |Spearman(topo_i, seq_len)| > 0.8 AND
  |Spearman(seq_len, is_correct)| < 0.1.
- RESCUED if length-residualized topology adds beyond confidence
  (paired Wilcoxon p < 0.05, median delta > 0); otherwise NULL-STANDS.

No new model runs — pure re-analysis of results/exp14_features.parquet.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from src.compute_stats_exp14 import _cv_auc, _summary, TOPO, CTRL, CONF

ALPHA = 0.05
RHO_DOMINATED = 0.8   # a feature this length-correlated is a length meter
RHO_LABEL_MAX = 0.1   # ...on a dataset where length says ~nothing about the label


def length_correlations(df: pd.DataFrame) -> dict:
    feats = {}
    for c in TOPO + CTRL + CONF:
        if c not in df.columns:
            continue
        rho, p = spearmanr(df[c], df["seq_len"])
        feats[c] = {"rho": float(rho), "p": float(p)}
    rho_y, p_y = spearmanr(df["seq_len"], df["is_correct"])
    max_topo_rho = max(abs(feats[c]["rho"]) for c in TOPO)
    return {
        "features": feats,
        "len_vs_correct": {"rho": float(rho_y), "p": float(p_y)},
        "max_topo_len_rho": float(max_topo_rho),
        "length_dominated": bool(max_topo_rho > RHO_DOMINATED
                                 and abs(rho_y) < RHO_LABEL_MAX),
    }


def residualize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """OLS-residualize cols on [1, z(len), z(len^2)] — removes the smooth
    length component while keeping whatever varies independently of it."""
    L = np.column_stack([df["seq_len"].to_numpy(float),
                         df["seq_len"].to_numpy(float) ** 2])
    L = (L - L.mean(axis=0)) / L.std(axis=0)
    X = np.column_stack([np.ones(len(df)), L])
    out = {}
    for c in cols:
        v = df[c].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        out[c] = v - X @ beta
    return pd.DataFrame(out, index=df.index)


def _delta(a, b):
    if not a or not b or len(a) != len(b):
        return None, None
    med = float(np.median(np.array(a) - np.array(b)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(wilcoxon(a, b, alternative="greater").pvalue)
    except ValueError:
        p = None
    return med, p


def analyze(df: pd.DataFrame) -> dict:
    corr = length_correlations(df)
    R = residualize(df, TOPO)
    y = df["is_correct"].to_numpy(int)

    X_raw = df[TOPO].to_numpy(float)
    X_res = R[TOPO].to_numpy(float)
    X_conf = df[CONF].to_numpy(float)
    X_cr = np.column_stack([X_conf, X_res])

    m_raw = _cv_auc(X_raw, y)
    m_res = _cv_auc(X_res, y)
    m_conf = _cv_auc(X_conf, y)
    m_cr = _cv_auc(X_cr, y)

    d_med, d_p = _delta(m_cr, m_conf)
    adds = bool(d_p is not None and d_p < ALPHA and d_med and d_med > 0)
    beats_chance = bool(m_res and np.mean(m_res) > 0.5)

    parts = []
    parts.append("LENGTH-DOMINATED" if corr["length_dominated"]
                 else "NOT-LENGTH-DOMINATED")
    parts.append("RESCUED" if adds else "NULL-STANDS")
    return {
        "n_items": int(len(df)),
        "length_stats": {
            "min": int(df["seq_len"].min()), "median": float(df["seq_len"].median()),
            "max": int(df["seq_len"].max()), "std": float(df["seq_len"].std()),
        },
        "length_correlations": corr,
        "auc": {
            "topology_raw":         _summary(m_raw),
            "topology_resid":       _summary(m_res),
            "confidence_only":      _summary(m_conf),
            "conf_plus_resid_topo": _summary(m_cr),
        },
        "delta_resid_topo_vs_conf": {"median": d_med, "wilcoxon_p": d_p,
                                     "adds": adds},
        "resid_topo_beats_chance": beats_chance,
        "verdict": ", ".join(parts),
    }


def compute_stats(features_path="results/exp14_features.parquet",
                  out_path="results/exp14b_stats.json"):
    df = pd.read_parquet(features_path)
    stats = {"based_on": features_path, **analyze(df)}
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    c = stats["length_correlations"]
    print("\n[exp14b] ── LENGTH-CONFOUND DIAGNOSTIC ────────────────────────")
    print(f"  n={stats['n_items']}  seq_len {stats['length_stats']['min']}–"
          f"{stats['length_stats']['max']} (std {stats['length_stats']['std']:.0f})")
    print("  Spearman vs seq_len:")
    for name, r in c["features"].items():
        print(f"    {name:24s} rho={r['rho']:+.3f}")
    print(f"    {'is_correct':24s} rho={c['len_vs_correct']['rho']:+.3f}")
    print(f"  length_dominated: {c['length_dominated']}")
    a = stats["auc"]
    print(f"\n  topology raw AUC:          {a['topology_raw']['mean_auc']:.3f}")
    print(f"  topology residualized AUC: {a['topology_resid']['mean_auc']:.3f}")
    print(f"  confidence AUC:            {a['confidence_only']['mean_auc']:.3f}")
    print(f"  conf + resid topo AUC:     {a['conf_plus_resid_topo']['mean_auc']:.3f}")
    d = stats["delta_resid_topo_vs_conf"]
    print(f"  resid topo adds beyond conf: {d['adds']} "
          f"(Δ={d['median']:.3f}, p={d['wilcoxon_p']:.4f})")
    print(f"  verdict: {stats['verdict']}")
    return stats


if __name__ == "__main__":
    compute_stats()
