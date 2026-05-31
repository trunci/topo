"""Authoritative statistics for Experiment 8 -> results/exp8_stats.json.

Single source of truth: results/exp8_features.parquet. Two pre-registered frames:

Frame 1 (Spike discrimination): does each sheaf feature discriminate reasoning
  hop-count, AND add beyond H1 persistence? Nested OLS predicting hop from
  [H1 + first-order controls] (baseline) vs [+ sheaf features] (full); report
  delta-R2 and nested-F. GREEN if delta-R2 >= 0.02 AND F p < 0.05.

Frame 2 (failure prediction): nested logistic predicting is_correct, 5-fold CV AUC.
  M0 = confidence; M1 = confidence + H1; M2 = confidence + sheaf;
  M3 = confidence + H1 + sheaf. GREEN if sheaf beats chance AND M2 > M0 (paired
  Wilcoxon across folds, p < 0.05); PARTIAL if beats chance but no add; RED else.

No numbers hand-typed downstream; FINDINGS_exp8 renders from this JSON.
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
import statsmodels.api as sm

ALPHA = 0.05
N_SPLITS = 5
SEED = 0
DELTA_R2_FLOOR = 0.02

SHEAF = ["sheaf_t1_fiedler_mean", "sheaf_harmonic_dim_mean", "sheaf_spectral_gap_mean",
         "sheaf_discord_mean", "sheaf_discord_max"]
H1 = ["h1_mean_persist", "h1_max_persist", "h1_frac_nontrivial"]
CTRL = ["ctrl_attn_distance", "ctrl_offdiag_mass", "ctrl_attn_entropy"]
CONF = ["confidence_margin"]


# ---- Frame 1: discrimination (nested OLS predicting hop) -------------------

def _ols_r2(df, cols):
    X = sm.add_constant(df[cols].to_numpy(dtype=float))
    y = df["hop"].to_numpy(dtype=float)
    model = sm.OLS(y, X).fit()
    return float(model.rsquared), model


def _frame1(df):
    base_cols = H1 + CTRL
    full_cols = base_cols + SHEAF
    r2_base, m_base = _ols_r2(df, base_cols)
    r2_full, m_full = _ols_r2(df, full_cols)
    # nested F-test for the block of sheaf features
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ftest = m_full.compare_f_test(m_base)
    f_stat, f_p, _ = ftest
    delta = r2_full - r2_base
    green = (delta >= DELTA_R2_FLOOR) and (f_p < ALPHA)
    return {
        "baseline_cols": base_cols,
        "baseline_r2": r2_base,
        "full_r2": r2_full,
        "delta_r2": float(delta),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_p),
        "verdict": "GREEN" if green else "RED",
    }


# ---- Frame 2: failure prediction (nested CV-AUC) --------------------------

def _cv_auc(X, y, seed=SEED):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(scaler.transform(X[tr]), y[tr])
            p = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        aucs.append(float(roc_auc_score(y[te], p)))
    return aucs


def _auc_for(df, cols):
    X = df[cols].to_numpy(dtype=float)
    y = df["is_correct"].to_numpy(dtype=int)
    return _cv_auc(X, y)


def _summary(folds):
    if not folds:
        return {"mean_auc": None, "std_auc": None, "n_folds": 0, "folds": []}
    return {"mean_auc": float(np.mean(folds)), "std_auc": float(np.std(folds)),
            "n_folds": len(folds), "folds": [float(x) for x in folds]}


def _paired(a, b):
    if not a or not b or len(a) != len(b):
        return None, None
    d = np.array(a) - np.array(b)
    med = float(np.median(d))
    if not np.any(d != 0):
        return None, med
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(wilcoxon(a, b, alternative="greater").pvalue), med
        except ValueError:
            return None, med


def _frame2(df):
    base_rate = float(df["is_correct"].mean())
    underpowered = base_rate > 0.85 or base_rate < 0.15
    m0 = _auc_for(df, CONF)
    m1 = _auc_for(df, CONF + H1)
    m2 = _auc_for(df, CONF + SHEAF)
    m3 = _auc_for(df, CONF + H1 + SHEAF)
    sheaf_only = _auc_for(df, SHEAF)

    p_sheaf_vs_conf, med_sheaf_vs_conf = _paired(m2, m0)
    p_sheaf_vs_h1full, med_sheaf_vs_h1full = _paired(m3, m1)

    sheaf_beats_chance = bool(sheaf_only and np.mean(sheaf_only) > 0.5)
    sheaf_adds = bool(p_sheaf_vs_conf is not None and p_sheaf_vs_conf < ALPHA
                      and med_sheaf_vs_conf is not None and med_sheaf_vs_conf > 0)

    if underpowered:
        verdict = "UNDERPOWERED"
    elif not sheaf_beats_chance:
        verdict = "RED"
    elif sheaf_adds:
        verdict = "GREEN"
    else:
        verdict = "PARTIAL"

    return {
        "base_rate_correct": base_rate,
        "underpowered": underpowered,
        "auc": {
            "confidence_only": _summary(m0),
            "confidence_plus_h1": _summary(m1),
            "confidence_plus_sheaf": _summary(m2),
            "confidence_plus_h1_plus_sheaf": _summary(m3),
            "sheaf_only": _summary(sheaf_only),
        },
        "sheaf_vs_confidence": {"median_delta": med_sheaf_vs_conf,
                                "wilcoxon_p": p_sheaf_vs_conf},
        "sheaf_vs_h1full": {"median_delta": med_sheaf_vs_h1full,
                            "wilcoxon_p": p_sheaf_vs_h1full},
        "sheaf_beats_chance": sheaf_beats_chance,
        "sheaf_adds_beyond_confidence": sheaf_adds,
        "verdict": verdict,
    }


def compute_stats(features_path="results/exp8_features.parquet",
                  out_path="results/exp8_stats.json"):
    df = pd.read_parquet(features_path)
    stats = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "alpha": ALPHA,
        "n_splits": N_SPLITS,
        "delta_r2_floor": DELTA_R2_FLOOR,
        "n_items": int(len(df)),
        "overall_accuracy": float(df["is_correct"].mean()),
        "frame1_discrimination": _frame1(df),
        "frame2_failure_prediction": _frame2(df),
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    compute_stats()
