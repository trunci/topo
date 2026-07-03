"""Authoritative statistics for Experiment 16 -> results/exp16_stats.json.

Implements exactly the pre-registered analyses of FINDINGS_exp16.md:
per condition (distractor, gold) --
* pooled MTop-Div probe
* fold-internal TOHA-style head-selected scalar (top 10 heads by train AUC)
* head-selected beyond confidence / beyond response-entropy controls
* item-level paired bootstrap on out-of-fold predictions (B = 10,000)

Verdict (pre-registered):
GREEN   head_selected bootstrap CI95 lower bound > 0.5 AND adds beyond
        confidence (bootstrap one-sided p < 0.05, delta median > 0)
PARTIAL head_selected beats chance by the CI rule, no add beyond confidence
RED     head_selected does not beat chance
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

N_SPLITS = 5
SEED = 0
TOP_HEADS = 10
B = 10_000
ALPHA = 0.05

POOLED = ["mtd_mean", "mtd_max", "mtd_std"]
CONF = ["confidence_margin"]
CTRL = ["ctrl_resp_entropy"]


def _folds(y):
    return list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=SEED).split(np.zeros_like(y), y))


def _auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def _fit_oof(X, y, folds):
    """Out-of-fold probabilities from the standard scaler+logistic probe."""
    oof = np.full(len(y), np.nan)
    fold_aucs = []
    for tr, te in folds:
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(scaler.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        if len(np.unique(y[te])) == 2:
            fold_aucs.append(float(roc_auc_score(y[te], oof[te])))
    return oof, fold_aucs


def _head_selected_scalar(PH, y, folds):
    """Fold-internal TOHA protocol -> one out-of-fold scalar per item.

    In each training fold: rank heads by |single-head train AUC - 0.5|, take
    TOP_HEADS, z-score on train stats, sign-align so higher = predicted
    failure (y = is_correct, so failure = y == 0), average.
    """
    scalar = np.full(len(y), np.nan)
    fail = 1 - y
    for tr, te in folds:
        aucs = np.array([_auc(fail[tr], PH[tr, h]) for h in range(PH.shape[1])],
                        dtype=float)
        top = np.argsort(-np.abs(aucs - 0.5))[:TOP_HEADS]
        mu = PH[tr][:, top].mean(axis=0)
        sd = PH[tr][:, top].std(axis=0) + 1e-12
        sign = np.where(aucs[top] >= 0.5, 1.0, -1.0)
        scalar[te] = ((PH[te][:, top] - mu) / sd * sign).mean(axis=1)
    return scalar


def _scalar_fold_aucs(scalar, y, folds):
    out = []
    for _, te in folds:
        if len(np.unique(y[te])) == 2:
            out.append(float(roc_auc_score(1 - y[te], scalar[te])))
    return out


def _bootstrap_auc(y_fail, score, rng):
    """CI on AUC of `score` for predicting y_fail."""
    n = len(y_fail)
    vals = []
    while len(vals) < B:
        idx = rng.integers(0, n, size=n)
        a = _auc(y_fail[idx], score[idx])
        if a is not None:
            vals.append(a)
    vals = np.array(vals)
    return {
        "auc": float(_auc(y_fail, score)),
        "ci95": [float(np.percentile(vals, 2.5)),
                 float(np.percentile(vals, 97.5))],
    }


def _bootstrap_delta(y_fail, score_base, score_full, rng):
    """Paired bootstrap on delta-AUC (full - base)."""
    n = len(y_fail)
    deltas = []
    while len(deltas) < B:
        idx = rng.integers(0, n, size=n)
        a = _auc(y_fail[idx], score_base[idx])
        b = _auc(y_fail[idx], score_full[idx])
        if a is None:
            continue
        deltas.append(b - a)
    deltas = np.array(deltas)
    return {
        "delta_auc": float(_auc(y_fail, score_full) - _auc(y_fail, score_base)),
        "ci95": [float(np.percentile(deltas, 2.5)),
                 float(np.percentile(deltas, 97.5))],
        "p_one_sided": float((np.sum(deltas <= 0) + 1) / (B + 1)),
    }


def _summary(fold_aucs):
    return {"mean_auc": float(np.mean(fold_aucs)) if fold_aucs else None,
            "std_auc": float(np.std(fold_aucs)) if fold_aucs else None,
            "n_folds": len(fold_aucs), "folds": [float(a) for a in fold_aucs]}


def _condition_stats(df, rng):
    y = df["is_correct"].to_numpy(dtype=int)
    fail = 1 - y
    base_rate = float(y.mean())
    if base_rate > 0.85 or base_rate < 0.15:
        return {"n_items": int(len(y)), "base_rate_correct": base_rate,
                "verdict": "UNDERPOWERED"}
    folds = _folds(y)
    PH = np.stack(df["ph_mtd"].to_list()).astype(float)

    conf_oof, conf_folds = _fit_oof(df[CONF].to_numpy(float), y, folds)
    pooled_oof, pooled_folds = _fit_oof(df[POOLED].to_numpy(float), y, folds)
    ctrl_oof, ctrl_folds = _fit_oof(df[CTRL].to_numpy(float), y, folds)

    hs = _head_selected_scalar(PH, y, folds)
    hs_folds = _scalar_fold_aucs(hs, y, folds)

    # for "beyond X" probes the head-selected scalar joins the feature block,
    # still fold-internally: refit the probe with hs as an extra column
    Xc = np.column_stack([df[CONF].to_numpy(float), hs])
    conf_hs_oof, conf_hs_folds = _fit_oof(Xc, y, folds)
    Xe = np.column_stack([df[CTRL].to_numpy(float), hs])
    ctrl_hs_oof, ctrl_hs_folds = _fit_oof(Xe, y, folds)

    # bootstrap: predict FAILURE, so flip probe probabilities of is_correct
    hs_boot = _bootstrap_auc(fail, hs, rng)
    pooled_boot = _bootstrap_auc(fail, 1 - pooled_oof, rng)
    add_conf = _bootstrap_delta(fail, 1 - conf_oof, 1 - conf_hs_oof, rng)
    add_ctrl = _bootstrap_delta(fail, 1 - ctrl_oof, 1 - ctrl_hs_oof, rng)

    def _wilc(m1, m0):
        d = np.array(m1) - np.array(m0)
        if not np.any(d != 0):
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return float(wilcoxon(m1, m0, alternative="greater").pvalue)
            except ValueError:
                return None

    beats_chance = hs_boot["ci95"][0] > 0.5
    adds_conf = (add_conf["p_one_sided"] < ALPHA
                 and add_conf["delta_auc"] > 0)
    verdict = ("GREEN" if beats_chance and adds_conf
               else "PARTIAL" if beats_chance else "RED")

    return {
        "n_items": int(len(y)),
        "base_rate_correct": base_rate,
        "acc": base_rate,
        "auc": {
            "confidence_only": _summary(conf_folds),
            "pooled_mtd": _summary(pooled_folds),
            "resp_entropy_controls": _summary(ctrl_folds),
            "head_selected": _summary(hs_folds),
            "confidence_plus_head_selected": _summary(conf_hs_folds),
            "controls_plus_head_selected": _summary(ctrl_hs_folds),
        },
        "bootstrap": {
            "head_selected": hs_boot,
            "pooled_mtd": pooled_boot,
            "head_selected_beyond_confidence": add_conf,
            "head_selected_beyond_controls": add_ctrl,
        },
        "fold_wilcoxon": {
            "beyond_confidence_p": _wilc(conf_hs_folds, conf_folds),
            "beyond_controls_p": _wilc(ctrl_hs_folds, ctrl_folds),
        },
        "head_selected_beats_chance": bool(beats_chance),
        "head_selected_adds_beyond_confidence": bool(adds_conf),
        "reaches_toha_band": bool(hs_boot["auc"] >= 0.63
                                  or pooled_boot["auc"] >= 0.63),
        "verdict": verdict,
    }


def compute_stats(out_path="results/exp16_stats.json"):
    rng = np.random.default_rng(0)
    stats = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "alpha": ALPHA,
        "n_splits": N_SPLITS,
        "top_heads": TOP_HEADS,
        "toha_reference_auroc": 0.71,
        "toha_reference_band_low": 0.63,
    }
    for cond in ("distractor", "gold"):
        df = pd.read_parquet(f"results/exp16_{cond}_features.parquet")
        stats[cond] = _condition_stats(df, rng)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    s = compute_stats()
    for cond in ("distractor", "gold"):
        c = s[cond]
        if c["verdict"] == "UNDERPOWERED":
            print(f"{cond}: UNDERPOWERED (base rate {c['base_rate_correct']:.3f})")
            continue
        hb = c["bootstrap"]["head_selected"]
        ac = c["bootstrap"]["head_selected_beyond_confidence"]
        print(f"{cond}: {c['verdict']}  acc={c['acc']:.3f}  "
              f"head_sel AUC={hb['auc']:.3f} CI[{hb['ci95'][0]:.3f},"
              f"{hb['ci95'][1]:.3f}]  conf={c['auc']['confidence_only']['mean_auc']:.3f}  "
              f"dAUC_vs_conf={ac['delta_auc']:.3f} (p={ac['p_one_sided']:.4f})  "
              f"toha_band={c['reaches_toha_band']}")
