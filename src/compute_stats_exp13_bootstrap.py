"""Item-level paired bootstrap for the headline delta-AUC claims
-> results/exp13_bootstrap_stats.json.

The fold-level Wilcoxon tests in exp13_stats / exp13_1p5b_stats sit at the
resolution floor of a five-fold test (p = 1/32). This script sharpens them
with an item-level analysis: assemble out-of-fold predictions from the same
StratifiedKFold(5, seed 0) probes as compute_stats_exp13, compute the AUC of
each nested model pair on the full out-of-fold vector, and bootstrap-resample
items (paired, B = 10 000, rng seed 0) to get a CI on delta-AUC and a
one-sided achieved significance level P(delta <= 0).

Claims tested:
* 0.5B seed 0, all items (n=120):   confidence+topology vs confidence
* 0.5B pooled 3 seeds (n=360):      confidence+topology vs confidence
* 1.5B, all items (n=180):          controls+topology  vs controls

The bootstrap conditions on the fitted fold models (standard practice for
comparing classifiers on shared test predictions).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.compute_stats_exp13 import CONF, CTRL, N_SPLITS, SEED, TOPO

B = 10_000
RNG_SEED = 0


def _oof_predictions(df, cols):
    """Out-of-fold predicted probabilities, same folds/probe as compute_stats_exp13."""
    X = df[cols].to_numpy(dtype=float)
    y = df["is_correct"].to_numpy(dtype=int)
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(scaler.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]
    assert not np.isnan(oof).any()
    return oof, y


def _auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def _paired_bootstrap(df, cols_a, cols_b):
    pa, y = _oof_predictions(df, cols_a)
    pb, _ = _oof_predictions(df, cols_b)
    auc_a, auc_b = _auc(y, pa), _auc(y, pb)
    delta = auc_b - auc_a

    rng = np.random.default_rng(RNG_SEED)
    n = len(y)
    deltas = []
    while len(deltas) < B:
        idx = rng.integers(0, n, size=n)
        da, db = _auc(y[idx], pa[idx]), _auc(y[idx], pb[idx])
        if da is None:  # resample lost a class; draw again
            continue
        deltas.append(db - da)
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # one-sided achieved significance level, add-one smoothed
    p_one_sided = (np.sum(deltas <= 0) + 1) / (B + 1)

    return {
        "n_items": int(n),
        "auc_baseline": float(auc_a),
        "auc_full": float(auc_b),
        "delta_auc": float(delta),
        "bootstrap": {
            "B": B,
            "ci95": [float(lo), float(hi)],
            "p_one_sided": float(p_one_sided),
        },
    }


def compute_stats(out_path="results/exp13_bootstrap_stats.json"):
    seed0 = pd.read_parquet("results/exp13_features.parquet")
    pooled = pd.concat([
        seed0,
        pd.read_parquet("results/exp13_s1_features.parquet"),
        pd.read_parquet("results/exp13_s2_features.parquet"),
    ], ignore_index=True)
    b15 = pd.read_parquet("results/exp13_1p5b_features.parquet")

    stats = {
        "method": ("out-of-fold predictions from StratifiedKFold(5, seed 0) "
                   "logistic probes; paired item bootstrap, B=10000, rng seed 0; "
                   "one-sided ASL = P(delta<=0), add-one smoothed"),
        "seed0_topo_beyond_confidence": _paired_bootstrap(
            seed0, CONF, CONF + TOPO),
        "pooled3_topo_beyond_confidence": _paired_bootstrap(
            pooled, CONF, CONF + TOPO),
        "b1p5_topo_beyond_controls": _paired_bootstrap(
            b15, CTRL, CTRL + TOPO),
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    s = compute_stats()
    for k in ("seed0_topo_beyond_confidence", "pooled3_topo_beyond_confidence",
              "b1p5_topo_beyond_controls"):
        v = s[k]
        print(f"{k}: n={v['n_items']}  {v['auc_baseline']:.3f} -> "
              f"{v['auc_full']:.3f}  dAUC={v['delta_auc']:.3f}  "
              f"CI95=[{v['bootstrap']['ci95'][0]:.3f}, "
              f"{v['bootstrap']['ci95'][1]:.3f}]  "
              f"p={v['bootstrap']['p_one_sided']:.5f}")
