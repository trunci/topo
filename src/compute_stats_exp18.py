"""Authoritative statistics for Experiment 18 -> results/exp18_stats.json.

Implements exactly the pre-registered hypotheses of FINDINGS_exp18.md:
H-B1 (provenance: parametric vs context-derived, on the distractor-correct
subset) and H-B2 (robustness: stays-correct under distractors, on the
gold-correct subset), with the amended primary per-head family = gold-mass
fraction. Secondary: sway replication (S1) and head depth profile (S2).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.compute_stats_exp16 import (_bootstrap_auc, _bootstrap_delta,
                                     _fit_oof, _folds)
from src.compute_stats_exp17 import _head_selected, _loose_agree, _probe_block

N_LAYERS, N_HEADS_PER = 32, 32
ALPHA = 0.05


def _ph(df, col):
    return np.stack(df[col].to_list()).astype(float)


def _gold_frac(df):
    g, d = _ph(df, "ph_gold_mass"), _ph(df, "ph_dist_mass")
    return g / (g + d + 1e-12)


def _hypothesis(df, y_pos, rng, extra_baselines=()):
    """Shared machinery for H-B1/H-B2: y_pos = 1 for the positive class the
    scalar should rank high (parametric / broke-under-distractors)."""
    base_rate = float(y_pos.mean())
    out = {"n": int(len(y_pos)), "base_rate_positive": base_rate}
    if not (0.15 <= base_rate <= 0.85) or len(y_pos) < 60:
        out["verdict"] = "UNDERPOWERED"
        return out
    folds = _folds(1 - y_pos)

    fams = {
        "gold_frac": _gold_frac(df),                       # PRIMARY
        "gold_mass": _ph(df, "ph_gold_mass"),
        "dist_mass": _ph(df, "ph_dist_mass"),
        "q_mass": _ph(df, "ph_q_mass"),
        "resp_ent": _ph(df, "ph_resp_ent"),
    }
    scalars, hs = {}, {}
    for name, PH in fams.items():
        scalar, heads = _head_selected(PH, y_pos, folds)
        scalars[name] = scalar
        hs[name] = {**_bootstrap_auc(y_pos, scalar, rng),
                    "selected_heads_per_fold": heads}

    baselines = {
        "confidence": _probe_block(df[["confidence_margin"]].to_numpy(float),
                                   y_pos, folds, rng),
        "ctx_mass_pooled": _probe_block(df[["ctx_mass_pooled"]].to_numpy(float),
                                        y_pos, folds, rng),
        "gold_frac_pooled": _probe_block(df[["gold_frac_pooled"]].to_numpy(float),
                                         y_pos, folds, rng),
    }
    for name, cols in extra_baselines:
        baselines[name] = _probe_block(df[cols].to_numpy(float), y_pos,
                                       folds, rng)

    # verdict deltas: primary scalar vs each required baseline, paired on items
    prim = scalars["gold_frac"]

    def _delta_vs(cols):
        oof, _ = _fit_oof(df[cols].to_numpy(float), 1 - y_pos, folds)
        return _bootstrap_delta(y_pos, 1 - oof, prim, rng)

    d_conf = _delta_vs(["confidence_margin"])
    d_mass = _delta_vs(["ctx_mass_pooled"])

    beats_chance = hs["gold_frac"]["ci95"][0] > 0.5
    beats_conf = d_conf["p_one_sided"] < ALPHA and d_conf["delta_auc"] > 0
    beats_mass = d_mass["p_one_sided"] < ALPHA and d_mass["delta_auc"] > 0
    verdict = ("GREEN" if beats_chance and beats_conf and beats_mass
               else "PARTIAL" if beats_chance else "RED")

    out.update({
        "head_selected": hs,
        "baselines": baselines,
        "primary_vs_confidence": d_conf,
        "primary_vs_ctx_mass": d_mass,
        "beats_chance": bool(beats_chance),
        "beats_confidence": bool(beats_conf),
        "beats_ctx_mass": bool(beats_mass),
        "verdict": verdict,
    })
    return out


def compute_stats(out_path="results/exp18_stats.json"):
    rng = np.random.default_rng(0)
    nc = pd.read_parquet("results/exp18_nocontext_features.parquet")
    sp = pd.read_parquet("results/exp18_spans_features.parquet")
    g16 = pd.read_parquet("results/exp16_gold_features.parquet")
    d16 = pd.read_parquet("results/exp16_distractor_features.parquet")

    m = sp.merge(nc[["id", "is_correct", "generated"]], on="id",
                 suffixes=("", "_nc"))
    m = m.merge(g16[["id", "is_correct", "generated"]].rename(
        columns={"is_correct": "is_correct_g16", "generated": "generated_g16"}),
        on="id")
    m = m.merge(d16[["id", "generated"]].rename(
        columns={"generated": "generated_d16"}), on="id")

    stats = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "n_items": int(len(m)),
        "acc_spans": float(m["is_correct"].mean()),
        "acc_nocontext": float(m["is_correct_nc"].mean()),
        "knew_anyway_rate": float(m["is_correct_nc"].mean()),
        "regen_agreement_with_exp16": float(np.mean([
            _loose_agree(a, b) for a, b in zip(m["generated"],
                                               m["generated_d16"])])),
    }

    # ---- H-B1: provenance on the distractor-correct subset ----
    sub1 = m[m["is_correct"] == 1].reset_index(drop=True)
    parametric = sub1["is_correct_nc"].to_numpy(int)   # 1 = knew anyway
    stats["h_b1_provenance"] = _hypothesis(sub1, parametric, rng)

    # ---- H-B2: robustness on the gold-correct subset ----
    sub2 = m[m["is_correct_g16"] == 1].reset_index(drop=True)
    broke = 1 - sub2["is_correct"].to_numpy(int)        # 1 = distractors broke it
    stats["h_b2_robustness"] = _hypothesis(sub2, broke, rng)

    # ---- S1: sway replication with span features (no verdict) ----
    sway = np.array([0 if _loose_agree(a, b) else 1
                     for a, b in zip(m["generated"], m["generated_g16"])])
    stats["s1_sway"] = _hypothesis(m, sway, rng)
    stats["s1_sway"]["note"] = "secondary, no verdict; label = exp18-spans vs exp16-gold answer divergence"

    # ---- S2: depth profile of primary heads on H-B1 ----
    hb1 = stats["h_b1_provenance"]
    if "head_selected" in hb1:
        sel = sorted({h for fold in
                      hb1["head_selected"]["gold_frac"]["selected_heads_per_fold"]
                      for h in fold})
        stats["s2_depth"] = {
            "selected_union": sel,
            "layer_hist": np.bincount([h // N_HEADS_PER for h in sel],
                                      minlength=N_LAYERS).tolist(),
        }

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def _print_hyp(name, h):
    if h.get("verdict") == "UNDERPOWERED":
        print(f"{name}: UNDERPOWERED (n={h['n']}, base={h['base_rate_positive']:.3f})")
        return
    p = h["head_selected"]["gold_frac"]
    print(f"{name}: {h['verdict']}  n={h['n']} base={h['base_rate_positive']:.3f}  "
          f"gold_frac AUC={p['auc']:.3f} CI[{p['ci95'][0]:.3f},{p['ci95'][1]:.3f}]  "
          f"conf={h['baselines']['confidence']['auc']:.3f}  "
          f"ctx_mass={h['baselines']['ctx_mass_pooled']['auc']:.3f}  "
          f"dConf p={h['primary_vs_confidence']['p_one_sided']:.4f}  "
          f"dMass p={h['primary_vs_ctx_mass']['p_one_sided']:.4f}")


if __name__ == "__main__":
    s = compute_stats()
    print(f"acc spans={s['acc_spans']:.3f} nocontext={s['acc_nocontext']:.3f} "
          f"regen-agreement={s['regen_agreement_with_exp16']:.3f}")
    _print_hyp("H-B1 provenance", s["h_b1_provenance"])
    _print_hyp("H-B2 robustness", s["h_b2_robustness"])
    _print_hyp("S1 sway", s["s1_sway"])
