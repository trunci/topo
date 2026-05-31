"""Phase A2 -- length/entity de-confounding of the Fiedler "beyond H1" signal (Exp 9b).

The ONLY surviving positive from Exp 9a was SHAPE: Fiedler value (algebraic
connectivity) added beyond H1 at predicting reasoning hop-count (delta_r2 = 0.044,
p = 4.3e-8). But more hops => longer prompt => more named entities, and Fiedler (a
size-/density-sensitive graph statistic) may simply be reading sequence length.

This module re-runs the decisive nested test while CONTROLLING for surface
complexity: sequence length (chat-templated token count, exactly as Exp 8 encoded
it) and entity count (distinct names in the prompt). Same machinery as Exp 9a,
same EXISTING results/exp8_features.parquet -- we only add two surface columns,
re-derived deterministically from the item ids.

Tests (predicting hop):
  * fiedler_beyond_h1            -- replicates Exp 9a (sanity; should reproduce 0.044).
  * fiedler_beyond_h1_len        -- Fiedler beyond [H1 + ctrl + LEN + ENT].  DECISIVE.
  * flow_beyond_shape_len        -- discord beyond [H1 + ctrl + Fiedler + LEN + ENT].
  * hop_from_length_only         -- R^2 of hop ~ LEN + ENT (how much of hop IS surface).
  * fiedler_partial_spearman     -- Spearman(fiedler_mean, hop | H1+ctrl+LEN+ENT).

Verdict (pre-registered):
  LENGTH-ROBUST    -- Fiedler still adds beyond [H1+ctrl+LEN+ENT] (delta_r2 >= 0.02, p<0.05).
  LENGTH-CONFOUND  -- Fiedler adds beyond [H1+ctrl] but NOT beyond [H1+ctrl+LEN+ENT].
  NULL             -- Fiedler adds in neither.

Single source of truth: writes results/exp9b_length_stats.json. No hand-typed numbers.
"""
from __future__ import annotations

import json

import pandas as pd

from src.compute_stats_exp9a import (ALPHA, DELTA_R2_FLOOR, H1, CTRL, FIEDLER, FLOW,
                                     _nested, _partial_spearman)
from src.data_gen import NAMES, build_items

SURFACE = ["seq_length", "n_entities"]


def _n_entities(prompt: str) -> int:
    """Distinct first names from the data_gen vocabulary appearing in the prompt."""
    return sum(1 for nm in NAMES if nm in prompt)


def add_surface_columns(df, tok, n_per_family=30, hops=(1, 2, 3), seed=0):
    """Attach seq_length (chat-templated token count) + n_entities per item id.

    Re-derives the exact items Exp 8 ran (same builder args) and tokenizes each
    prompt with the same chat template / tokenizer Exp 8 used, so seq_length matches
    the graph the sheaf was built on.
    """
    from src.attn_extract import format_prompt
    items = {it.id: it for it in build_items(n_per_family=n_per_family, hops=hops, seed=seed)}
    seq_len, n_ent = [], []
    for _id in df["id"]:
        it = items[_id]
        ids = tok(format_prompt(tok, it.prompt))["input_ids"]
        seq_len.append(len(ids))
        n_ent.append(_n_entities(it.prompt))
    df = df.copy()
    df["seq_length"] = seq_len
    df["n_entities"] = n_ent
    return df


def compute_stats(df, out_path="results/exp9b_length_stats.json", target="hop"):
    """df must already carry seq_length + n_entities (via add_surface_columns)."""
    fiedler_vs_h1 = _nested(df, H1 + CTRL, FIEDLER, target)
    fiedler_vs_h1_len = _nested(df, H1 + CTRL + SURFACE, FIEDLER, target)
    flow_vs_shape_len = _nested(df, H1 + CTRL + FIEDLER + SURFACE, FLOW, target)
    hop_from_len = _nested(df, [], SURFACE, target)  # baseline = intercept only
    fiedler_partial = _partial_spearman(df, "sheaf_t1_fiedler_mean", target,
                                        H1 + CTRL + SURFACE)

    if fiedler_vs_h1_len["adds"]:
        verdict = "LENGTH-ROBUST"
    elif fiedler_vs_h1["adds"]:
        verdict = "LENGTH-CONFOUND"
    else:
        verdict = "NULL"

    stats = {
        "phase": "A2 length/entity de-confounding",
        "source": "results/exp8_features.parquet",
        "target": target,
        "n_items": int(len(df)),
        "alpha": ALPHA,
        "delta_r2_floor": DELTA_R2_FLOOR,
        "surface_cols": SURFACE,
        "fiedler_beyond_h1": fiedler_vs_h1,
        "fiedler_beyond_h1_len": fiedler_vs_h1_len,
        "flow_beyond_shape_len": flow_vs_shape_len,
        "hop_from_length_only": hop_from_len,
        "fiedler_partial_spearman_len": fiedler_partial,
        "verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main(features_path="results/exp8_features.parquet",
         out_path="results/exp9b_length_stats.json"):
    from src.attn_extract import MODEL_NAME
    from transformers import AutoTokenizer
    df = pd.read_parquet(features_path)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = add_surface_columns(df, tok)
    return compute_stats(df, out_path=out_path)


if __name__ == "__main__":
    main()
