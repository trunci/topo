"""Experiment 2: per-head induction score vs H1 topology in GPT-2 small.

Loads GPT-2 (CPU), runs repeated-random sequences, and for each (layer, head)
records induction score, H1 total persistence (reusing topology.h1_features),
and cheap distance baselines -> results/exp2.parquet (one row per head).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch

from src.attn_extract import load_named
from src.topology import h1_features
from src.induction import (
    make_repeat_batch, induction_score, attention_distance, offdiag_mass,
)

MODEL = "gpt2"
SEQ_LEN = 25
PREFIX_LEN = 5
N_SEQS = 8
TOP_K = 8
SEED = 0


def run(out_path="results/exp2.parquet", seq_len=SEQ_LEN, prefix_len=PREFIX_LEN,
        n_seqs=N_SEQS, top_k=TOP_K, seed=SEED):
    os.makedirs("results", exist_ok=True)
    model, tok, device = load_named(MODEL, device="cpu")
    L = model.config.n_layer
    H = model.config.n_head
    vocab = model.config.vocab_size

    batch = make_repeat_batch(vocab, seq_len, prefix_len, n_seqs, seed=seed)
    input_ids = torch.tensor(batch, dtype=torch.long, device=device)
    n = input_ids.shape[1]

    sums = {(l, h): {"ind": 0.0, "h1": 0.0, "dist": 0.0, "off": 0.0}
            for l in range(L) for h in range(H)}

    with torch.no_grad():
        for s in range(input_ids.shape[0]):
            out = model(input_ids[s:s + 1], output_attentions=True)
            for l in range(L):
                att = out.attentions[l][0].float().cpu().numpy()  # [H, n, n]
                for h in range(H):
                    A = att[h]
                    sums[(l, h)]["ind"] += induction_score(A, seq_len, prefix_len)
                    sums[(l, h)]["h1"] += h1_features(A, top_k=top_k)["total_persistence"]
                    sums[(l, h)]["dist"] += attention_distance(A)
                    sums[(l, h)]["off"] += offdiag_mass(A)
            print(f"[exp2] sequence {s + 1}/{input_ids.shape[0]} done (n={n})", flush=True)

    nseq = input_ids.shape[0]
    rows = []
    for (l, h), d in sums.items():
        rows.append({
            "layer": l, "head": h,
            "induction_score": d["ind"] / nseq,
            "h1_persistence": d["h1"] / nseq,
            "attn_distance": d["dist"] / nseq,
            "offdiag_mass": d["off"] / nseq,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"[exp2] wrote {len(df)} head rows to {out_path}", flush=True)
    return df


if __name__ == "__main__":
    run()
