"""Experiment 4: per-head circuit scores + H1 topology + first-order scalars,
across two models (gpt2, distilgpt2) -> results/exp4.parquet.

Reuses the Exp 3 CPU loop pattern, adds two circuit columns (previous-token,
duplicate-token) and a "model" column, and iterates over both models. One model
is held in memory at a time (del + gc.collect() between models).

Memory-safe: CPU only, float32, eager attention, torch threads capped, modest
batch (n_seqs<=8, seq_len<=25). No MPS, no parallel heavy work. (Earlier MPS/
large/parallel runs crash-restarted the machine.)
"""
from __future__ import annotations

import gc
import os
import pandas as pd
import torch

from src.attn_extract import load_named
from src.topology import h1_features
from src.residual import attention_entropy
from src.circuits import previous_token_score, duplicate_token_score
from src.induction import (
    make_repeat_batch, induction_score, attention_distance, offdiag_mass,
)

MODELS = ["gpt2", "distilgpt2"]
SEQ_LEN = 25
PREFIX_LEN = 5
N_SEQS = 8
TOP_K = 8
SEED = 0


def _run_model(model_name, seq_len, prefix_len, n_seqs, top_k, seed):
    """Return a list of per-(layer,head) row dicts for one model (CPU)."""
    print(f"[exp4] loading {model_name} (cpu)...", flush=True)
    model, tok, device = load_named(model_name, device="cpu")
    try:
        L = model.config.n_layer
        H = model.config.n_head
        vocab = model.config.vocab_size

        batch = make_repeat_batch(vocab, seq_len, prefix_len, n_seqs, seed=seed)
        input_ids = torch.tensor(batch, dtype=torch.long, device=device)
        n = input_ids.shape[1]

        sums = {(l, h): {"ind": 0.0, "prev": 0.0, "dup": 0.0, "h1": 0.0,
                         "dist": 0.0, "off": 0.0, "ent": 0.0}
                for l in range(L) for h in range(H)}

        with torch.no_grad():
            for s in range(input_ids.shape[0]):
                out = model(input_ids[s:s + 1], output_attentions=True)
                for l in range(L):
                    att = out.attentions[l][0].float().cpu().numpy()  # [H, n, n]
                    for h in range(H):
                        A = att[h]
                        d = sums[(l, h)]
                        d["ind"] += induction_score(A, seq_len, prefix_len)
                        d["prev"] += previous_token_score(A)
                        d["dup"] += duplicate_token_score(A, seq_len, prefix_len)
                        d["h1"] += h1_features(A, top_k=top_k)["total_persistence"]
                        d["dist"] += attention_distance(A)
                        d["off"] += offdiag_mass(A)
                        d["ent"] += attention_entropy(A)
                print(f"[exp4] {model_name} sequence {s + 1}/{input_ids.shape[0]} "
                      f"done (n={n})", flush=True)

        nseq = input_ids.shape[0]
        rows = []
        for (l, h), d in sums.items():
            rows.append({
                "model": model_name, "layer": l, "head": h,
                "induction_score": d["ind"] / nseq,
                "prev_token_score": d["prev"] / nseq,
                "dup_token_score": d["dup"] / nseq,
                "h1_persistence": d["h1"] / nseq,
                "attn_distance": d["dist"] / nseq,
                "offdiag_mass": d["off"] / nseq,
                "attn_entropy": d["ent"] / nseq,
            })
        print(f"[exp4] {model_name}: {len(rows)} head rows", flush=True)
        return rows
    finally:
        del model
        gc.collect()


def run(out_path="results/exp4.parquet", models=MODELS, seq_len=SEQ_LEN,
        prefix_len=PREFIX_LEN, n_seqs=N_SEQS, top_k=TOP_K, seed=SEED):
    torch.set_num_threads(2)
    os.makedirs("results", exist_ok=True)
    all_rows = []
    for model_name in models:
        all_rows.extend(_run_model(model_name, seq_len, prefix_len, n_seqs,
                                   top_k, seed))
        gc.collect()
    df = pd.DataFrame(all_rows)
    df.to_parquet(out_path)
    print(f"[exp4] wrote {len(df)} rows ({df['model'].nunique()} models) to "
          f"{out_path}", flush=True)
    return df


if __name__ == "__main__":
    run()
