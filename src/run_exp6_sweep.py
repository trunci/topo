"""Experiment 6b: gpt2 top_k sweep -- does the INCONCLUSIVE verdict just reflect a
starved treatment budget?

Exp 6 left gpt2 INCONCLUSIVE: mean K ~ 2.5 directed gap~S critical-cycle edges per
sequence (several seqs K=0), so the ablation had almost nothing to remove. This
script tests whether raising top_k (the per-head sparsification budget) yields more
gap~S critical-cycle edges and a measurable causal effect -- isolating the budget
variable on the SAME model.

Loads gpt2 ONCE, captures attentions for all sequences once, then re-derives the
cycle/magnitude/random ablations for each top_k over the cached attentions (gentle:
no repeated model loads; only the forward passes per condition are re-run).

Output: results/exp6_sweep.parquet, one row per (top_k, seq, condition) with
second_copy_loss and k_ablated. Downstream stats per top_k via compute_stats_exp6
(filtering by the top_k column).

Memory-safe: CPU only, float32, eager, torch.set_num_threads(2), one model.
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import torch

from src.attn_extract import load_named
from src.induction import make_repeat_batch
from src.run_exp5 import second_copy_loss, _head_induction_scores
from src.run_exp6 import (directed_gap_cycle_edges, directed_causal_edges,
                          select_magnitude_directed, select_random_directed)

MODEL = "gpt2"
SEQ_LEN = 25
PREFIX_LEN = 5
N_SEQS = 12
TOP_KS = [8, 16, 24]
GAP_TOL = 2
K_HEADS = 10
SEED = 0


def run(out="results/exp6_sweep.parquet", model_name=MODEL, seq_len=SEQ_LEN,
        prefix_len=PREFIX_LEN, n_seqs=N_SEQS, top_ks=TOP_KS, gap_tol=GAP_TOL,
        k_heads=K_HEADS, seed=SEED):
    torch.set_num_threads(2)
    os.makedirs("results", exist_ok=True)
    from src.pruned_forward import MaskedModel, ablation_to_bias

    print(f"[exp6b] loading {model_name} (cpu)...", flush=True)
    model, tok, device = load_named(model_name, device="cpu")
    rows = []
    try:
        L, H, vocab = model.config.n_layer, model.config.n_head, model.config.vocab_size
        mm = MaskedModel(model)
        batch = make_repeat_batch(vocab, seq_len, prefix_len, n_seqs, seed=seed)
        input_ids = torch.tensor(batch, dtype=torch.long, device=device)
        n = input_ids.shape[1]

        attentions = []
        with torch.no_grad():
            for s in range(n_seqs):
                o = model(input_ids[s:s + 1], output_attentions=True)
                attentions.append(np.stack([a[0].float().cpu().numpy()
                                            for a in o.attentions]))
                print(f"[exp6b] captured seq {s + 1}/{n_seqs}", flush=True)

        scores = _head_induction_scores(attentions, L, H, seq_len, prefix_len)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        induction_heads = [lh for lh, _ in ordered[:k_heads]]
        print(f"[exp6b] induction heads {induction_heads}", flush=True)

        # cache unmasked loss per seq (independent of top_k)
        unmasked = {}
        for s in range(n_seqs):
            enc = {"input_ids": input_ids[s:s + 1]}
            unmasked[s] = second_copy_loss(mm.logits(enc, biases=None),
                                           input_ids[s:s + 1], prefix_len, seq_len)

        for tk in top_ks:
            print(f"[exp6b] === top_k = {tk} ===", flush=True)
            for s in range(n_seqs):
                att = attentions[s]
                enc = {"input_ids": input_ids[s:s + 1]}
                abl_cycle, abl_mag, abl_rnd = {}, {}, {}
                k_total = 0
                for (l, h) in induction_heads:
                    A = att[l, h]
                    cyc = directed_gap_cycle_edges(A, seq_len, gap_tol, tk)
                    m = len(cyc)
                    if m == 0:
                        continue
                    k_total += m
                    cand = directed_causal_edges(A)
                    abl_cycle[(l, h)] = cyc
                    abl_mag[(l, h)] = select_magnitude_directed(A, cand, m)
                    abl_rnd[(l, h)] = select_random_directed(
                        cand, m, seed=seed * 100000 + s * 1000 + l * 100 + h)

                losses = {"unmasked": unmasked[s]}
                for cond, abl in {"cycle": abl_cycle, "magnitude": abl_mag,
                                  "random": abl_rnd}.items():
                    biases = ablation_to_bias(abl, n=n, H=H, L=L, device=device)
                    losses[cond] = second_copy_loss(
                        mm.logits(enc, biases=biases),
                        input_ids[s:s + 1], prefix_len, seq_len)

                for cond, val in losses.items():
                    rows.append({"model": model_name, "top_k": tk, "seq": s,
                                 "condition": cond, "second_copy_loss": val,
                                 "k_ablated": k_total})
                print(f"[exp6b] tk={tk} seq {s + 1}/{n_seqs} K={k_total}: "
                      f"un={losses['unmasked']:.4f} cyc={losses['cycle']:.4f} "
                      f"mag={losses['magnitude']:.4f} rnd={losses['random']:.4f}",
                      flush=True)
    finally:
        del model
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f"[exp6b] wrote {len(df)} rows to {out}", flush=True)
    return df


if __name__ == "__main__":
    run()
