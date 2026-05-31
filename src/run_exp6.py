"""Experiment 6: stronger directed causal test of induction cycle edges.

Pre-registered design: docs/superpowers/specs/
2026-05-31-experiment6-stronger-causal-design.md

Resolves Exp 5's underpowered E1. The instrument is now:
* DIRECTED  -- the induction edge is q>k (later position attends back), so we
  ablate directed (q, k) pairs, not symmetric undirected edges.
* EDGE-EXACT -- ablation_to_bias removes exactly the named edges and keeps all
  else (no common-mode nuking of the rest of the graph as in E1).
* GAP-CONSTRAINED -- only cycle edges at gap ~ S (the copy distance) are the
  treatment, mirroring the E2-GREEN finding.
* CUMULATIVE -- ablate across ALL induction heads at once (kills cross-head
  redundancy).

Conditions, budget-matched per head (m = #directed gap~S cycle edges of that head):
* cycle     -- ablate the m directed gap~S cycle edges.
* magnitude -- ablate the m highest-weight directed causal (q>k) edges (any gap).
* random    -- ablate m random directed causal (q>k) edges (seeded).

Output (single source of truth = parquet): results/exp6_ablation.parquet, one row
per (model, seq, condition) with second_copy_loss and the total ablated count K.

Memory-safe: CPU only, float32, eager, torch.set_num_threads(2), one model at a
time, gc between models. (Earlier MPS/large/parallel runs crash-restarted the box.)
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import torch

from src.attn_extract import load_named
from src.topology import symmetrize, sparsify
from src.morse import critical_cycle_edges
from src.induction import make_repeat_batch, induction_score
from src.run_exp5 import second_copy_loss, _head_induction_scores

MODELS = ["gpt2", "distilgpt2"]
SEQ_LEN = 25          # S = induction copy distance
PREFIX_LEN = 5
N_SEQS = 12
TOP_K = 8             # per-head sparsification budget (same as Exp 5)
GAP_TOL = 2           # |gap - S| <= GAP_TOL counts as a copy edge
K_HEADS = 10          # number of top induction heads
SEED = 0


def directed_gap_cycle_edges(A, seq_len, gap_tol, top_k):
    """Directed (q, k) cycle edges at gap ~ S for one head.

    critical_cycle_edges returns undirected frozenset{i, j}. Keep those with
    |i - j| within gap_tol of seq_len, and direct them q = max, k = min (the
    later position attends back to the earlier one -- the induction direction).
    """
    out = set()
    for e in critical_cycle_edges(A, top_k=top_k):
        i, j = tuple(e)
        if abs(abs(i - j) - seq_len) <= gap_tol:
            out.add((max(i, j), min(i, j)))
    return out


def directed_causal_edges(A):
    """All directed causal pairs (q, k) with q > k (later attends earlier)."""
    n = A.shape[0]
    return [(q, k) for q in range(n) for k in range(q)]


def select_magnitude_directed(A, candidates, m):
    """The m highest-weight directed edges by raw attention A[q, k]."""
    ranked = sorted(candidates, key=lambda e: A[e[0], e[1]], reverse=True)
    return set(ranked[:m])


def select_random_directed(candidates, m, seed):
    """m random directed edges (seeded, deterministic)."""
    rng = np.random.default_rng(seed)
    cand = sorted(candidates)  # stable order
    if m >= len(cand):
        return set(cand)
    idx = rng.choice(len(cand), size=m, replace=False)
    return {cand[i] for i in idx}


def run(out="results/exp6_ablation.parquet", models=MODELS, seq_len=SEQ_LEN,
        prefix_len=PREFIX_LEN, n_seqs=N_SEQS, top_k=TOP_K, gap_tol=GAP_TOL,
        k_heads=K_HEADS, seed=SEED):
    torch.set_num_threads(2)
    os.makedirs("results", exist_ok=True)
    from src.pruned_forward import MaskedModel, ablation_to_bias

    rows = []
    for model_name in models:
        print(f"[exp6] loading {model_name} (cpu)...", flush=True)
        model, tok, device = load_named(model_name, device="cpu")
        try:
            L = model.config.n_layer
            H = model.config.n_head
            vocab = model.config.vocab_size
            mm = MaskedModel(model)

            batch = make_repeat_batch(vocab, seq_len, prefix_len, n_seqs, seed=seed)
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            n = input_ids.shape[1]
            print(f"[exp6] {model_name}: n_seqs={n_seqs} n={n} S={seq_len} "
                  f"L={L} H={H}", flush=True)

            # capture attentions, one sequence at a time
            attentions = []
            with torch.no_grad():
                for s in range(n_seqs):
                    out_s = model(input_ids[s:s + 1], output_attentions=True)
                    att = np.stack([a[0].float().cpu().numpy()
                                    for a in out_s.attentions])
                    attentions.append(att)  # [L, H, n, n]
                    print(f"[exp6] {model_name}: captured seq {s + 1}/{n_seqs}",
                          flush=True)

            # pick induction heads (mean score over batch)
            scores = _head_induction_scores(attentions, L, H, seq_len, prefix_len)
            ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            induction_heads = [lh for lh, _ in ordered[:k_heads]]
            print(f"[exp6] {model_name}: induction heads {induction_heads}",
                  flush=True)

            # ablation per sequence
            for s in range(n_seqs):
                att = attentions[s]
                enc = {"input_ids": input_ids[s:s + 1]}

                abl_cycle, abl_mag, abl_rnd = {}, {}, {}
                k_total = 0
                for (l, h) in induction_heads:
                    A = att[l, h]
                    cyc = directed_gap_cycle_edges(A, seq_len, gap_tol, top_k)
                    m = len(cyc)
                    if m == 0:
                        continue
                    k_total += m
                    cand = directed_causal_edges(A)
                    abl_cycle[(l, h)] = cyc
                    abl_mag[(l, h)] = select_magnitude_directed(A, cand, m)
                    abl_rnd[(l, h)] = select_random_directed(
                        cand, m, seed=seed * 100000 + s * 1000 + l * 100 + h)

                loss_un = second_copy_loss(mm.logits(enc, biases=None),
                                           input_ids[s:s + 1], prefix_len, seq_len)
                conds = {"cycle": abl_cycle, "magnitude": abl_mag, "random": abl_rnd}
                losses = {"unmasked": loss_un}
                for cond, abl in conds.items():
                    biases = ablation_to_bias(abl, n=n, H=H, L=L, device=device)
                    losses[cond] = second_copy_loss(
                        mm.logits(enc, biases=biases),
                        input_ids[s:s + 1], prefix_len, seq_len)

                for cond, val in losses.items():
                    rows.append({"model": model_name, "seq": s, "condition": cond,
                                 "second_copy_loss": val, "k_ablated": k_total})
                print(f"[exp6] {model_name} seq {s + 1}/{n_seqs} K={k_total}: "
                      f"un={losses['unmasked']:.4f} cyc={losses['cycle']:.4f} "
                      f"mag={losses['magnitude']:.4f} rnd={losses['random']:.4f}",
                      flush=True)
        finally:
            del model
            gc.collect()

    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f"[exp6] wrote {len(df)} rows to {out}", flush=True)
    return df


if __name__ == "__main__":
    run()
