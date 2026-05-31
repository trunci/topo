"""Experiment 5: causal ablation (E1) + cycle inspection (E2) for induction.

Pre-registered design: docs/superpowers/specs/
2026-05-31-experiment5-causal-and-cycles-design.md

Model: gpt2 (CPU, float32, eager). One model in memory; torch threads capped.
Inputs: repeated-random sequences [prefix; block S; block S] (n_seqs<=8, S=25,
prefix_len=5) -- the same substrate as Exp 2/3/4.

Two outputs (single source of truth = the parquets; stats computed downstream):
* results/exp5_cycles.parquet  -- E2: one row per critical cycle edge of every
  induction / non-induction head on every sequence, with its gap |i-j|.
* results/exp5_ablation.parquet -- E1: one row per (sequence, condition) with the
  model's second-copy next-token loss; conditions = unmasked/cycle/random/magnitude.

Memory-safe: CPU only, float32, eager attention, torch.set_num_threads(2), modest
batch. No MPS, no parallel heavy work. (Earlier MPS/large/parallel runs crash-
restarted the machine.)
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

MODEL = "gpt2"
SEQ_LEN = 25          # S = induction copy distance
PREFIX_LEN = 5
N_SEQS = 8
TOP_K = 8             # per-head sparsification budget
K_HEADS = 10          # top-K induction heads / bottom-K non-induction heads
SEED = 0


def second_copy_loss(logits, input_ids, prefix_len, seq_len) -> float:
    """Mean next-token cross-entropy on the second-copy positions.

    logits: [1, n, vocab]; input_ids: [1, n]. The repeat row is
    [prefix(prefix_len); block(S); block(S)]; second-copy positions are
    prefix_len+S .. n-1. For target position t we score logits[0, t-1]
    predicting input_ids[0, t]. Returns the mean CE over second-copy targets.
    """
    n = input_ids.shape[1]
    start = prefix_len + seq_len
    logp = torch.log_softmax(logits[0], dim=-1)  # [n, vocab]
    losses = []
    for t in range(start, n):
        gold = int(input_ids[0, t].item())
        losses.append(float(-logp[t - 1, gold].item()))
    return float(np.mean(losses)) if losses else float("nan")


def all_candidate_edges(W) -> set:
    """Upper-triangular (i<j) positive-weight edges of a sparsified graph."""
    n = W.shape[0]
    iu = np.triu_indices(n, k=1)
    return {frozenset((int(i), int(j))) for i, j in zip(*iu) if W[i, j] > 0.0}


def _select_edges_by_magnitude(W, candidates, m):
    """The m highest-weight edges among `candidates`."""
    ranked = sorted(candidates, key=lambda e: W[tuple(e)], reverse=True)
    return set(ranked[:m])


def _select_edges_random(candidates, m, seed):
    """m random edges from `candidates` (seeded, deterministic)."""
    rng = np.random.default_rng(seed)
    cand = sorted(candidates, key=lambda e: tuple(sorted(e)))  # stable order
    if m >= len(cand):
        return set(cand)
    idx = rng.choice(len(cand), size=m, replace=False)
    return {cand[i] for i in idx}


def _head_induction_scores(attentions, L, H, seq_len, prefix_len):
    """Mean induction score per (layer, head) over a batch of attentions.

    attentions: list over sequences of [L, H, n, n] numpy arrays.
    Returns dict (layer, head) -> mean induction score.
    """
    sums = {(l, h): 0.0 for l in range(L) for h in range(H)}
    for att in attentions:
        for l in range(L):
            for h in range(H):
                sums[(l, h)] += induction_score(att[l, h], seq_len, prefix_len)
    n = len(attentions)
    return {lh: s / n for lh, s in sums.items()}


def run(out_cycles="results/exp5_cycles.parquet",
        out_ablation="results/exp5_ablation.parquet",
        model_name=MODEL, seq_len=SEQ_LEN, prefix_len=PREFIX_LEN,
        n_seqs=N_SEQS, top_k=TOP_K, k_heads=K_HEADS, seed=SEED):
    torch.set_num_threads(2)
    os.makedirs("results", exist_ok=True)

    from src.pruned_forward import MaskedModel, keepsets_to_bias

    print(f"[exp5] loading {model_name} (cpu)...", flush=True)
    model, tok, device = load_named(model_name, device="cpu")
    try:
        L = model.config.n_layer
        H = model.config.n_head
        vocab = model.config.vocab_size
        mm = MaskedModel(model)

        batch = make_repeat_batch(vocab, seq_len, prefix_len, n_seqs, seed=seed)
        input_ids = torch.tensor(batch, dtype=torch.long, device=device)
        n = input_ids.shape[1]
        print(f"[exp5] batch: n_seqs={n_seqs} n={n} S={seq_len} prefix={prefix_len}",
              flush=True)

        # --- capture attentions for every sequence (CPU, one at a time) -------
        attentions = []
        with torch.no_grad():
            for s in range(n_seqs):
                out = model(input_ids[s:s + 1], output_attentions=True)
                att = np.stack([a[0].float().cpu().numpy() for a in out.attentions])
                attentions.append(att)  # [L, H, n, n]
                print(f"[exp5] captured attentions seq {s + 1}/{n_seqs}", flush=True)

        # --- pick induction / non-induction heads (mean score over batch) -----
        scores = _head_induction_scores(attentions, L, H, seq_len, prefix_len)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        induction_heads = [lh for lh, _ in ordered[:k_heads]]
        non_induction_heads = [lh for lh, _ in ordered[-k_heads:]]
        print(f"[exp5] induction heads (top {k_heads}): {induction_heads}", flush=True)
        print(f"[exp5] non-induction heads (bottom {k_heads}): {non_induction_heads}",
              flush=True)

        # --- E2: cycle-edge gaps per head per sequence ------------------------
        cycle_rows = []
        for s in range(n_seqs):
            att = attentions[s]
            for is_ind, heads in ((1, induction_heads), (0, non_induction_heads)):
                for (l, h) in heads:
                    cyc = critical_cycle_edges(att[l, h], top_k=top_k)
                    for e in cyc:
                        i, j = tuple(e)
                        cycle_rows.append({
                            "model": model_name, "seq": s,
                            "layer": int(l), "head": int(h),
                            "is_induction": int(is_ind),
                            "gap": int(abs(i - j)),
                        })
        cycles_df = pd.DataFrame(cycle_rows)
        cycles_df.to_parquet(out_cycles)
        print(f"[exp5] wrote {len(cycles_df)} cycle-edge rows to {out_cycles}",
              flush=True)

        # --- E1: ablation over the K induction heads (all together) ----------
        # Precompute per (seq, head): sparsified W, candidate edges, cycle edges.
        ablation_rows = []
        for s in range(n_seqs):
            att = attentions[s]
            enc = {"input_ids": input_ids[s:s + 1]}

            # build keepsets for each condition
            ks_cycle, ks_random, ks_magnitude = {}, {}, {}
            for (l, h) in induction_heads:
                W = sparsify(symmetrize(att[l, h]), top_k=top_k)
                cand = all_candidate_edges(W)
                cyc = critical_cycle_edges(att[l, h], top_k=top_k) & cand
                m = len(cyc)
                # treatment: ablate the cycle edges -> keep candidates minus cycle
                ks_cycle[(l, h)] = cand - cyc
                # control-magnitude: ablate m highest-weight candidate edges
                mag = _select_edges_by_magnitude(W, cand, m)
                ks_magnitude[(l, h)] = cand - mag
                # control-random: ablate m random candidate edges (seeded)
                rnd = _select_edges_random(
                    cand, m, seed=seed * 100000 + s * 1000 + l * 100 + h)
                ks_random[(l, h)] = cand - rnd

            # unmasked
            logits = mm.logits(enc, biases=None)
            loss_un = second_copy_loss(logits, input_ids[s:s + 1],
                                       prefix_len, seq_len)
            # masked conditions
            conds = {"cycle": ks_cycle, "random": ks_random,
                     "magnitude": ks_magnitude}
            losses = {"unmasked": loss_un}
            for cond, ks in conds.items():
                biases = keepsets_to_bias(ks, n=n, H=H, L=L, device=device)
                logits = mm.logits(enc, biases=biases)
                losses[cond] = second_copy_loss(logits, input_ids[s:s + 1],
                                                prefix_len, seq_len)

            for cond, val in losses.items():
                ablation_rows.append({
                    "model": model_name, "seq": s,
                    "condition": cond, "second_copy_loss": val,
                })
            print(f"[exp5] ablation seq {s + 1}/{n_seqs}: "
                  f"un={losses['unmasked']:.4f} cyc={losses['cycle']:.4f} "
                  f"rnd={losses['random']:.4f} mag={losses['magnitude']:.4f}",
                  flush=True)

        ablation_df = pd.DataFrame(ablation_rows)
        ablation_df.to_parquet(out_ablation)
        print(f"[exp5] wrote {len(ablation_df)} ablation rows to {out_ablation}",
              flush=True)
        return cycles_df, ablation_df
    finally:
        del model
        gc.collect()


if __name__ == "__main__":
    run()
