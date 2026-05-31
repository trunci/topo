"""Experiment 11: clean IOI causal readout (fixes Exp 10 Part B's compromised instrument).

Exp 10 Part B read the IO-S logit *margin* after ablating salient edges into END.
That instrument was compromised: name-mover heads' edges feed BOTH the IO and the S
logit, so ablation often RAISED the margin (every median damage was <=0). The margin
cannot cleanly score "necessity of the IO-moving edge."

This experiment swaps the readout for the IO token's **log-probability** at END
(no S term), so "damage = base_logprob - ablated_logprob" is positive exactly when
the ablated edges were necessary for recalling the IO name. It also adds a
ground-truth instrument check: a `gt_io` condition that ablates EXACTLY the END->IO
edge in every name-mover head -- this MUST produce large positive damage if the
readout is sound.

Conditions per prompt: unmasked (reference), gt_io (instrument check), and the four
saliency-selected edge sets from Exp 10 (cycle, sheaf, magnitude, random), each
ablating the same number of edges per head.

Single source of truth: results/exp11_ablation.parquet. Memory-safe: CPU, float32,
eager, threads=2, one model, gc per prompt. Reuses Exp 10's capture + selection.
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import torch

from src.attn_extract import load_named
from src.data_ioi import build_ioi_items, ioi_positions, io_s_token_ids
from src.topology import symmetrize, sparsify
from src.morse import critical_cycle_edges
from src.sheaf import edge_discord
from src.run_exp6 import select_magnitude_directed, select_random_directed
from src.run_exp10 import end_in_edges, _node_frames, _pca_reduce

MODEL = "gpt2"
N_PROMPTS = 48
TOP_K = 16
SHEAF_DIM = 4
K_HEADS = 8
SEED = 0


def run(out_abl="results/exp11_ablation.parquet", model_name=MODEL,
        n_prompts=N_PROMPTS, top_k=TOP_K, d=SHEAF_DIM, k_heads=K_HEADS, seed=SEED):
    torch.set_num_threads(2)
    os.makedirs("results", exist_ok=True)
    from src.pruned_forward import MaskedModel, ablation_to_bias

    print(f"[exp11] loading {model_name} (cpu)...", flush=True)
    model, tok, device = load_named(model_name, device="cpu")
    abl_rows = []
    try:
        L, H = model.config.n_layer, model.config.n_head
        mm = MaskedModel(model)
        items = build_ioi_items(tok, n=n_prompts, seed=seed)

        data = []
        with torch.no_grad():
            for it in items:
                pos = ioi_positions(tok, it)
                if pos is None:
                    continue
                io_pos, end_pos = pos
                enc = tok(it.text, return_tensors="pt").to(device)
                o = model(**enc, output_attentions=True, output_hidden_states=True)
                atts = np.stack([a[0].float().cpu().numpy() for a in o.attentions])
                X = _pca_reduce(o.hidden_states[-1][0].float().cpu().numpy(), d)
                io_id, s_id = io_s_token_ids(tok, it)
                data.append((it.id, enc, atts, X, io_pos, end_pos, io_id, s_id))
        print(f"[exp11] captured {len(data)} prompts", flush=True)

        # name-mover heads: top-K by attention(END -> io_pos) averaged (same as Exp 10)
        nm = {(l, h): 0.0 for l in range(L) for h in range(H)}
        for (_, _, atts, _, io_pos, end_pos, _, _) in data:
            for l in range(L):
                for h in range(H):
                    nm[(l, h)] += atts[l, h, end_pos, io_pos]
        heads = [lh for lh, _ in sorted(nm.items(), key=lambda kv: kv[1], reverse=True)[:k_heads]]
        print(f"[exp11] name-mover heads {heads}", flush=True)

        def io_logprob(enc, biases, io_id, end_pos):
            logits = mm.logits(enc, biases=biases)[0, end_pos]
            return float(torch.log_softmax(logits, dim=-1)[io_id])

        for (pid, enc, atts, X, io_pos, end_pos, io_id, s_id) in data:
            base = io_logprob(enc, None, io_id, end_pos)
            cond_sets = {"gt_io": {}, "cycle": {}, "sheaf": {}, "magnitude": {}, "random": {}}
            k_total = 0
            for (l, h) in heads:
                A = atts[l, h]
                cands = end_in_edges(end_pos)
                W = sparsify(symmetrize(A), top_k=top_k)
                crit = critical_cycle_edges(A, top_k=top_k)
                O = _node_frames(X, W, d)
                cyc_dir = [(q, k) for (q, k) in cands if frozenset((q, k)) in crit]
                m = len(cyc_dir)
                # ground-truth edge present in every head (instrument check)
                cond_sets["gt_io"][(l, h)] = {(end_pos, io_pos)}
                if m == 0:
                    continue
                k_total += m
                cond_sets["cycle"][(l, h)] = set(cyc_dir)
                cond_sets["sheaf"][(l, h)] = set(sorted(
                    cands, key=lambda e: edge_discord(O, X, e[0], e[1]), reverse=True)[:m])
                cond_sets["magnitude"][(l, h)] = select_magnitude_directed(A, cands, m)
                cond_sets["random"][(l, h)] = select_random_directed(
                    cands, m, seed=seed * 100000 + hash(pid) % 1000 + l * 100 + h)
            margins = {"unmasked": base}
            for cond, sets in cond_sets.items():
                biases = ablation_to_bias(sets, n=atts.shape[-1], H=H, L=L, device=device)
                margins[cond] = io_logprob(enc, biases, io_id, end_pos)
            for cond, val in margins.items():
                abl_rows.append({"model": model_name, "prompt": pid, "condition": cond,
                                 "io_logprob": val, "k_ablated": k_total})
            print(f"[exp11] {pid} K={k_total}: un={base:.3f} gt={margins['gt_io']:.3f} "
                  f"cyc={margins['cycle']:.3f} shf={margins['sheaf']:.3f} "
                  f"mag={margins['magnitude']:.3f} rnd={margins['random']:.3f}", flush=True)
    finally:
        del model
        gc.collect()

    pd.DataFrame(abl_rows).to_parquet(out_abl)
    print(f"[exp11] wrote {len(abl_rows)} ablation rows", flush=True)
    return abl_rows


if __name__ == "__main__":
    run()
