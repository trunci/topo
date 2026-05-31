"""Experiment 10: topological attribution + causal validation on IOI.

Pre-registered: docs/superpowers/specs/2026-05-31-experiment10-ioi-attribution-design.md

Mirrors Exp 9 but on IOI / name-mover heads in GPT-2 small, where attention is more
diffuse (verified) so the magnitude baseline is handicapped.

Per name-mover head, per prompt: score every directed edge into END (END -> k) by each
saliency; ROC-AUC + p@1 at recovering the ground-truth name-mover edge (END -> IO_pos).
Part B ablates top-salient/magnitude/random edges and reads the IO-S logit margin.

Single source of truth: results/exp10_attribution.parquet, results/exp10_ablation.parquet.
Memory-safe: CPU, float32, eager, threads=2, one model, gc per prompt.
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.attn_extract import load_named
from src.data_ioi import build_ioi_items, ioi_positions, io_s_token_ids
from src.topology import symmetrize, sparsify
from src.morse import critical_cycle_edges
from src.sheaf import edge_discord, local_pca_frame
from src.run_exp6 import select_magnitude_directed, select_random_directed

MODEL = "gpt2"
N_PROMPTS = 48
TOP_K = 16
SHEAF_DIM = 4
K_HEADS = 8
SEED = 0


def _pca_reduce(Hm, d):
    if Hm.shape[1] <= d:
        out = np.zeros((Hm.shape[0], d)); out[:, :Hm.shape[1]] = Hm; return out
    Hc = Hm - Hm.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Hc, full_matrices=False)
    return Hc @ vt[:d].T


def _node_frames(X, W, d):
    n = X.shape[0]; O = {}
    for v in range(n):
        nbrs = np.flatnonzero(W[v] > 0.0)
        idx = np.concatenate([[v], nbrs]) if nbrs.size else np.array([v])
        O[v] = local_pca_frame(X[idx], d)
    return O


def end_in_edges(end_pos):
    """Directed candidate edges into END: (END -> k) for k < END."""
    return [(end_pos, k) for k in range(end_pos)]


def attribution_for_head(A, X, end_pos, io_pos, top_k, d):
    """Per-saliency ROC-AUC + p@1 at recovering the ground-truth (END->io_pos) edge."""
    cands = end_in_edges(end_pos)
    labels = np.array([1 if k == io_pos else 0 for (_, k) in cands])
    if labels.sum() != 1:
        return None
    W = sparsify(symmetrize(A), top_k=top_k)
    crit = critical_cycle_edges(A, top_k=top_k)
    O = _node_frames(X, W, d)
    sal = {"magnitude": [], "cycle_participation": [], "sheaf_discord": []}
    for (q, k) in cands:
        sal["magnitude"].append(float(A[q, k]))
        sal["cycle_participation"].append(1.0 if frozenset((q, k)) in crit else 0.0)
        sal["sheaf_discord"].append(edge_discord(O, X, q, k))
    out = {}
    for name, vals in sal.items():
        vals = np.asarray(vals, float)
        try:
            auc = float(roc_auc_score(labels, vals))
        except ValueError:
            auc = float("nan")
        p_at_1 = float(labels[int(np.argmax(vals))] == 1)
        out[name] = {"auc": auc, "p_at_1": p_at_1}
    return out


def run(out_attr="results/exp10_attribution.parquet",
        out_abl="results/exp10_ablation.parquet", model_name=MODEL,
        n_prompts=N_PROMPTS, top_k=TOP_K, d=SHEAF_DIM, k_heads=K_HEADS, seed=SEED):
    torch.set_num_threads(2)
    os.makedirs("results", exist_ok=True)
    from src.pruned_forward import MaskedModel, ablation_to_bias

    print(f"[exp10] loading {model_name} (cpu)...", flush=True)
    model, tok, device = load_named(model_name, device="cpu")
    attr_rows, abl_rows = [], []
    try:
        L, H = model.config.n_layer, model.config.n_head
        mm = MaskedModel(model)
        items = build_ioi_items(tok, n=n_prompts, seed=seed)

        # capture attentions + hidden states + positions per prompt
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
        print(f"[exp10] captured {len(data)} prompts", flush=True)

        # name-mover heads: top-K by attention(END -> io_pos) averaged
        nm = {(l, h): 0.0 for l in range(L) for h in range(H)}
        for (_, _, atts, _, io_pos, end_pos, _, _) in data:
            for l in range(L):
                for h in range(H):
                    nm[(l, h)] += atts[l, h, end_pos, io_pos]
        heads = [lh for lh, _ in sorted(nm.items(), key=lambda kv: kv[1], reverse=True)[:k_heads]]
        print(f"[exp10] name-mover heads {heads}", flush=True)

        # Part A: attribution
        for (pid, _, atts, X, io_pos, end_pos, _, _) in data:
            for (l, h) in heads:
                res = attribution_for_head(atts[l, h], X, end_pos, io_pos, top_k, d)
                if res is None:
                    continue
                for sal, m in res.items():
                    attr_rows.append({"model": model_name, "prompt": pid, "layer": l,
                                      "head": h, "saliency": sal,
                                      "auc": m["auc"], "p_at_1": m["p_at_1"]})

        # Part B: causal ablation, IO-S margin readout
        def io_margin(enc, biases, io_id, s_id, end_pos):
            logits = mm.logits(enc, biases=biases)[0, end_pos]
            return float(logits[io_id] - logits[s_id])

        for (pid, enc, atts, X, io_pos, end_pos, io_id, s_id) in data:
            base = io_margin(enc, None, io_id, s_id, end_pos)
            cond_sets = {"cycle": {}, "sheaf": {}, "magnitude": {}, "random": {}}
            k_total = 0
            for (l, h) in heads:
                A = atts[l, h]
                cands = end_in_edges(end_pos)
                W = sparsify(symmetrize(A), top_k=top_k)
                crit = critical_cycle_edges(A, top_k=top_k)
                O = _node_frames(X, W, d)
                cyc_dir = [(q, k) for (q, k) in cands if frozenset((q, k)) in crit]
                m = len(cyc_dir)
                if m == 0:
                    continue
                k_total += m
                cond_sets["cycle"][(l, h)] = set(cyc_dir)
                cond_sets["sheaf"][(l, h)] = set(sorted(
                    cands, key=lambda e: edge_discord(O, X, e[0], e[1]), reverse=True)[:m])
                cond_sets["magnitude"][(l, h)] = select_magnitude_directed(A, cands, m)
                cond_sets["random"][(l, h)] = select_random_directed(
                    cands, m, seed=seed*100000 + hash(pid) % 1000 + l*100 + h)
            margins = {"unmasked": base}
            for cond, sets in cond_sets.items():
                biases = ablation_to_bias(sets, n=atts.shape[-1], H=H, L=L, device=device)
                margins[cond] = io_margin(enc, biases, io_id, s_id, end_pos)
            for cond, val in margins.items():
                abl_rows.append({"model": model_name, "prompt": pid, "condition": cond,
                                 "io_margin": val, "k_ablated": k_total})
            print(f"[exp10] ablate {pid} K={k_total}: un={base:.3f} "
                  f"cyc={margins['cycle']:.3f} shf={margins['sheaf']:.3f} "
                  f"mag={margins['magnitude']:.3f} rnd={margins['random']:.3f}", flush=True)
    finally:
        del model; gc.collect()

    pd.DataFrame(attr_rows).to_parquet(out_attr)
    pd.DataFrame(abl_rows).to_parquet(out_abl)
    print(f"[exp10] wrote {len(attr_rows)} attribution, {len(abl_rows)} ablation rows",
          flush=True)
    return attr_rows, abl_rows


if __name__ == "__main__":
    run()
