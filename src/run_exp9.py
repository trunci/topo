"""Experiment 9: topological attribution + causal validation.

Pre-registered: docs/superpowers/specs/2026-05-31-experiment9-topological-attribution-design.md

Within-example, per-edge (so sequence length cannot confound). For each induction head
on a fixed-length repeated-random sequence we assign every directed causal edge (q>k) a
saliency and ask: does the saliency rank the GROUND-TRUTH induction copy edge
(i -> i-S+1) above other edges, and does it beat the attention-magnitude baseline?

Saliencies:
  * cycle_participation  -- 1 if {q,k} is a critical 1-cell (Morse), else 0
  * sheaf_discord        -- per-edge ||O_v x_v - O_u x_u||^2 over residual-stream sheaf
  * magnitude (baseline) -- raw A[q,k]

Part A -> results/exp9_attribution.parquet (per head x seq x saliency: ROC-AUC, p@k).
Part B -> results/exp9_ablation.parquet  (reuse Exp 6 directed ablation; damage by saliency).

Memory-safe: CPU, float32, eager, threads=2, one model at a time, gc per item.
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.attn_extract import load_named
from src.induction import make_repeat_batch, induction_score
from src.topology import symmetrize, sparsify
from src.morse import critical_cycle_edges
from src.sheaf import edge_discord, local_pca_frame
from src.run_exp5 import second_copy_loss, _head_induction_scores
from src.run_exp6 import directed_causal_edges, select_magnitude_directed, select_random_directed

MODELS = ["gpt2", "distilgpt2"]
SEQ_LEN = 25
PREFIX_LEN = 5
N_SEQS = 12
TOP_K = 16          # adequate budget (Exp 6b lesson)
SHEAF_DIM = 4
K_HEADS = 10
SEED = 0


def _pca_reduce(H, d):
    if H.shape[1] <= d:
        out = np.zeros((H.shape[0], d)); out[:, :H.shape[1]] = H; return out
    Hc = H - H.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Hc, full_matrices=False)
    return Hc @ vt[:d].T


def _node_frames(X, W, d):
    n = X.shape[0]; O = {}
    for v in range(n):
        nbrs = np.flatnonzero(W[v] > 0.0)
        idx = np.concatenate([[v], nbrs]) if nbrs.size else np.array([v])
        O[v] = local_pca_frame(X[idx], d)
    return O


def ground_truth_copy_edges(n, seq_len, prefix_len):
    """Directed induction copy edges (q -> k) for second-copy query positions."""
    gt = set()
    for q in range(prefix_len + seq_len, n):
        k = q - seq_len + 1
        if 0 <= k < q:
            gt.add((q, k))
    return gt


def attribution_for_head(A, X, top_k, d, seq_len, prefix_len):
    """Per-saliency ROC-AUC + precision@k at recovering ground-truth copy edges."""
    n = A.shape[0]
    cands = directed_causal_edges(A)               # all (q>k)
    gt = ground_truth_copy_edges(n, seq_len, prefix_len)
    labels = np.array([1 if e in gt else 0 for e in cands])
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None                                # degenerate, skip

    # saliencies over directed candidate edges
    W = sparsify(symmetrize(A), top_k=top_k)
    crit = critical_cycle_edges(A, top_k=top_k)    # set of frozenset{i,j} (undirected)
    O = _node_frames(X, W, d)
    sal = {"magnitude": [], "cycle_participation": [], "sheaf_discord": []}
    for (q, k) in cands:
        sal["magnitude"].append(float(A[q, k]))
        sal["cycle_participation"].append(1.0 if frozenset((q, k)) in crit else 0.0)
        sal["sheaf_discord"].append(edge_discord(O, X, q, k))
    kk = int(labels.sum())
    out = {}
    for name, vals in sal.items():
        vals = np.asarray(vals, dtype=float)
        try:
            auc = float(roc_auc_score(labels, vals))
        except ValueError:
            auc = float("nan")
        topk_idx = np.argsort(-vals)[:kk]
        pk = float(labels[topk_idx].mean())
        out[name] = {"auc": auc, "p_at_k": pk}
    return out


def run(out_attr="results/exp9_attribution.parquet",
        out_abl="results/exp9_ablation.parquet", models=MODELS, seq_len=SEQ_LEN,
        prefix_len=PREFIX_LEN, n_seqs=N_SEQS, top_k=TOP_K, d=SHEAF_DIM,
        k_heads=K_HEADS, seed=SEED):
    torch.set_num_threads(2)
    os.makedirs("results", exist_ok=True)
    from src.pruned_forward import MaskedModel, ablation_to_bias

    attr_rows, abl_rows = [], []
    for model_name in models:
        print(f"[exp9] loading {model_name} (cpu)...", flush=True)
        model, tok, device = load_named(model_name, device="cpu")
        try:
            L, H, vocab = model.config.n_layer, model.config.n_head, model.config.vocab_size
            mm = MaskedModel(model)
            ids = torch.tensor(make_repeat_batch(vocab, seq_len, prefix_len, n_seqs, seed=seed),
                               dtype=torch.long, device=device)
            n = ids.shape[1]
            atts, Xs = [], []
            with torch.no_grad():
                for s in range(n_seqs):
                    o = model(ids[s:s+1], output_attentions=True, output_hidden_states=True)
                    atts.append(np.stack([a[0].float().cpu().numpy() for a in o.attentions]))
                    Xs.append(_pca_reduce(o.hidden_states[-1][0].float().cpu().numpy(), d))
                    print(f"[exp9] {model_name}: captured {s+1}/{n_seqs}", flush=True)

            scores = _head_induction_scores(atts, L, H, seq_len, prefix_len)
            heads = [lh for lh, _ in sorted(scores.items(), key=lambda kv: kv[1],
                                            reverse=True)[:k_heads]]
            print(f"[exp9] {model_name}: induction heads {heads}", flush=True)

            # --- Part A: attribution AUC / p@k per head x seq ---
            for s in range(n_seqs):
                for (l, h) in heads:
                    res = attribution_for_head(atts[s][l, h], Xs[s], top_k, d,
                                               seq_len, prefix_len)
                    if res is None:
                        continue
                    for sal_name, m in res.items():
                        attr_rows.append({"model": model_name, "seq": s, "layer": l,
                                          "head": h, "saliency": sal_name,
                                          "auc": m["auc"], "p_at_k": m["p_at_k"]})

            # --- Part B: causal ablation, salient vs magnitude vs random ---
            for s in range(n_seqs):
                enc = {"input_ids": ids[s:s+1]}
                loss_un = second_copy_loss(mm.logits(enc, biases=None), ids[s:s+1],
                                           prefix_len, seq_len)
                # build per-condition ablate sets across all induction heads
                cond_sets = {"cycle": {}, "sheaf": {}, "magnitude": {}, "random": {}}
                k_total = 0
                for (l, h) in heads:
                    A = atts[s][l, h]
                    cands = directed_causal_edges(A)
                    W = sparsify(symmetrize(A), top_k=top_k)
                    crit = critical_cycle_edges(A, top_k=top_k)
                    O = _node_frames(Xs[s], W, d)
                    # rank candidates by each saliency, take top-m where m = #cycle edges among candidates
                    cyc_dir = [(q, k) for (q, k) in cands if frozenset((q, k)) in crit]
                    m = len(cyc_dir)
                    if m == 0:
                        continue
                    k_total += m
                    cond_sets["cycle"][(l, h)] = set(cyc_dir)
                    disc_rank = sorted(cands, key=lambda e: edge_discord(O, Xs[s], e[0], e[1]),
                                       reverse=True)[:m]
                    cond_sets["sheaf"][(l, h)] = set(disc_rank)
                    cond_sets["magnitude"][(l, h)] = select_magnitude_directed(A, cands, m)
                    cond_sets["random"][(l, h)] = select_random_directed(
                        cands, m, seed=seed*100000 + s*1000 + l*100 + h)
                losses = {"unmasked": loss_un}
                for cond, sets in cond_sets.items():
                    biases = ablation_to_bias(sets, n=n, H=H, L=L, device=device)
                    losses[cond] = second_copy_loss(mm.logits(enc, biases=biases),
                                                    ids[s:s+1], prefix_len, seq_len)
                for cond, val in losses.items():
                    abl_rows.append({"model": model_name, "seq": s, "condition": cond,
                                     "second_copy_loss": val, "k_ablated": k_total})
                print(f"[exp9] {model_name} ablate seq {s+1}/{n_seqs} K={k_total}: "
                      f"un={losses['unmasked']:.4f} cyc={losses['cycle']:.4f} "
                      f"shf={losses['sheaf']:.4f} mag={losses['magnitude']:.4f} "
                      f"rnd={losses['random']:.4f}", flush=True)
        finally:
            del model; gc.collect()

    pd.DataFrame(attr_rows).to_parquet(out_attr)
    pd.DataFrame(abl_rows).to_parquet(out_abl)
    print(f"[exp9] wrote {len(attr_rows)} attribution rows, {len(abl_rows)} ablation rows",
          flush=True)
    return attr_rows, abl_rows


if __name__ == "__main__":
    run()
