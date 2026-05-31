"""Experiment 7: failure prediction from attention topology.

Pre-registered design: docs/superpowers/specs/
2026-05-31-experiment7-failure-prediction-design.md

For each mixed-hop reasoning item: run Qwen2.5-0.5B-Instruct once, record
(a) correctness (does greedy generation contain the gold token), (b) the model's
own confidence (answer-token logit margin = top1 - top2 at the first generated
position), (c) aggregate H1 topology features over all (layer, head), and
(d) first-order attention controls (distance, off-diagonal mass, entropy).

Output (single source of truth): results/exp7_features.parquet, one row per item.
Downstream: compute_stats_exp7 fits nested logistic models; write_findings_exp7
renders FINDINGS_exp7.md.

Memory-safe: CPU only, float32, eager, torch.set_num_threads(2), one model, gc per
item. (The earlier machine crashes were MPS/parallel; this gentle path is stable.)
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import torch

from src.attn_extract import load_named, format_prompt, is_correct
from src.data_gen import build_items
from src.topology import h1_features
from src.induction import attention_distance, offdiag_mass
from src.residual import attention_entropy

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_PER_FAMILY = 30          # 2 families x 30 x 3 hops = 180 items
HOPS = (1, 2, 3)
TOP_K = 8
SEED = 0


@torch.no_grad()
def confidence_margin(model, tok, device, text: str) -> float:
    """Answer-token logit margin: top1 - top2 logit at the first generated position.

    A scalar proxy for the model's own confidence in its answer, independent of
    topology. Uses the prompt's final-position next-token logits (greedy answer).
    """
    enc = tok(format_prompt(tok, text), return_tensors="pt").to(device)
    out = model(**enc)
    logits = out.logits[0, -1]                  # next-token logits at end of prompt
    top2 = torch.topk(logits, 2).values
    return float((top2[0] - top2[1]).item())


@torch.no_grad()
def topo_features(model, tok, device, text: str, top_k: int) -> dict:
    """Aggregate H1 features over all (layer, head) for one item."""
    enc = tok(format_prompt(tok, text), return_tensors="pt").to(device)
    out = model(**enc, output_attentions=True)
    atts = [a[0].float().cpu().numpy() for a in out.attentions]  # each [H, n, n]
    tot, mx, nontriv, dist, off, ent = [], [], 0, [], [], []
    n_heads = 0
    for layer in atts:
        for h in range(layer.shape[0]):
            A = layer[h]
            f = h1_features(A, top_k=top_k)
            tot.append(f["total_persistence"])
            mx.append(f["max_persistence"])
            nontriv += 1 if f["total_persistence"] > 0 else 0
            dist.append(attention_distance(A))
            off.append(offdiag_mass(A))
            ent.append(attention_entropy(A))
            n_heads += 1
    return {
        "topo_mean_persist": float(np.mean(tot)),
        "topo_max_persist": float(np.max(mx)),
        "topo_frac_nontrivial": float(nontriv / n_heads),
        "ctrl_attn_distance": float(np.mean(dist)),
        "ctrl_offdiag_mass": float(np.mean(off)),
        "ctrl_attn_entropy": float(np.mean(ent)),
    }


def run(out="results/exp7_features.parquet", model_name=MODEL,
        n_per_family=N_PER_FAMILY, hops=HOPS, top_k=TOP_K, seed=SEED):
    torch.set_num_threads(2)
    os.makedirs("results", exist_ok=True)

    items = build_items(n_per_family=n_per_family, hops=hops, seed=seed)
    print(f"[exp7] {len(items)} items (n_per_family={n_per_family}, hops={hops})",
          flush=True)
    print(f"[exp7] loading {model_name} (cpu)...", flush=True)
    model, tok, device = load_named(model_name, device="cpu")
    rows = []
    try:
        for k, it in enumerate(items):
            correct, ans = is_correct(model, tok, device, it.prompt, it.gold)
            conf = confidence_margin(model, tok, device, it.prompt)
            feats = topo_features(model, tok, device, it.prompt, top_k)
            rows.append({
                "id": it.id, "family": it.family, "hop": it.hop,
                "is_correct": int(bool(correct)), "confidence_margin": conf,
                **feats,
            })
            if (k + 1) % 10 == 0 or k == 0:
                acc = np.mean([r["is_correct"] for r in rows])
                print(f"[exp7] {k + 1}/{len(items)} done; running acc={acc:.3f} "
                      f"(last: hop{it.hop} correct={int(bool(correct))})", flush=True)
            gc.collect()
    finally:
        del model
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_parquet(out)
    acc = df["is_correct"].mean()
    print(f"[exp7] wrote {len(df)} rows to {out}; overall acc={acc:.3f}", flush=True)
    print(f"[exp7] acc by hop:\n{df.groupby('hop')['is_correct'].mean()}", flush=True)
    return df


if __name__ == "__main__":
    run()
