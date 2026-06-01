"""Experiment 12: distractor-augmented failure prediction.

Pre-registered design: docs/superpowers/specs/
2026-06-01-experiment12-distractor-failure-prediction-design.md

For each item (clean or distracted): run Qwen2.5-1.5B-Instruct once, record
(a) correctness, (b) confidence margin, (c) aggregate H1 topology features,
(d) first-order attention controls, (e) item_type and distractor_name.

Output: results/exp12_features.parquet, one row per item.
Downstream: compute_stats_exp12 -> results/exp12_stats.json
             write_findings_exp12 -> FINDINGS_exp12.md
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import torch

from src.attn_extract import load_named, format_prompt, is_correct
from src.data_gen import build_items_distracted
from src.topology import h1_features
from src.induction import attention_distance, offdiag_mass
from src.residual import attention_entropy

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_PER_FAMILY = 40
HOPS = (1, 2, 3)
TOP_K = 8
SEED = 0


@torch.no_grad()
def confidence_margin(model, tok, device, text: str) -> float:
    enc = tok(format_prompt(tok, text), return_tensors="pt").to(device)
    out = model(**enc)
    logits = out.logits[0, -1]
    top2 = torch.topk(logits, 2).values
    return float((top2[0] - top2[1]).item())


@torch.no_grad()
def topo_features(model, tok, device, text: str, top_k: int) -> dict:
    enc = tok(format_prompt(tok, text), return_tensors="pt").to(device)
    out = model(**enc, output_attentions=True)
    atts = [a[0].float().cpu().numpy() for a in out.attentions]
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


def run(out="results/exp12_features.parquet", model_name=MODEL,
        n_per_family=N_PER_FAMILY, hops=HOPS, top_k=TOP_K, seed=SEED):
    torch.set_num_threads(2)
    os.makedirs("results", exist_ok=True)

    items = build_items_distracted(n_per_family=n_per_family, hops=hops, seed=seed)
    print(f"[exp12] {len(items)} items ({n_per_family} per family per hop per type)",
          flush=True)
    print(f"[exp12] loading {model_name} (cpu, float32)...", flush=True)
    model, tok, device = load_named(model_name, device="cpu")
    rows = []
    try:
        for k, it in enumerate(items):
            correct, ans = is_correct(model, tok, device, it.prompt, it.gold)
            conf = confidence_margin(model, tok, device, it.prompt)
            feats = topo_features(model, tok, device, it.prompt, top_k)
            rows.append({
                "id": it.id,
                "family": it.family,
                "hop": it.hop,
                "item_type": it.item_type,
                "distractor_name": it.distractor_name,
                "is_correct": int(bool(correct)),
                "confidence_margin": conf,
                **feats,
            })
            if (k + 1) % 20 == 0 or k == 0:
                clean = [r for r in rows if r["item_type"] == "clean"]
                dist = [r for r in rows if r["item_type"] == "distracted"]
                acc_c = np.mean([r["is_correct"] for r in clean]) if clean else float("nan")
                acc_d = np.mean([r["is_correct"] for r in dist]) if dist else float("nan")
                print(f"[exp12] {k + 1}/{len(items)} done; "
                      f"acc clean={acc_c:.3f} distracted={acc_d:.3f} "
                      f"(last: {it.item_type} hop{it.hop} correct={int(bool(correct))})",
                      flush=True)
            gc.collect()
    finally:
        del model
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f"[exp12] wrote {len(df)} rows to {out}", flush=True)
    print(f"[exp12] accuracy by hop and type:\n"
          f"{df.groupby(['hop', 'item_type'])['is_correct'].mean().unstack()}", flush=True)
    return df


if __name__ == "__main__":
    run()
