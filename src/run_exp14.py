"""Experiment 14: failure prediction on real QA (HotpotQA bridge).

Tests whether the Exp 13 topology-adds-beyond-confidence result holds on a
real 2-hop QA benchmark, and provides a direct comparison point against TOHA
(Bazarova et al., ACL 2026, AUROC 0.71 on HotpotQA/Mistral-7B).

Model: Qwen2.5-1.5B-Instruct (showed unique topology contribution in Exp 13).
Dataset: HotpotQA validation, bridge questions only, n=200 random sample.
Correctness: gold answer string appears in model's generated text (case-insensitive).

Output: results/exp14_features.parquet
Downstream: compute_stats_exp14 -> results/exp14_stats.json
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import torch

from src.attn_extract import load_named, format_prompt, is_correct
from src.data_hotpotqa import build_hotpot_items
from src.topology import h1_features
from src.induction import attention_distance, offdiag_mass
from src.residual import attention_entropy

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_ITEMS = 200
SEED = 0
TOP_K = 8


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
        "seq_len": int(enc["input_ids"].shape[1]),
    }


def run(out="results/exp14_features.parquet", model_name=MODEL,
        n_items=N_ITEMS, seed=SEED, top_k=TOP_K):
    torch.set_num_threads(2)
    os.makedirs("results", exist_ok=True)

    items = build_hotpot_items(n=n_items, seed=seed)
    print(f"[exp14] {len(items)} HotpotQA bridge items (seed={seed})", flush=True)
    print(f"[exp14] loading {model_name} (cpu)...", flush=True)
    model, tok, device = load_named(model_name, device="cpu")

    rows = []
    try:
        for k, it in enumerate(items):
            correct, ans = is_correct(model, tok, device, it.prompt, it.answer,
                                      max_new_tokens=20)
            conf = confidence_margin(model, tok, device, it.prompt)
            feats = topo_features(model, tok, device, it.prompt, top_k)
            rows.append({
                "id": it.id,
                "level": it.level,
                "answer": it.answer,
                "generated": ans,
                "is_correct": int(bool(correct)),
                "confidence_margin": conf,
                **feats,
            })
            if (k + 1) % 20 == 0 or k == 0:
                acc = np.mean([r["is_correct"] for r in rows])
                print(f"[exp14] {k+1}/{len(items)} done; acc={acc:.3f} "
                      f"(last: {it.level} correct={int(bool(correct))} "
                      f"ans='{it.answer}' gen='{ans[:30]}')", flush=True)
            gc.collect()
    finally:
        del model
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f"[exp14] wrote {len(df)} rows to {out}; "
          f"overall acc={df['is_correct'].mean():.3f}", flush=True)
    print(f"[exp14] acc by level:\n{df.groupby('level')['is_correct'].mean()}", flush=True)
    return df


if __name__ == "__main__":
    run()
