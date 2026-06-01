"""Experiment 14: failure prediction on real QA (HotpotQA bridge).

Direct comparison to TOHA (Bazarova et al., ACL 2026, arXiv 2504.10063):
same model (Mistral-7B-Instruct-v0.3), same benchmark (HotpotQA bridge),
same task (predict whether the model's answer is correct from internal signals).

TOHA reports AUROC 0.71 on HotpotQA/Mistral-7B.
KL-divergence probe (arXiv 2605.05025) reports AUROC 0.78-0.80.

Our additions beyond the literature:
1. Regime-conditional analysis: stratify by question level (easy/medium/hard)
   to test whether topology is most useful when accuracy is near 50%.
2. Controls decomposition: does topology add beyond first-order attention stats?

Model: mistralai/Mistral-7B-Instruct-v0.3 (same as TOHA).
Auto-detects CUDA; uses bfloat16 on GPU, float32 on CPU.
Attention matrices always cast to float32 before topology computation.

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

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
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


def _kl_from_uniform(A: np.ndarray) -> float:
    """KL divergence of each attention row from uniform, averaged over rows.
    KL(p || u) = log(n) - H(p). Returns mean over all rows in the matrix."""
    n = A.shape[-1]
    eps = 1e-10
    ent_rows = -np.sum(A * np.log(A + eps), axis=-1)   # shape [n]
    kl_rows = np.log(n) - ent_rows
    return float(np.mean(kl_rows))


@torch.no_grad()
def topo_features(model, tok, device, text: str, top_k: int) -> dict:
    enc = tok(format_prompt(tok, text), return_tensors="pt").to(device)
    out = model(**enc, output_attentions=True)
    atts = [a[0].float().cpu().numpy() for a in out.attentions]
    tot, mx, nontriv, dist, off, ent, kl = [], [], 0, [], [], [], []
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
            kl.append(_kl_from_uniform(A))
            n_heads += 1
    return {
        "topo_mean_persist":    float(np.mean(tot)),
        "topo_max_persist":     float(np.max(mx)),
        "topo_frac_nontrivial": float(nontriv / n_heads),
        "ctrl_attn_distance":   float(np.mean(dist)),
        "ctrl_offdiag_mass":    float(np.mean(off)),
        "ctrl_attn_entropy":    float(np.mean(ent)),
        "ctrl_kl_from_uniform": float(np.mean(kl)),   # KL-probe baseline feature
        "seq_len":              int(enc["input_ids"].shape[1]),
    }


def run(out="results/exp14_features.parquet", model_name=MODEL,
        n_items=N_ITEMS, seed=SEED, top_k=TOP_K):
    import torch as _torch
    has_gpu = _torch.cuda.is_available()
    if not has_gpu:
        _torch.set_num_threads(4)
    os.makedirs("results", exist_ok=True)

    items = build_hotpot_items(n=n_items, seed=seed)
    device_str = "cuda" if has_gpu else "cpu"
    print(f"[exp14] {len(items)} HotpotQA bridge items (seed={seed})", flush=True)
    print(f"[exp14] loading {model_name} ({device_str})...", flush=True)
    model, tok, device = load_named(model_name, device=device_str)

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
