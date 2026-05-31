"""Experiment 8: cellular sheaves over the residual stream.

Pre-registered design: docs/superpowers/specs/2026-05-31-experiment8-sheaves-design.md

One forward pass per item (Qwen2.5-0.5B-Instruct) captures BOTH attentions and
hidden states, so Frame 1 (Spike discrimination) and Frame 2 (failure prediction)
share the sweep. Per (layer, head) we compute, on the symmetrized top-k attention
graph:
  * Tier 1 (shape baseline): Fiedler value of the weighted graph Laplacian.
  * Tier 2 (sheaf): harmonic dim, spectral gap, and mean discord of the
    connection-Laplacian sheaf whose node stalks are PCA-reduced residual-stream
    vectors and whose restriction maps are local-PCA orthogonal frames.
We also recompute H1 persistence, the model's confidence margin, and first-order
controls so all predictors live in one row.

Output: results/exp8_features.parquet (one row per item). Memory-safe: CPU,
float32, eager, threads=2, one model, gc per item.
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import torch

from src.attn_extract import load_named, format_prompt, is_correct
from src.data_gen import build_items
from src.topology import h1_features, symmetrize, sparsify
from src.induction import attention_distance, offdiag_mass
from src.residual import attention_entropy
from src.sheaf import (fiedler_value, sheaf_laplacian, normalized_sheaf_laplacian,
                       harmonic_dim, spectral_gap, mean_discord, local_pca_frame)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_PER_FAMILY = 30
HOPS = (1, 2, 3)
TOP_K = 8
SHEAF_DIM = 4              # d: stalk dimension (PCA-reduced residual vectors)
SEED = 0


def _pca_reduce(H: np.ndarray, d: int) -> np.ndarray:
    """Reduce [n, D] hidden states to [n, d] via PCA (top-d principal directions)."""
    if H.shape[1] <= d:
        out = np.zeros((H.shape[0], d))
        out[:, :H.shape[1]] = H
        return out
    Hc = H - H.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Hc, full_matrices=False)
    return Hc @ vt[:d].T


def _node_frames(X: np.ndarray, W: np.ndarray, d: int) -> dict:
    """Local-PCA orthogonal frame per node from its top-k neighbours (+ itself)."""
    n = X.shape[0]
    O = {}
    for v in range(n):
        nbrs = np.flatnonzero(W[v] > 0.0)
        idx = np.concatenate([[v], nbrs]) if nbrs.size else np.array([v])
        O[v] = local_pca_frame(X[idx], d)
    return O


def sheaf_features_for_head(A: np.ndarray, Xfull: np.ndarray, top_k: int,
                            d: int) -> dict:
    """Tier-1 + Tier-2 features for one (layer, head) attention matrix A.

    Xfull: [n, d] PCA-reduced residual vectors for this item (shared across heads).
    """
    W = sparsify(symmetrize(A), top_k=top_k)
    fied = fiedler_value(W, normalized=True)
    O = _node_frames(Xfull, W, d)
    L = sheaf_laplacian(W, O, d)
    Ln = normalized_sheaf_laplacian(L, d)
    return {
        "t1_fiedler": fied,
        "t2_harmonic_dim": float(harmonic_dim(Ln, d)),
        "t2_spectral_gap": float(spectral_gap(Ln)),
        "t2_mean_discord": float(mean_discord(W, O, Xfull)),
    }


@torch.no_grad()
def confidence_margin_from_logits(logits_last) -> float:
    top2 = torch.topk(logits_last, 2).values
    return float((top2[0] - top2[1]).item())


def run(out="results/exp8_features.parquet", model_name=MODEL,
        n_per_family=N_PER_FAMILY, hops=HOPS, top_k=TOP_K, d=SHEAF_DIM, seed=SEED):
    torch.set_num_threads(2)
    os.makedirs("results", exist_ok=True)

    items = build_items(n_per_family=n_per_family, hops=hops, seed=seed)
    print(f"[exp8] {len(items)} items; d={d} top_k={top_k}", flush=True)
    print(f"[exp8] loading {model_name} (cpu)...", flush=True)
    model, tok, device = load_named(model_name, device="cpu")
    rows = []
    try:
        for k, it in enumerate(items):
            correct, _ = is_correct(model, tok, device, it.prompt, it.gold)
            enc = tok(format_prompt(tok, it.prompt), return_tensors="pt").to(device)
            out_m = model(**enc, output_attentions=True, output_hidden_states=True)
            conf = confidence_margin_from_logits(out_m.logits[0, -1])

            atts = [a[0].float().cpu().numpy() for a in out_m.attentions]  # [H,n,n] each
            # residual stream: use the final layer's hidden states [n, D]
            H_last = out_m.hidden_states[-1][0].float().cpu().numpy()
            Xfull = _pca_reduce(H_last, d)

            t1f, hdim, sgap, disc = [], [], [], []
            htot, hmax, hnt = [], [], 0
            adist, aoff, aent = [], [], []
            n_heads = 0
            for layer in atts:
                for h in range(layer.shape[0]):
                    A = layer[h]
                    sf = sheaf_features_for_head(A, Xfull, top_k, d)
                    t1f.append(sf["t1_fiedler"])
                    hdim.append(sf["t2_harmonic_dim"])
                    sgap.append(sf["t2_spectral_gap"])
                    disc.append(sf["t2_mean_discord"])
                    hf = h1_features(A, top_k=top_k)
                    htot.append(hf["total_persistence"])
                    hmax.append(hf["max_persistence"])
                    hnt += 1 if hf["total_persistence"] > 0 else 0
                    adist.append(attention_distance(A))
                    aoff.append(offdiag_mass(A))
                    aent.append(attention_entropy(A))
                    n_heads += 1

            rows.append({
                "id": it.id, "family": it.family, "hop": it.hop,
                "is_correct": int(bool(correct)), "confidence_margin": conf,
                # sheaf Tier 1 / Tier 2 (mean + max aggregation)
                "sheaf_t1_fiedler_mean": float(np.mean(t1f)),
                "sheaf_t1_fiedler_max": float(np.max(t1f)),
                "sheaf_harmonic_dim_mean": float(np.mean(hdim)),
                "sheaf_spectral_gap_mean": float(np.mean(sgap)),
                "sheaf_discord_mean": float(np.mean(disc)),
                "sheaf_discord_max": float(np.max(disc)),
                # H1 topology (for the nested "beyond H1" test)
                "h1_mean_persist": float(np.mean(htot)),
                "h1_max_persist": float(np.max(hmax)),
                "h1_frac_nontrivial": float(hnt / n_heads),
                # first-order controls
                "ctrl_attn_distance": float(np.mean(adist)),
                "ctrl_offdiag_mass": float(np.mean(aoff)),
                "ctrl_attn_entropy": float(np.mean(aent)),
            })
            if (k + 1) % 10 == 0 or k == 0:
                acc = np.mean([r["is_correct"] for r in rows])
                print(f"[exp8] {k + 1}/{len(items)} done; acc={acc:.3f} "
                      f"(hop{it.hop}); discord_mean={rows[-1]['sheaf_discord_mean']:.4f}",
                      flush=True)
            gc.collect()
    finally:
        del model
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f"[exp8] wrote {len(df)} rows to {out}; acc={df['is_correct'].mean():.3f}",
          flush=True)
    return df


if __name__ == "__main__":
    run()
