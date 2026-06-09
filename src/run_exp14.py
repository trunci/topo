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
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from src.attn_extract import (load_named, format_prompt, is_correct,
                              get_attentions_lowmem)
from src.data_hotpotqa import build_hotpot_items
from src.topology import h1_features_from_edges
from src.fast_features import dense_features

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
N_ITEMS = 200
SEED = 0
TOP_K = 8
N_JOBS = int(os.environ.get("EXP14_N_JOBS", "0")) or (os.cpu_count() or 1)


def _worker_init():
    """Hide the GPU from topology workers — they are pure-CPU and must never
    initialise a CUDA context (which would squat on GPU memory)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _h1_persist_from_edges(args):
    """Worker: H1 scalars from a compact (n_nodes, edge-list).

    The main process does symmetrize+sparsify and ships only the edge list
    (~10k floats) rather than the dense 1319x1319 matrix (~7MB), so the
    process-boundary pickling cost drops ~170x. Identical math.
    """
    n, edges = args
    f = h1_features_from_edges(n, edges)
    return f["total_persistence"], f["max_persistence"]


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
def topo_features(model, tok, device, text: str, top_k: int,
                  pool: ProcessPoolExecutor | None = None,
                  threads: ThreadPoolExecutor | None = None) -> dict:
    # Low-memory per-layer capture: never holds more than one layer's attention
    # on the GPU, so long contexts don't OOM a 22 GB card holding a 7B model.
    atts, n_seq = get_attentions_lowmem(model, tok, device, text)

    # Flatten to a list of per-head matrices.
    mats = [layer[h] for layer in atts for h in range(layer.shape[0])]
    n_heads = len(mats)

    # Per-head dense features (first-order controls + sparsified edge list),
    # computed loop-free (src.fast_features) so numpy releases the GIL and a
    # thread pool parallelises across heads. Identical numbers to the originals
    # (verified by fast_features.verify_identical). The token-distance matrix is
    # the same for every head, so build it once and share it.
    n = mats[0].shape[0]
    idx = np.arange(n)
    D = np.abs(idx[None, :] - idx[:, None])

    def _df(A):
        return dense_features(A, top_k, D)

    if threads is not None:
        feats_list = list(threads.map(_df, mats))
    else:
        feats_list = [_df(A) for A in mats]

    dist = [f["dist"] for f in feats_list]
    off  = [f["off"]  for f in feats_list]
    ent  = [f["ent"]  for f in feats_list]
    kl   = [f["kl"]   for f in feats_list]
    args = [f["edges"] for f in feats_list]   # compact edge lists for the H1 pool

    # H1 persistent homology is the bottleneck — fan it across worker processes.
    if pool is not None:
        results = list(pool.map(_h1_persist_from_edges, args, chunksize=8))
    else:
        results = [_h1_persist_from_edges(a) for a in args]
    tot = [r[0] for r in results]
    mx  = [r[1] for r in results]
    nontriv = sum(1 for t in tot if t > 0)

    return {
        "topo_mean_persist":    float(np.mean(tot)),
        "topo_max_persist":     float(np.max(mx)),
        "topo_frac_nontrivial": float(nontriv / n_heads),
        "ctrl_attn_distance":   float(np.mean(dist)),
        "ctrl_offdiag_mass":    float(np.mean(off)),
        "ctrl_attn_entropy":    float(np.mean(ent)),
        "ctrl_kl_from_uniform": float(np.mean(kl)),   # KL-probe baseline feature
        "seq_len":              n_seq,
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
    print(f"[exp14] topology parallelism: {N_JOBS} workers", flush=True)
    # Per-item checkpoint (JSONL): on a Spot/preemptible VM the run can be
    # reclaimed mid-way, and results are otherwise only written at the end. We
    # append each completed row to a .jsonl and resume by skipping done ids, so
    # a preemption never loses progress — just relaunch the same command.
    ckpt = os.path.splitext(out)[0] + ".jsonl"
    rows = []
    done_ids = set()
    if os.path.exists(ckpt):
        with open(ckpt) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append(r)
                done_ids.add(r["id"])
        print(f"[exp14] resuming from checkpoint: {len(done_ids)} items already done",
              flush=True)

    model, tok, device = load_named(model_name, device=device_str)

    skipped = 0
    ckpt_fh = open(ckpt, "a")
    # 'spawn' context + GPU hidden in workers: topology workers start fresh and
    # never inherit or create a CUDA context, so they can't squat on GPU memory
    # (a fork-after-CUDA-init pool orphans workers that hold the model's 14 GB).
    pool = (ProcessPoolExecutor(max_workers=N_JOBS, mp_context=mp.get_context("spawn"),
                                initializer=_worker_init)
            if N_JOBS > 1 else None)
    # Threads (not processes) for the dense per-head features: the work is numpy
    # that releases the GIL, so threads parallelise it without pickling matrices.
    threads = ThreadPoolExecutor(max_workers=N_JOBS) if N_JOBS > 1 else None
    try:
        for k, it in enumerate(items):
            if it.id in done_ids:
                continue
            try:
                correct, ans = is_correct(model, tok, device, it.prompt, it.answer,
                                          max_new_tokens=20)
                conf = confidence_margin(model, tok, device, it.prompt)
                feats = topo_features(model, tok, device, it.prompt, top_k,
                                      pool=pool, threads=threads)
            except torch.OutOfMemoryError:
                skipped += 1
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"[exp14] {k+1}/{len(items)} OOM-skipped "
                      f"({it.level}); total skipped={skipped}", flush=True)
                continue
            row = {
                "id": it.id,
                "level": it.level,
                "answer": it.answer,
                "generated": ans,
                "is_correct": int(bool(correct)),
                "confidence_margin": conf,
                **feats,
            }
            rows.append(row)
            ckpt_fh.write(json.dumps(row) + "\n")
            ckpt_fh.flush()
            os.fsync(ckpt_fh.fileno())   # durable before a possible preemption
            if (k + 1) % 20 == 0 or k == 0:
                acc = np.mean([r["is_correct"] for r in rows])
                print(f"[exp14] {k+1}/{len(items)} done; n_kept={len(rows)} "
                      f"acc={acc:.3f} skipped={skipped} "
                      f"(last: {it.level} correct={int(bool(correct))} "
                      f"ans='{it.answer}' gen='{ans[:30]}')", flush=True)
            # gc FIRST (drop reference-cycle model outputs / KV cache), THEN
            # empty_cache can actually reclaim the GPU memory — otherwise the
            # resident baseline creeps up ~3GB/item and later items OOM.
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        if pool is not None:
            pool.shutdown()
        if threads is not None:
            threads.shutdown()
        ckpt_fh.close()
        del model
        gc.collect()
    print(f"[exp14] FINISHED: kept {len(rows)}/{len(items)}, "
          f"OOM-skipped {skipped}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f"[exp14] wrote {len(df)} rows to {out}; "
          f"overall acc={df['is_correct'].mean():.3f}", flush=True)
    print(f"[exp14] acc by level:\n{df.groupby('level')['is_correct'].mean()}", flush=True)
    return df


if __name__ == "__main__":
    run()
