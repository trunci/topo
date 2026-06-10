"""Experiment 15: gold-only HotpotQA — does controlling context geometry rescue topology?

Exp 14 (Result Q) was a clean NULL on full 10-paragraph HotpotQA contexts, and
Exp 14b showed why: prompts span ~200-3800 tokens and the pooled global H1
features are length meters (rho up to 0.95 with seq_len) on a dataset where
length says nothing about correctness. Exp 13 (GREEN) had template-constant
lengths. Exp 15 isolates that variable: the SAME 200 bridge items (same model,
same seed, same pipeline), but the context cut to the 2 gold supporting
paragraphs — short, tight-variance, distractor-free, while the task stays real.

- GREEN here => the Exp 13 boundary is context GEOMETRY (length heterogeneity +
  distractor padding), not synthetic-vs-real semantics.
- RED here => real-task semantics kill the effect even when clean; the
  regime-conditional claim is strictly synthetic.

Lesson from Exp 14 applied: per-head features ARE persisted this time
(`ph_*` list columns, heads flattened layer-major: layer0 h0..h31, layer1 ...),
so supervised head selection / TOHA-style re-analysis is possible offline
without re-running the model.

Model: mistralai/Mistral-7B-Instruct-v0.3 (same as TOHA / Exp 14).
Output: results/exp15_features.parquet (+ .jsonl per-item checkpoint, resumable)
Downstream: compute_stats_exp15 -> results/exp15_stats.json
"""
from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
import torch

from src.attn_extract import load_named, is_correct, get_attentions_lowmem
from src.data_hotpotqa import build_hotpot_items
from src.fast_features import dense_features
from src.run_exp14 import (_worker_init, _h1_persist_from_edges,
                           confidence_margin, MODEL, TOP_K)

N_ITEMS = 200
SEED = 0
# Gold-only graphs are tiny (~150-350 nodes), so a handful of H1 workers
# saturates the work. More importantly each SPAWN worker re-imports the full
# torch/transformers stack (~1GB RSS, no copy-on-write) — 40 workers OOM'd the
# first L4 host's system RAM and broke the pool. Cap the default hard.
N_JOBS = (int(os.environ.get("EXP15_N_JOBS", "0"))
          or min(8, os.cpu_count() or 1))


@torch.no_grad()
def topo_features_perhead(model, tok, device, text: str, top_k: int,
                          pool: ProcessPoolExecutor | None = None,
                          threads: ThreadPoolExecutor | None = None) -> dict:
    """Same math as run_exp14.topo_features, but per-head vectors are kept.

    Pooled scalars are defined identically to Exp 14 (mean/max/frac over heads),
    so cross-experiment comparisons are apples-to-apples.
    """
    atts, n_seq = get_attentions_lowmem(model, tok, device, text)
    mats = [layer[h] for layer in atts for h in range(layer.shape[0])]
    n_heads = len(mats)

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
    args = [f["edges"] for f in feats_list]

    if pool is not None:
        results = list(pool.map(_h1_persist_from_edges, args, chunksize=8))
    else:
        results = [_h1_persist_from_edges(a) for a in args]
    tot = [r[0] for r in results]
    mx  = [r[1] for r in results]
    nontriv = sum(1 for t in tot if t > 0)

    return {
        # pooled (identical definitions to Exp 14)
        "topo_mean_persist":    float(np.mean(tot)),
        "topo_max_persist":     float(np.max(mx)),
        "topo_frac_nontrivial": float(nontriv / n_heads),
        "ctrl_attn_distance":   float(np.mean(dist)),
        "ctrl_offdiag_mass":    float(np.mean(off)),
        "ctrl_attn_entropy":    float(np.mean(ent)),
        "ctrl_kl_from_uniform": float(np.mean(kl)),
        "seq_len":              n_seq,
        # per-head (layer-major flattening) — the Exp 14 schema mistake, fixed
        "ph_tot":  [float(v) for v in tot],
        "ph_max":  [float(v) for v in mx],
        "ph_dist": [float(v) for v in dist],
        "ph_off":  [float(v) for v in off],
        "ph_ent":  [float(v) for v in ent],
        "ph_kl":   [float(v) for v in kl],
    }


def run(out="results/exp15_features.parquet", model_name=MODEL,
        n_items=N_ITEMS, seed=SEED, top_k=TOP_K):
    has_gpu = torch.cuda.is_available()
    if not has_gpu:
        torch.set_num_threads(4)
    os.makedirs("results", exist_ok=True)

    items = build_hotpot_items(n=n_items, seed=seed, gold_only=True)
    device_str = "cuda" if has_gpu else "cpu"
    print(f"[exp15] {len(items)} HotpotQA bridge items, GOLD-ONLY contexts "
          f"(seed={seed})", flush=True)
    print(f"[exp15] loading {model_name} ({device_str})...", flush=True)
    print(f"[exp15] topology parallelism: {N_JOBS} workers", flush=True)

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
        print(f"[exp15] resuming from checkpoint: {len(done_ids)} items already done",
              flush=True)

    model, tok, device = load_named(model_name, device=device_str)

    skipped = 0
    ckpt_fh = open(ckpt, "a")

    def _make_pool():
        return (ProcessPoolExecutor(max_workers=N_JOBS,
                                    mp_context=mp.get_context("spawn"),
                                    initializer=_worker_init)
                if N_JOBS > 1 else None)

    pool = _make_pool()
    threads = ThreadPoolExecutor(max_workers=N_JOBS) if N_JOBS > 1 else None
    try:
        for k, it in enumerate(items):
            if it.id in done_ids:
                continue
            try:
                correct, ans = is_correct(model, tok, device, it.prompt, it.answer,
                                          max_new_tokens=20)
                conf = confidence_margin(model, tok, device, it.prompt)
                feats = topo_features_perhead(model, tok, device, it.prompt, top_k,
                                              pool=pool, threads=threads)
            except torch.OutOfMemoryError:
                skipped += 1
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"[exp15] {k+1}/{len(items)} OOM-skipped "
                      f"({it.level}); total skipped={skipped}", flush=True)
                continue
            except BrokenProcessPool:
                # A dead worker poisons the whole pool; rebuild it and move on.
                # The item is NOT checkpointed, so relaunching the run retries it.
                skipped += 1
                print(f"[exp15] {k+1}/{len(items)} pool broke — rebuilding; "
                      f"item left for a resume pass (skipped={skipped})", flush=True)
                if pool is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
                pool = _make_pool()
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
            os.fsync(ckpt_fh.fileno())
            if (k + 1) % 20 == 0 or k == 0:
                acc = np.mean([r["is_correct"] for r in rows])
                print(f"[exp15] {k+1}/{len(items)} done; n_kept={len(rows)} "
                      f"acc={acc:.3f} skipped={skipped} seq_len={feats['seq_len']} "
                      f"(last: correct={int(bool(correct))} "
                      f"ans='{it.answer}' gen='{ans[:30]}')", flush=True)
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
    print(f"[exp15] FINISHED: kept {len(rows)}/{len(items)}, "
          f"OOM-skipped {skipped}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f"[exp15] wrote {len(df)} rows to {out}; "
          f"overall acc={df['is_correct'].mean():.3f} "
          f"mean_seq_len={df['seq_len'].mean():.0f}", flush=True)
    return df


if __name__ == "__main__":
    run()
