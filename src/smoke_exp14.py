"""Smoke test for Exp 14 on the L4: run the FULL pipeline on the N longest-context
items (worst case for OOM) and report per-item GPU peak memory + wall-clock.

Purpose: decide whether a single 22GB L4 can run all 200 HotpotQA bridge items
without OOM, before committing to the full run. We deliberately pick the longest
prompts because those are the ones that spiked past the L4 last session.

This does NOT write results — it only measures feasibility/throughput.
"""
from __future__ import annotations

import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from src.attn_extract import load_named, format_prompt, is_correct
from src.data_hotpotqa import build_hotpot_items
from src.run_exp14 import (MODEL, TOP_K, N_JOBS, _worker_init,
                           confidence_margin, topo_features)

N_LONGEST = int(os.environ.get("SMOKE_N", "5"))


def main():
    items = build_hotpot_items(n=200, seed=0)
    model, tok, device = load_named(MODEL, device="cuda")

    # Rank items by prompt token length (an INPUT property, no results involved).
    lens = []
    for it in items:
        n = tok(format_prompt(tok, it.prompt), return_tensors="pt")["input_ids"].shape[1]
        lens.append((n, it))
    lens.sort(key=lambda x: x[0], reverse=True)
    chosen = lens[:N_LONGEST]
    print(f"[smoke] total items={len(items)} "
          f"token-len: min={lens[-1][0]} median={lens[len(lens)//2][0]} "
          f"max={lens[0][0]}", flush=True)
    print(f"[smoke] testing the {N_LONGEST} LONGEST items "
          f"(lens={[c[0] for c in chosen]}); N_JOBS={N_JOBS}", flush=True)

    pool = ProcessPoolExecutor(max_workers=N_JOBS,
                               mp_context=mp.get_context("spawn"),
                               initializer=_worker_init)
    oom = 0
    try:
        for rank, (n, it) in enumerate(chosen):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                correct, ans = is_correct(model, tok, device, it.prompt, it.answer,
                                          max_new_tokens=20)
                conf = confidence_margin(model, tok, device, it.prompt)
                feats = topo_features(model, tok, device, it.prompt, TOP_K, pool=pool)
                peak = torch.cuda.max_memory_allocated() / 1e9
                dt = time.time() - t0
                print(f"[smoke] #{rank+1} len={n} OK  peak_gpu={peak:.1f}GB  "
                      f"wall={dt:.1f}s  correct={int(bool(correct))} "
                      f"frac_nontriv={feats['topo_frac_nontrivial']:.3f}", flush=True)
            except torch.OutOfMemoryError:
                oom += 1
                torch.cuda.empty_cache()
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[smoke] #{rank+1} len={n} *** OOM *** peak_gpu={peak:.1f}GB",
                      flush=True)
            torch.cuda.empty_cache()
    finally:
        pool.shutdown()
    print(f"[smoke] DONE: {N_LONGEST - oom}/{N_LONGEST} survived, OOM={oom}", flush=True)
    if oom == 0:
        print("[smoke] VERDICT: L4 handles the longest contexts -> safe to run all 200 clean.",
              flush=True)
    else:
        print("[smoke] VERDICT: L4 OOMs on long tail -> need hardware cap or bigger GPU.",
              flush=True)


if __name__ == "__main__":
    main()
