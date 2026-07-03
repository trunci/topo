"""Experiment 19: boundary-interpolation ladder (pre-registered FINDINGS_exp19.md).

Stages (select with EXP19_STAGE):
* equiv    -- verify the exp15 fast feature path reproduces exp13's slow
              topo_features on 5 R1 items (pre-registered gate for reusing the
              fast path across all rungs).
* rewrite  -- Mistral-7B-Instruct-v0.3 generates R3 contexts and R4 questions
              for the 120 R1 items, validated programmatically (<=3 attempts,
              greedy first then sampled retries); persisted to
              results/exp19_rewrites.jsonl (resumable).
* features -- for each subject model (EXP19_MODELS, default both Qwen sizes)
              and each rung R1-R5: correctness, confidence, pooled
              topology+control features via the fast path. Checkpointed JSONL,
              final parquet per model: results/exp19_features_{key}.parquet.

EXP19_SMOKE=1 -> tiny run into smoke_-prefixed files.
Downstream: compute_stats_exp19 -> results/exp19_stats.json
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

from src.attn_extract import format_prompt, is_correct, load_named
from src.data_gen import Item
from src.data_gen_exp19 import (build_r1, build_r2, rewrite_context_prompt,
                                rewrite_question_prompt, validate_context,
                                validate_question)
from src.data_hotpotqa import build_hotpot_items
from src.run_exp13 import confidence_margin
from src.run_exp13 import topo_features as topo_features_slow
from src.run_exp14 import _worker_init
from src.run_exp15 import topo_features_perhead

REWRITER = "mistralai/Mistral-7B-Instruct-v0.3"
MODELS = {"0p5b": "Qwen/Qwen2.5-0.5B-Instruct",
          "1p5b": "Qwen/Qwen2.5-1.5B-Instruct"}
TOP_K = 8
SEED = 0
POOLED_KEYS = ("topo_mean_persist", "topo_max_persist", "topo_frac_nontrivial",
               "ctrl_attn_distance", "ctrl_offdiag_mass", "ctrl_attn_entropy",
               "ctrl_kl_from_uniform", "seq_len")
N_JOBS = int(os.environ.get("EXP19_N_JOBS", "0")) or min(8, os.cpu_count() or 1)


def _smoke():
    return os.environ.get("EXP19_SMOKE") == "1"


def _prefix():
    return "smoke_" if _smoke() else ""


def _pooled_fast(model, tok, device, text, pool, threads):
    feats = topo_features_perhead(model, tok, device, text, TOP_K,
                                  pool=pool, threads=threads)
    return {k: feats[k] for k in POOLED_KEYS}


# ---------------- stage: equiv ----------------

def stage_equiv():
    model, tok, device = load_named(MODELS["0p5b"], device="cpu")
    items = build_r1(n_per_family=2)[:5]
    for it in items:
        slow = topo_features_slow(model, tok, device, it.prompt, TOP_K)
        fast = _pooled_fast(model, tok, device, it.prompt, None, None)
        for k, v in slow.items():
            assert abs(fast[k] - v) < 1e-6, (it.id, k, v, fast[k])
        print(f"[equiv] {it.id} OK", flush=True)
    print("[equiv] PASS — fast path reproduces exp13 features", flush=True)


# ---------------- stage: rewrite ----------------

@torch.no_grad()
def _gen(model, tok, device, prompt_text, attempt):
    enc = tok(format_prompt(tok, prompt_text), return_tensors="pt").to(device)
    if attempt == 0:
        out = model.generate(**enc, max_new_tokens=180, do_sample=False)
    else:
        torch.manual_seed(SEED + attempt)
        out = model.generate(**enc, max_new_tokens=180, do_sample=True,
                             temperature=0.7, top_p=0.95)
    return tok.decode(out[0, enc["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def stage_rewrite():
    n_pf = 2 if _smoke() else 30
    ckpt = f"results/{_prefix()}exp19_rewrites.jsonl"
    done = {}
    if os.path.exists(ckpt):
        with open(ckpt) as fh:
            done = {json.loads(l)["id"]: json.loads(l) for l in fh if l.strip()}
        print(f"[rewrite] resuming: {len(done)} done", flush=True)
    items = build_r1(n_per_family=n_pf)
    model, tok, device = load_named(REWRITER)
    fh = open(ckpt, "a")
    try:
        for k, it in enumerate(items):
            if it.id in done:
                continue
            rec = {"id": it.id, "ctx_r3": None, "q_r4": None,
                   "ctx_attempts": 0, "q_attempts": 0}
            for a in range(3):
                rec["ctx_attempts"] = a + 1
                text = _gen(model, tok, device, rewrite_context_prompt(it), a)
                if validate_context(it, text):
                    rec["ctx_r3"] = " ".join(text.split())
                    break
            for a in range(3):
                rec["q_attempts"] = a + 1
                text = _gen(model, tok, device, rewrite_question_prompt(it), a)
                if validate_question(it, text):
                    rec["q_r4"] = " ".join(text.split())
                    break
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if (k + 1) % 20 == 0 or k == 0:
                print(f"[rewrite] {k+1}/{len(items)} "
                      f"(ctx ok={rec['ctx_r3'] is not None} "
                      f"q ok={rec['q_r4'] is not None})", flush=True)
    finally:
        fh.close()
        del model
        gc.collect()
    print("[rewrite] DONE", flush=True)


# ---------------- stage: features ----------------

def _load_rewrites():
    ckpt = f"results/{_prefix()}exp19_rewrites.jsonl"
    with open(ckpt) as fh:
        return {r["id"]: r for r in map(json.loads, fh) if r["id"]}


def build_rungs():
    n_pf = 2 if _smoke() else 30
    n_hot = 4 if _smoke() else 200
    r1 = build_r1(n_per_family=n_pf)
    rungs = {"r1": r1, "r2": build_r2(n_per_family=n_pf)}
    rw = _load_rewrites()
    r3, r4, dropped = [], [], 0
    for it in r1:
        rec = rw.get(it.id)
        if not rec or not rec["ctx_r3"]:
            dropped += 1
            continue
        ctx = rec["ctx_r3"]
        r3.append(Item(it.id + "-r3", it.family, it.hop, it.pair_id, ctx,
                       it.question, f"{ctx} {it.question}", it.gold))
        if rec["q_r4"]:
            q = rec["q_r4"]
            r4.append(Item(it.id + "-r4", it.family, it.hop, it.pair_id, ctx,
                           q, f"{ctx} {q}", it.gold))
    rungs["r3"], rungs["r4"] = r3, r4
    hot = build_hotpot_items(n=n_hot, seed=SEED, gold_only=True)
    rungs["r5"] = [Item(h.id, "hotpot", 0, h.id, "", h.question, h.prompt,
                        h.answer) for h in hot]
    print(f"[features] rung sizes: "
          f"{ {k: len(v) for k, v in rungs.items()} }; r3/r4 attrition from "
          f"{len(r1)}: r3={len(r1)-len(r3)}, r4={len(r1)-len(r4)}", flush=True)
    return rungs


def stage_features(model_keys):
    rungs = build_rungs()
    for key in model_keys:
        out = f"results/{_prefix()}exp19_features_{key}.parquet"
        ckpt = os.path.splitext(out)[0] + ".jsonl"
        rows, done = [], set()
        if os.path.exists(ckpt):
            with open(ckpt) as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        rows.append(r)
                        done.add((r["rung"], r["id"]))
            print(f"[features/{key}] resuming: {len(done)} done", flush=True)
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        model, tok, device = load_named(MODELS[key], device=device_str)
        pool = (ProcessPoolExecutor(max_workers=N_JOBS,
                                    mp_context=mp.get_context("spawn"),
                                    initializer=_worker_init)
                if N_JOBS > 1 else None)
        threads = ThreadPoolExecutor(max_workers=N_JOBS) if N_JOBS > 1 else None
        fh = open(ckpt, "a")
        try:
            for rung, items in rungs.items():
                mnt = 20 if rung == "r5" else 12
                for k, it in enumerate(items):
                    if (rung, it.id) in done:
                        continue
                    correct, ans = is_correct(model, tok, device, it.prompt,
                                              it.gold, max_new_tokens=mnt)
                    conf = confidence_margin(model, tok, device, it.prompt)
                    feats = _pooled_fast(model, tok, device, it.prompt,
                                         pool, threads)
                    row = {"rung": rung, "id": it.id, "family": it.family,
                           "hop": it.hop, "is_correct": int(bool(correct)),
                           "confidence_margin": conf, "generated": ans,
                           **feats}
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                    if (k + 1) % 20 == 0 or k == 0:
                        acc = np.mean([r["is_correct"] for r in rows
                                       if r["rung"] == rung])
                        print(f"[features/{key}/{rung}] {k+1}/{len(items)} "
                              f"acc={acc:.3f}", flush=True)
                    gc.collect()
        finally:
            fh.close()
            if pool is not None:
                pool.shutdown()
            if threads is not None:
                threads.shutdown()
            del model
            gc.collect()
        df = pd.DataFrame(rows)
        df.to_parquet(out)
        print(f"[features/{key}] wrote {len(df)} rows to {out}", flush=True)
        print(df.groupby("rung")["is_correct"].mean(), flush=True)


def main():
    os.makedirs("results", exist_ok=True)
    stage = os.environ.get("EXP19_STAGE", "features")
    if stage == "equiv":
        stage_equiv()
    elif stage == "rewrite":
        stage_rewrite()
    elif stage == "features":
        keys = os.environ.get("EXP19_MODELS", "0p5b,1p5b").split(",")
        stage_features([k.strip() for k in keys if k.strip()])
    else:
        raise SystemExit(f"unknown EXP19_STAGE={stage}")


if __name__ == "__main__":
    main()
