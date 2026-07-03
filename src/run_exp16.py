"""Experiment 16: can TOHA-style engineered topology cross the real-QA boundary?

Exps 14-15 (RED) showed our hand-specified H1 features are at chance on
HotpotQA/Mistral-7B while TOHA reports 0.71 there. The paper's closing claim
("the gap is feature construction") is currently untested. Exp 16 tests it
head-on: implement TOHA's MTop-Div (generation-time topological divergence of
answer tokens attaching to the prompt, per head) inside OUR pipeline, on OUR
exact items, and score it with OUR probes.

Conditions (same 200 bridge items, seed 0, as Exps 14/15):
* distractor -- full 10-paragraph contexts (Exp 14 geometry)
* gold       -- gold-only contexts (Exp 15 geometry)

Per item: greedy answer (labels + generated ids), confidence margin, then one
teacher-forced forward capturing response-row attention only; per-head
MTop-Div + response-row entropy controls are persisted (ph_* lists,
layer-major), so all head-selection analysis happens offline.

Verdict rules are pre-registered in FINDINGS_exp16.md BEFORE the run.

Model: mistralai/Mistral-7B-Instruct-v0.3.
Output: results/exp16_{condition}_features.parquet (+ .jsonl checkpoint, resumable)
Downstream: compute_stats_exp16 -> results/exp16_stats.json
"""
from __future__ import annotations

import gc
import json
import os

import numpy as np
import pandas as pd
import torch

from src.attn_extract import format_prompt, load_named
from src.data_hotpotqa import build_hotpot_items
from src.run_exp14 import MODEL, confidence_margin
from src.toha_features import get_response_rows, toha_head_features

N_ITEMS = 200
SEED = 0
MAX_NEW_TOKENS = 20
CONDITIONS = (("distractor", False), ("gold", True))


@torch.no_grad()
def generate_with_ids(model, tok, device, text: str, gold: str,
                      max_new_tokens: int = MAX_NEW_TOKENS):
    """Greedy answer + correctness label + generated ids (specials stripped)."""
    enc = tok(format_prompt(tok, text), return_tensors="pt").to(device)
    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = gen[0, enc["input_ids"].shape[1]:]
    ans = tok.decode(gen_ids, skip_special_tokens=True)
    special = set(tok.all_special_ids)
    keep = torch.tensor([i for i in gen_ids.tolist() if i not in special],
                        dtype=gen_ids.dtype)
    return (gold.lower() in ans.lower()), ans, keep


def run_condition(model, tok, device, condition: str, gold_only: bool,
                  n_items=N_ITEMS, seed=SEED):
    # EXP16_SMOKE=1 -> tiny run into smoke_-prefixed files, so a pod smoke test
    # can never contaminate the real checkpoints (its item ids differ: sampling
    # with a different n draws a different subset).
    smoke = os.environ.get("EXP16_SMOKE") == "1"
    prefix = "smoke_" if smoke else ""
    if smoke:
        n_items = 2
    out = f"results/{prefix}exp16_{condition}_features.parquet"
    ckpt = os.path.splitext(out)[0] + ".jsonl"

    items = build_hotpot_items(n=n_items, seed=seed, gold_only=gold_only)
    rows, done_ids = [], set()
    if os.path.exists(ckpt):
        with open(ckpt) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    rows.append(r)
                    done_ids.add(r["id"])
        print(f"[exp16/{condition}] resuming: {len(done_ids)} done", flush=True)

    skipped = 0
    ckpt_fh = open(ckpt, "a")
    try:
        for k, it in enumerate(items):
            if it.id in done_ids:
                continue
            try:
                correct, ans, gen_ids = generate_with_ids(
                    model, tok, device, it.prompt, it.answer)
                if len(gen_ids) == 0:
                    skipped += 1
                    print(f"[exp16/{condition}] {k+1}/{len(items)} empty "
                          f"generation, skipped", flush=True)
                    continue
                conf = confidence_margin(model, tok, device, it.prompt)
                rows_pl, p, n = get_response_rows(model, tok, device,
                                                  it.prompt, gen_ids)
                feats = toha_head_features(rows_pl, p)
                del rows_pl
            except torch.OutOfMemoryError:
                skipped += 1
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"[exp16/{condition}] {k+1}/{len(items)} OOM-skipped "
                      f"(total {skipped})", flush=True)
                continue
            row = {
                "id": it.id,
                "level": it.level,
                "answer": it.answer,
                "generated": ans,
                "is_correct": int(bool(correct)),
                "confidence_margin": conf,
                "seq_len": n,
                "n_prompt": p,
                "n_resp": int(len(gen_ids)),
                **feats,
            }
            rows.append(row)
            ckpt_fh.write(json.dumps(row) + "\n")
            ckpt_fh.flush()
            os.fsync(ckpt_fh.fileno())
            if (k + 1) % 20 == 0 or k == 0:
                acc = np.mean([r["is_correct"] for r in rows])
                print(f"[exp16/{condition}] {k+1}/{len(items)} done "
                      f"acc={acc:.3f} skipped={skipped} n={n} R={len(gen_ids)} "
                      f"mtd_mean={feats['mtd_mean']:.4f} "
                      f"(ans='{it.answer}' gen='{ans[:30]}')", flush=True)
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        ckpt_fh.close()

    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f"[exp16/{condition}] wrote {len(df)} rows to {out}; "
          f"acc={df['is_correct'].mean():.3f}", flush=True)
    return df


def run(model_name=MODEL):
    os.makedirs("results", exist_ok=True)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cpu":
        torch.set_num_threads(4)
    print(f"[exp16] loading {model_name} ({device_str})...", flush=True)
    model, tok, device = load_named(model_name, device=device_str)
    try:
        for condition, gold_only in CONDITIONS:
            print(f"[exp16] === condition: {condition} ===", flush=True)
            run_condition(model, tok, device, condition, gold_only)
    finally:
        del model
        gc.collect()
    print("[exp16] FINISHED both conditions", flush=True)


if __name__ == "__main__":
    run()
