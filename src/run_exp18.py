"""Experiment 18: grounding heads Phase B — NoContext + span-resolved capture.

Pre-registered in FINDINGS_exp18.md BEFORE this file was first run. Two passes
over the same 200 HotpotQA bridge items (seed 0) as Exps 14-17:

* nocontext -- question-only prompt; greedy answer, correctness, confidence.
  Defines the parametric-memory label (`knew_anyway`).
* spans     -- full-distractor geometry; greedy answer, confidence, then one
  teacher-forced forward capturing answer-row attention, resolved into per-head
  attachment to gold paragraphs / distractor paragraphs / the question span.

Model: mistralai/Mistral-7B-Instruct-v0.3.
Output: results/exp18_nocontext_features.parquet, results/exp18_spans_features.parquet
        (+ .jsonl checkpoints, resumable). EXP18_SMOKE=1 -> 2 items, smoke_ prefix.
Downstream: compute_stats_exp18 -> results/exp18_stats.json
"""
from __future__ import annotations

import gc
import json
import os

import numpy as np
import pandas as pd
import torch

from src.attn_extract import load_named
from src.data_hotpotqa import build_hotpot_items_spans
from src.run_exp14 import MODEL, confidence_margin
from src.run_exp16 import generate_with_ids
from src.span_features import paragraph_token_masks, span_head_features
from src.toha_features import get_response_rows

N_ITEMS = 200
SEED = 0


def _ckpt_paths(condition, smoke):
    prefix = "smoke_" if smoke else ""
    out = f"results/{prefix}exp18_{condition}_features.parquet"
    return out, os.path.splitext(out)[0] + ".jsonl"


def _load_done(ckpt):
    rows, done = [], set()
    if os.path.exists(ckpt):
        with open(ckpt) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    rows.append(r)
                    done.add(r["id"])
    return rows, done


def run_condition(model, tok, device, items, condition, smoke):
    out, ckpt = _ckpt_paths(condition, smoke)
    rows, done = _load_done(ckpt)
    if done:
        print(f"[exp18/{condition}] resuming: {len(done)} done", flush=True)
    skipped = 0
    fh = open(ckpt, "a")
    try:
        for k, it in enumerate(items):
            if it.id in done:
                continue
            try:
                prompt = it.nocontext_prompt if condition == "nocontext" else it.prompt
                correct, ans, gen_ids = generate_with_ids(
                    model, tok, device, prompt, it.answer)
                conf = confidence_margin(model, tok, device, prompt)
                row = {
                    "id": it.id, "level": it.level, "answer": it.answer,
                    "generated": ans, "is_correct": int(bool(correct)),
                    "confidence_margin": conf, "n_resp": int(len(gen_ids)),
                }
                if condition == "spans":
                    if len(gen_ids) == 0:
                        skipped += 1
                        continue
                    rows_pl, p, n = get_response_rows(model, tok, device,
                                                      prompt, gen_ids)
                    gold_m, dist_m, q_m = paragraph_token_masks(
                        tok, prompt, it.paragraph_lines, it.is_gold,
                        it.question, p)
                    row.update({"seq_len": n, "n_prompt": p,
                                **span_head_features(rows_pl, p, gold_m,
                                                     dist_m, q_m)})
                    del rows_pl
            except torch.OutOfMemoryError:
                skipped += 1
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"[exp18/{condition}] {k+1}/{len(items)} OOM-skipped "
                      f"(total {skipped})", flush=True)
                continue
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if (k + 1) % 20 == 0 or k == 0:
                acc = np.mean([r["is_correct"] for r in rows])
                extra = (f" gold_frac={row.get('gold_frac_pooled', 0):.3f}"
                         if condition == "spans" else "")
                print(f"[exp18/{condition}] {k+1}/{len(items)} acc={acc:.3f} "
                      f"skipped={skipped}{extra} "
                      f"(ans='{it.answer}' gen='{ans[:30]}')", flush=True)
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        fh.close()
    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f"[exp18/{condition}] wrote {len(df)} rows to {out}; "
          f"acc={df['is_correct'].mean():.3f}", flush=True)
    return df


def run(model_name=MODEL):
    os.makedirs("results", exist_ok=True)
    smoke = os.environ.get("EXP18_SMOKE") == "1"
    n_items = 2 if smoke else N_ITEMS
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    items = build_hotpot_items_spans(n=N_ITEMS, seed=SEED)[:n_items]
    print(f"[exp18] {len(items)} items, smoke={smoke}; loading {model_name} "
          f"({device_str})...", flush=True)
    model, tok, device = load_named(model_name, device=device_str)
    try:
        for condition in ("nocontext", "spans"):
            print(f"[exp18] === condition: {condition} ===", flush=True)
            run_condition(model, tok, device, items, condition, smoke)
    finally:
        del model
        gc.collect()
    print("[exp18] FINISHED both conditions", flush=True)


if __name__ == "__main__":
    run()
