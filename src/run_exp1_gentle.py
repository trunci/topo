"""Gentle, low-memory driver for Experiment 1.

Reuses run_exp1's building blocks but:
- forces CPU (avoids MPS memory exhaustion that was crashing the machine),
- caps torch threads,
- frees memory (gc) after every example,
- prints flushed per-example progress,
- uses a small config.

Writes the same results/exp1.parquet that compute_stats_exp1 / analyze_exp1 read.
Override the small config via env vars: EXP1_N_WIKI, EXP1_MAXTOK, EXP1_KIN_PER_FAM.
"""
from __future__ import annotations

import gc
import math
import os
import sys

import numpy as np
import pandas as pd
import torch

from src.attn_extract import load_named, format_prompt, MODEL_NAME
from src.pruned_forward import MaskedModel, keepsets_to_bias
from src.eval_data import wikitext_lines, kinship_eval_items
from src.run_exp1 import (
    METHODS, TOP_K, keepsets_for_example, _attention_from_ids, _mean_sparsity,
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
INSTRUCT_MODEL = MODEL_NAME

# Small, safe defaults (override via env).
N_WIKITEXT = int(os.environ.get("EXP1_N_WIKI", "12"))
MAX_TOKENS = int(os.environ.get("EXP1_MAXTOK", "24"))
KINSHIP_N_PER_FAMILY = int(os.environ.get("EXP1_KIN_PER_FAM", "6"))
WIKITEXT_MIN_CHARS = 100
SEED = 0
DEVICE = "cpu"  # deliberately CPU for stability


def _log(msg):
    print(msg, flush=True)


def _free():
    gc.collect()


def main(out_path="results/exp1.parquet"):
    torch.set_num_threads(int(os.environ.get("EXP1_THREADS", "2")))
    os.makedirs("results", exist_ok=True)
    rows = []

    # ---- WikiText perplexity on the BASE model ----
    _log(f"[setup] loading base model on {DEVICE} ...")
    base_model, base_tok, device = load_named(BASE_MODEL, device=DEVICE)
    mm = MaskedModel(base_model)
    L = base_model.config.num_hidden_layers
    H = base_model.config.num_attention_heads
    texts = wikitext_lines(n=N_WIKITEXT, min_chars=WIKITEXT_MIN_CHARS)
    _log(f"[setup] {len(texts)} wikitext windows; {L}L x {H}H; max_tokens={MAX_TOKENS}")
    for idx, text in enumerate(texts):
        enc = base_tok(text, return_tensors="pt", truncation=True,
                       max_length=MAX_TOKENS).to(device)
        n = enc["input_ids"].shape[1]
        if n < 4:
            _log(f"[wikitext {idx+1}/{len(texts)}] skipped (n={n})")
            continue
        att = _attention_from_ids(mm.model, enc)
        per_method = keepsets_for_example(att, top_k=TOP_K, seed=idx)
        for method in METHODS:
            ks = per_method[method]
            biases = keepsets_to_bias(ks, n=n, H=H, L=L, device=device)
            loss = mm.loss(enc, biases=biases)
            rows.append({
                "example_id": f"wiki-{idx}", "domain": "wikitext", "method": method,
                "loss": loss, "ppl": math.exp(loss), "correct": float("nan"),
                "mean_sparsity": _mean_sparsity(ks, att), "seq_len": n,
            })
            del biases
        del att, per_method, enc
        _free()
        _log(f"[wikitext {idx+1}/{len(texts)}] done (n={n})")

    del base_model, mm
    _free()

    # ---- Kinship accuracy on the INSTRUCT model ----
    _log(f"[setup] loading instruct model on {DEVICE} ...")
    inst_model, inst_tok, device = load_named(INSTRUCT_MODEL, device=DEVICE)
    mm = MaskedModel(inst_model)
    L = inst_model.config.num_hidden_layers
    H = inst_model.config.num_attention_heads
    items = kinship_eval_items(n_per_family=KINSHIP_N_PER_FAMILY, seed=SEED)
    _log(f"[setup] {len(items)} kinship items")
    for idx, it in enumerate(items):
        text = format_prompt(inst_tok, it["prompt"])
        enc = inst_tok(text, return_tensors="pt").to(device)
        n = enc["input_ids"].shape[1]
        att = _attention_from_ids(mm.model, enc)
        per_method = keepsets_for_example(att, top_k=TOP_K, seed=1000 + idx)
        gold_id = inst_tok(" " + it["gold"], add_special_tokens=False)["input_ids"][0]
        for method in METHODS:
            ks = per_method[method]
            biases = keepsets_to_bias(ks, n=n, H=H, L=L, device=device)
            logits = mm.logits(enc, biases=biases)
            last = logits[0, -1]
            pred_id = int(last.argmax().item())
            logp = torch.log_softmax(last, dim=-1)
            gold_loss = float(-logp[gold_id].item())
            rows.append({
                "example_id": f"kin-{it['id']}", "domain": "kinship", "method": method,
                "loss": gold_loss, "ppl": float("nan"),
                "correct": float(pred_id == gold_id),
                "mean_sparsity": _mean_sparsity(ks, att), "seq_len": n,
            })
            del biases, logits
        del att, per_method, enc
        _free()
        _log(f"[kinship {idx+1}/{len(items)}] done (n={n})")

    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    _log(f"\n[done] wrote {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    main()
