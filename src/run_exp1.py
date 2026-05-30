"""Experiment 1 orchestration: capture attention, build per-method keep-sets,
re-run masked, record loss/ppl/accuracy into results/exp1.parquet.

DMT sets the per-head edge budget; baselines match it exactly.
"""
from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
import torch

from src.attn_extract import load_named, format_prompt, MODEL_NAME
from src.topology import symmetrize, sparsify
from src.morse import morse_keep
from src import keepsets as _keepsets_mod
from src.pruned_forward import MaskedModel, keepsets_to_bias
from src.eval_data import wikitext_lines, kinship_eval_items

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
INSTRUCT_MODEL = MODEL_NAME  # "Qwen/Qwen2.5-0.5B-Instruct"
METHODS = ["unpruned", "dmt", "magnitude", "random", "window"]

TOP_K = 8
N_WIKITEXT = 50
WIKITEXT_MIN_CHARS = 100
MAX_TOKENS = 64
KINSHIP_N_PER_FAMILY = 15
SEED = 0


def _all_edges_keepset(W):
    n = W.shape[0]
    iu = np.triu_indices(n, k=1)
    return {frozenset((int(i), int(j))) for i, j in zip(*iu) if W[i, j] > 0.0}


def keepsets_for_example(att, top_k=TOP_K, seed=0):
    """att: numpy [L, H, n, n]. Returns dict method -> {(layer,head): keepset}.

    DMT decides the per-head budget k; magnitude/random/window keep exactly k.
    unpruned keeps all candidate edges.
    """
    L, H, n, _ = att.shape
    out = {m: {} for m in METHODS}
    for li in range(L):
        for h in range(H):
            W = sparsify(symmetrize(att[li, h]), top_k=top_k)
            dmt = morse_keep(W, top_k=top_k, symmetrized=True)
            k = len(dmt)
            out["dmt"][(li, h)] = dmt
            out["unpruned"][(li, h)] = _all_edges_keepset(W)
            out["magnitude"][(li, h)] = _keepsets_mod.magnitude_keep(W, k)
            # deterministic per-(example handled by caller seed)+layer+head seed
            out["random"][(li, h)] = _keepsets_mod.random_keep(W, k, seed=seed * 1000 + li * 100 + h)
            out["window"][(li, h)] = _keepsets_mod.window_keep(W, k)
    return out


def _mean_sparsity(keepsets_for_method, att):
    """Fraction of candidate edges kept, averaged over heads."""
    L, H, n, _ = att.shape
    fracs = []
    for li in range(L):
        for h in range(H):
            W = sparsify(symmetrize(att[li, h]), top_k=TOP_K)
            cand = _all_edges_keepset(W)
            if len(cand) == 0:
                continue
            fracs.append(len(keepsets_for_method[(li, h)]) / len(cand))
    return float(np.mean(fracs)) if fracs else float("nan")


@torch.no_grad()
def _attention_from_ids(model, enc):
    """Return attentions [L, H, n, n] for the exact input_ids (no re-templating)."""
    out = model(**enc, output_attentions=True)
    return np.stack([a[0].float().cpu().numpy() for a in out.attentions])


def run_wikitext(mm, tok, device, rows):
    texts = wikitext_lines(n=N_WIKITEXT, min_chars=WIKITEXT_MIN_CHARS)
    L = mm.model.config.num_hidden_layers
    H = mm.model.config.num_attention_heads
    for idx, text in enumerate(texts):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(device)
        n = enc["input_ids"].shape[1]
        if n < 4:
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
        print(f"[wikitext {idx+1}/{len(texts)}] done (n={n})")


def run_kinship(mm, tok, device, rows):
    items = kinship_eval_items(n_per_family=KINSHIP_N_PER_FAMILY, seed=SEED)
    L = mm.model.config.num_hidden_layers
    H = mm.model.config.num_attention_heads
    for idx, it in enumerate(items):
        text = format_prompt(tok, it["prompt"])
        enc = tok(text, return_tensors="pt").to(device)
        n = enc["input_ids"].shape[1]
        att = _attention_from_ids(mm.model, enc)
        per_method = keepsets_for_example(att, top_k=TOP_K, seed=1000 + idx)
        # gold first token id (leading space variant matches Qwen tokenization)
        gold_ids = tok(" " + it["gold"], add_special_tokens=False)["input_ids"]
        gold_id = gold_ids[0]
        for method in METHODS:
            ks = per_method[method]
            biases = keepsets_to_bias(ks, n=n, H=H, L=L, device=device)
            logits = mm.logits(enc, biases=biases)
            last = logits[0, -1]
            pred_id = int(last.argmax().item())
            # teacher-forced gold-token loss at the final position
            logp = torch.log_softmax(last, dim=-1)
            gold_loss = float(-logp[gold_id].item())
            rows.append({
                "example_id": f"kin-{it['id']}", "domain": "kinship", "method": method,
                "loss": gold_loss, "ppl": float("nan"),
                "correct": float(pred_id == gold_id),
                "mean_sparsity": _mean_sparsity(ks, att), "seq_len": n,
            })
        print(f"[kinship {idx+1}/{len(items)}] done (n={n})")


def main(out_path="results/exp1.parquet"):
    os.makedirs("results", exist_ok=True)
    rows = []
    # WikiText perplexity on the BASE model
    base_model, base_tok, device = load_named(BASE_MODEL)
    mm_base = MaskedModel(base_model)
    run_wikitext(mm_base, base_tok, device, rows)
    del base_model, mm_base
    # Kinship accuracy on the INSTRUCT model
    inst_model, inst_tok, device = load_named(INSTRUCT_MODEL)
    mm_inst = MaskedModel(inst_model)
    run_kinship(mm_inst, inst_tok, device, rows)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"\nWrote {len(df)} rows to {out_path}")
    for domain in ("wikitext", "kinship"):
        sub = df[df.domain == domain]
        if len(sub):
            metric = "ppl" if domain == "wikitext" else "correct"
            print(f"\n{domain} mean {metric} by method:")
            print(sub.groupby("method")[metric].mean().to_string())
    return df


if __name__ == "__main__":
    main()
