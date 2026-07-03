"""Span-resolved per-head attachment features (Experiment 18, Phase B).

For each head, measure how the generated answer's attention distributes over
labeled regions of the prompt: gold paragraphs, distractor paragraphs, and the
question. Uses the same response-row capture as toha_features; spans are
recovered by substring search in the chat-formatted prompt + the fast
tokenizer's offset mapping.
"""
from __future__ import annotations

import numpy as np

from src.attn_extract import format_prompt
from src.toha_features import mtop_div


def paragraph_token_masks(tok, prompt_text: str, paragraph_lines,
                          is_gold, question: str, n_prompt_tokens: int):
    """Boolean masks (over prompt token positions) for gold paragraphs,
    distractor paragraphs, and the question line."""
    formatted = format_prompt(tok, prompt_text)
    enc = tok(formatted, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"][:n_prompt_tokens]

    def char_span_to_mask(cs: int, ce: int) -> np.ndarray:
        m = np.zeros(n_prompt_tokens, dtype=bool)
        for i, (s, e) in enumerate(offsets):
            if e > cs and s < ce and e > s:
                m[i] = True
        return m

    gold = np.zeros(n_prompt_tokens, dtype=bool)
    dist = np.zeros(n_prompt_tokens, dtype=bool)
    search_from = 0
    for line, g in zip(paragraph_lines, is_gold):
        cs = formatted.find(line, search_from)
        if cs == -1:                       # fall back to global search
            cs = formatted.find(line)
        if cs == -1:
            raise ValueError(f"paragraph line not found in prompt: {line[:60]!r}")
        ce = cs + len(line)
        search_from = ce
        m = char_span_to_mask(cs, ce)
        if g:
            gold |= m
        else:
            dist |= m

    qline = f"Question: {question}"
    qs = formatted.find(qline)
    if qs == -1:
        raise ValueError("question line not found in prompt")
    qmask = char_span_to_mask(qs, qs + len(qline))
    return gold, dist, qmask


def span_head_features(rows_per_layer, p: int, gold_mask, dist_mask,
                       q_mask) -> dict:
    """Per-head span attachment + whole-prompt controls.

    rows_per_layer: list of [heads, R, n] response-row attention arrays.
    Heads flattened layer-major, as everywhere in this repo.
    """
    ph = {k: [] for k in ("gold_mass", "dist_mass", "q_mass",
                          "gold_max", "dist_max", "resp_ent", "mtd")}
    eps = 1e-10
    for layer_rows in rows_per_layer:
        for h in range(layer_rows.shape[0]):
            A = layer_rows[h]                       # [R, n]
            P = A[:, :p]
            ph["gold_mass"].append(float(P[:, gold_mask].sum(axis=1).mean()))
            ph["dist_mass"].append(float(P[:, dist_mask].sum(axis=1).mean()))
            ph["q_mass"].append(float(P[:, q_mask].sum(axis=1).mean()))
            ph["gold_max"].append(
                float(P[:, gold_mask].max()) if gold_mask.any() else 0.0)
            ph["dist_max"].append(
                float(P[:, dist_mask].max()) if dist_mask.any() else 0.0)
            ph["resp_ent"].append(
                float(np.mean(-np.sum(A * np.log(A + eps), axis=-1))))
            ph["mtd"].append(mtop_div(A, p))

    gm = np.array(ph["gold_mass"])
    dm = np.array(ph["dist_mass"])
    out = {
        "ctx_mass_pooled": float((gm + dm).mean()),
        "gold_frac_pooled": float((gm / (gm + dm + 1e-12)).mean()),
        "q_mass_pooled": float(np.mean(ph["q_mass"])),
        "resp_ent_pooled": float(np.mean(ph["resp_ent"])),
        "gold_mask_tokens": int(gold_mask.sum()),
        "dist_mask_tokens": int(dist_mask.sum()),
    }
    out.update({f"ph_{k}": [float(v) for v in vs] for k, vs in ph.items()})
    return out
