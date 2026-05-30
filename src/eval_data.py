"""Evaluation data for Experiment 1: WikiText-2 text slice + kinship prompts."""
from __future__ import annotations

from src.data_gen import build_pairs


def kinship_eval_items(n_per_family: int = 15, seed: int = 0):
    """Kinship minimal-pair items as dicts with prompt/gold/hop/pair_id/id.

    Only the kinship family (the clean, exactly-length-matched one).
    """
    items = []
    for it in build_pairs(n_per_family=n_per_family, seed=seed):
        if it.family != "kinship":
            continue
        items.append({"id": it.id, "pair_id": it.pair_id, "hop": it.hop,
                      "prompt": it.prompt, "gold": it.gold})
    return items


def wikitext_lines(n: int = 50, min_chars: int = 100):
    """Return the first n WikiText-2 test lines with at least min_chars characters."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    out = []
    for row in ds:
        t = row["text"].strip()
        if len(t) >= min_chars:
            out.append(t)
        if len(out) >= n:
            break
    return out
