"""HotpotQA bridge-question items for Exp 14 (real QA benchmark).

Loads the validation split, filters for bridge (2-hop) questions, and formats
each item as a context-grounded prompt with all 10 supporting paragraphs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from datasets import load_dataset


@dataclass(frozen=True)
class HotpotItem:
    id: str
    question: str
    answer: str
    level: str
    prompt: str       # formatted with context paragraphs


@dataclass(frozen=True)
class HotpotItemSpans:
    """Exp 18: distractor-geometry item with paragraph-level annotations.

    paragraph_lines[i] is the exact "title: para" line embedded in `prompt`,
    so token spans can be recovered by substring search + offset mapping.
    """
    id: str
    question: str
    answer: str
    level: str
    prompt: str
    nocontext_prompt: str
    paragraph_lines: tuple[str, ...]
    is_gold: tuple[bool, ...]


def _format_context(titles: list[str], sentences: list[list[str]]) -> str:
    parts = []
    for title, sents in zip(titles, sentences):
        para = " ".join(sents)
        parts.append(f"{title}: {para}")
    return "\n".join(parts)


def _gold_indices(titles: list[str], supporting_titles: list[str]) -> list[int]:
    """Context-order indices of the paragraphs named in supporting_facts."""
    gold = set(supporting_titles)
    return [i for i, t in enumerate(titles) if t in gold]


def build_hotpot_items(n: int = 200, seed: int = 0,
                       level: str | None = None,
                       gold_only: bool = False) -> list[HotpotItem]:
    """Sample n bridge questions from HotpotQA validation.

    level: None = all levels, or "easy"/"medium"/"hard".
    gold_only: keep only the gold supporting paragraphs (per supporting_facts)
        instead of all 10. Sampling happens BEFORE formatting, so the same
        (n, seed) yields the same item ids as the full-context variant —
        gold_only changes only the prompt geometry (Exp 15 vs Exp 14).
    """
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    items = [x for x in ds if x["type"] == "bridge"]
    if level:
        items = [x for x in items if x["level"] == level]

    rng = random.Random(seed)
    sample = rng.sample(items, min(n, len(items)))

    result = []
    for ex in sample:  # noqa: keep identical sampling to the spans variant
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        if gold_only:
            idx = _gold_indices(titles, ex["supporting_facts"]["title"])
            titles = [titles[i] for i in idx]
            sentences = [sentences[i] for i in idx]
        ctx = _format_context(titles, sentences)
        prompt = (f"Answer the following question based on the context.\n\n"
                  f"Context:\n{ctx}\n\nQuestion: {ex['question']}\n\n"
                  f"Answer in as few words as possible.")
        result.append(HotpotItem(
            id=ex["id"],
            question=ex["question"],
            answer=ex["answer"],
            level=ex["level"],
            prompt=prompt,
        ))
    return result


def build_hotpot_items_spans(n: int = 200, seed: int = 0) -> list[HotpotItemSpans]:
    """Exp 18 variant: full-distractor items with paragraph span annotations
    and a no-context prompt. Sampling is IDENTICAL to build_hotpot_items
    (same filter, same rng, same n/seed => same item ids, same prompt text)."""
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    items = [x for x in ds if x["type"] == "bridge"]

    rng = random.Random(seed)
    sample = rng.sample(items, min(n, len(items)))

    result = []
    for ex in sample:
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        gold = set(ex["supporting_facts"]["title"])
        lines = tuple(f"{t}: {' '.join(s)}" for t, s in zip(titles, sentences))
        ctx = "\n".join(lines)
        prompt = (f"Answer the following question based on the context.\n\n"
                  f"Context:\n{ctx}\n\nQuestion: {ex['question']}\n\n"
                  f"Answer in as few words as possible.")
        nocontext = (f"Answer the following question.\n\n"
                     f"Question: {ex['question']}\n\n"
                     f"Answer in as few words as possible.")
        result.append(HotpotItemSpans(
            id=ex["id"],
            question=ex["question"],
            answer=ex["answer"],
            level=ex["level"],
            prompt=prompt,
            nocontext_prompt=nocontext,
            paragraph_lines=lines,
            is_gold=tuple(t in gold for t in titles),
        ))
    return result
