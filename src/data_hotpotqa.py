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
    for ex in sample:
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
