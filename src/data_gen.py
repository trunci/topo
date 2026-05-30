"""Controlled minimal pairs: matched 1-hop / 2-hop questions over identical context."""
from __future__ import annotations

import random
from dataclasses import dataclass

NAMES = [
    "Tom", "Mary", "Sue", "Anna", "Beth", "Cara", "David", "Emma", "Frank",
    "Grace", "Henry", "Iris", "Jack", "Kara", "Liam", "Nina", "Owen", "Pia",
    "Quinn", "Rosa", "Sam", "Tina", "Umar", "Vera", "Will", "Xena", "Yara", "Zane",
]


@dataclass(frozen=True)
class Item:
    id: str
    family: str
    hop: int           # 1 or 2
    pair_id: str
    context: str       # identical within a pair
    question: str
    prompt: str        # context + " " + question
    gold: str


def _kinship_pair(pid: str, names: list[str]) -> list[Item]:
    a, b, c = names
    ctx = f"{a} is {b}'s father. {b} is {c}'s father."
    q2 = f"Who is {c}'s grandfather?"
    q1 = f"Who is {c}'s father?"
    return [
        Item(f"{pid}-h2", "kinship", 2, pid, ctx, q2, f"{ctx} {q2}", a),
        Item(f"{pid}-h1", "kinship", 1, pid, ctx, q1, f"{ctx} {q1}", b),
    ]


def _ordering_pair(pid: str, names: list[str]) -> list[Item]:
    a, b, c = names
    ctx = f"{a} is taller than {b}. {b} is taller than {c}."
    q2 = f"Among {a}, {b}, and {c}, who is the tallest?"
    q1 = f"Who is taller, {a} or {b}?"
    return [
        Item(f"{pid}-h2", "ordering", 2, pid, ctx, q2, f"{ctx} {q2}", a),
        Item(f"{pid}-h1", "ordering", 1, pid, ctx, q1, f"{ctx} {q1}", a),
    ]


_BUILDERS = {"kinship": _kinship_pair, "ordering": _ordering_pair}


def build_pairs(n_per_family: int = 30, seed: int = 0) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []
    for family, builder in _BUILDERS.items():
        for k in range(n_per_family):
            names = rng.sample(NAMES, 3)
            pid = f"{family}-{k:03d}"
            items.extend(builder(pid, names))
    return items
