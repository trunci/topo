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


def _kinship_item(pid: str, hop: int, names: list[str]) -> Item:
    """One kinship item at the requested hop (1/2/3) over a father-chain context."""
    if hop == 3:
        a, b, c, d = names[:4]
        ctx = f"{a} is {b}'s father. {b} is {c}'s father. {c} is {d}'s father."
        q = f"Who is {d}'s great-grandfather?"
        gold = a
    elif hop == 2:
        a, b, c = names[:3]
        ctx = f"{a} is {b}'s father. {b} is {c}'s father."
        q = f"Who is {c}'s grandfather?"
        gold = a
    else:
        a, b, c = names[:3]
        ctx = f"{a} is {b}'s father. {b} is {c}'s father."
        q = f"Who is {c}'s father?"
        gold = b
    return Item(f"{pid}-h{hop}", "kinship", hop, pid, ctx, q, f"{ctx} {q}", gold)


def _ordering_item(pid: str, hop: int, names: list[str]) -> Item:
    """One ordering item at the requested hop (1/2/3) over a taller-than chain."""
    if hop == 3:
        a, b, c, d = names[:4]
        ctx = f"{a} is taller than {b}. {b} is taller than {c}. {c} is taller than {d}."
        q = f"Among {a}, {b}, {c}, and {d}, who is the tallest?"
        gold = a
    elif hop == 2:
        a, b, c = names[:3]
        ctx = f"{a} is taller than {b}. {b} is taller than {c}."
        q = f"Among {a}, {b}, and {c}, who is the tallest?"
        gold = a
    else:
        a, b, c = names[:3]
        ctx = f"{a} is taller than {b}. {b} is taller than {c}."
        q = f"Who is taller, {a} or {b}?"
        gold = a
    return Item(f"{pid}-h{hop}", "ordering", hop, pid, ctx, q, f"{ctx} {q}", gold)


_ITEM_BUILDERS = {"kinship": _kinship_item, "ordering": _ordering_item}


def build_items(n_per_family: int = 30, hops: tuple[int, ...] = (1, 2, 3),
                seed: int = 0) -> list[Item]:
    """Mixed-hop items (not pairs) for the failure-prediction probe (Exp 7).

    For each family and each k, draw 4 distinct names once and emit one item per
    requested hop from that same name draw (so difficulty varies over a shared
    cast). 3-hop needs 4 names; 1/2-hop use the first 3.
    """
    rng = random.Random(seed)
    items: list[Item] = []
    for family, builder in _ITEM_BUILDERS.items():
        for k in range(n_per_family):
            names = rng.sample(NAMES, 4)
            pid = f"{family}-{k:03d}"
            for hop in hops:
                items.append(builder(pid, hop, names))
    return items


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
