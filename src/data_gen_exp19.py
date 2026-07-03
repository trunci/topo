"""Experiment 19 rung generators (pre-registered in FINDINGS_exp19.md).

R1: canonical exp13 templates (build_items, verbatim).
R2: handcrafted paraphrase bank — same casts, same facts, surface form drawn
    per sentence from ~6 semantically-exact variants; questions from a
    3-variant bank. Sentence order fixed.
R3: LLM narrative rewrite of the R1 context (canonical question).
R4: R3 context + LLM-rephrased question (relation term pinned).
R5: HotpotQA gold-only items (Exp 15's, via build_hotpot_items).

R3/R4 rewrite prompts + validators live here (pure functions); the actual
rewriting runs in run_exp19 stage "rewrite" with Mistral-7B and is persisted
to results/exp19_rewrites.jsonl.
"""
from __future__ import annotations

import random
from dataclasses import replace

from src.data_gen import Item, build_items

HOPS = (4, 5)
N_PER_FAMILY = 30
SEED = 0

# ---- R2: paraphrase banks (every variant states EXACTLY the same fact) ----

KIN_SENT = (
    "{a} is {b}'s father.",
    "{b}'s father is {a}.",
    "The father of {b} is {a}.",
    "{a} is the father of {b}.",
    "{b} has {a} for a father.",
    "{a} fathered {b}.",
)
ORD_SENT = (
    "{a} is taller than {b}.",
    "{b} is shorter than {a}.",
    "{a} stands taller than {b}.",
    "{b} is not as tall as {a}.",
    "Compared to {b}, {a} is taller.",
    "In height, {a} exceeds {b}.",
)
KIN_Q = (
    "Who is {x}'s {rel}?",
    "What is the name of {x}'s {rel}?",
    "Name {x}'s {rel}.",
)
ORD_Q = (
    "Among {lst}, who is the tallest?",
    "Of {lst}, who is the tallest?",
    "Who is the tallest among {lst}?",
)
KIN_REL = {4: "great-great-grandfather", 5: "great-great-great-grandfather"}


def _chain_pairs(names, hop):
    """(a, b) pairs meaning 'a relates-to b' along the chain, R1 order."""
    return [(names[i], names[i + 1]) for i in range(hop)]


def _r2_item(it: Item, rng: random.Random) -> Item:
    names = _names_of(it)
    pairs = _chain_pairs(names, it.hop)
    bank = KIN_SENT if it.family == "kinship" else ORD_SENT
    ctx = " ".join(rng.choice(bank).format(a=a, b=b) for a, b in pairs)
    if it.family == "kinship":
        q = rng.choice(KIN_Q).format(x=names[-1], rel=KIN_REL[it.hop])
    else:
        lst = ", ".join(names[:-1]) + f", and {names[-1]}"
        q = rng.choice(ORD_Q).format(lst=lst)
    return replace(it, id=it.id + "-r2", context=ctx, question=q,
                   prompt=f"{ctx} {q}")


def _names_of(it: Item) -> list[str]:
    """Recover the cast, in chain order, from the canonical R1 context."""
    names = []
    for token in it.context.replace("'s", " ").replace(".", " ").split():
        if token[0].isupper() and token not in names and token not in (
                "Who", "Among", "Compared", "In", "The", "Of", "Name", "What"):
            names.append(token)
    assert len(names) == it.hop + 1, (it.id, names, it.context)
    return names


def build_r1(n_per_family=N_PER_FAMILY, hops=HOPS, seed=SEED) -> list[Item]:
    return build_items(n_per_family=n_per_family, hops=hops, seed=seed)


def build_r2(n_per_family=N_PER_FAMILY, hops=HOPS, seed=SEED) -> list[Item]:
    rng = random.Random(seed + 1000)
    return [_r2_item(it, rng) for it in build_r1(n_per_family, hops, seed)]


# ---- R3/R4: rewrite prompts and validators ----

def rewrite_context_prompt(it: Item) -> str:
    return (
        "Rewrite the following facts as one short natural paragraph of "
        "flowing prose (2-4 sentences). Keep every person's name exactly as "
        "written. State exactly the same facts and nothing else — do not add, "
        "remove, or imply any other relationship or detail. Output only the "
        f"paragraph.\n\nFacts: {it.context}"
    )


def rewrite_question_prompt(it: Item) -> str:
    rel = KIN_REL[it.hop] if it.family == "kinship" else "tallest"
    return (
        "Rephrase this question so it sounds natural and conversational, "
        "without changing its meaning. Keep every person's name exactly as "
        f"written, and keep the exact term '{rel}'. Output only the "
        f"question.\n\nQuestion: {it.question}"
    )


def validate_context(it: Item, text: str) -> bool:
    names = _names_of(it)
    rel = KIN_REL[it.hop] if it.family == "kinship" else "tallest"
    t = " ".join(text.split())
    if not (20 <= len(t) <= 600):
        return False
    if any(n not in t for n in names):
        return False
    if rel.lower() in t.lower():  # relation-term leak would shortcut the hop
        return False
    return True


def validate_question(it: Item, text: str) -> bool:
    rel = KIN_REL[it.hop] if it.family == "kinship" else "tallest"
    t = " ".join(text.split())
    if not (5 <= len(t) <= 300) or "?" not in t:
        return False
    if rel.lower() not in t.lower():
        return False
    # the subject of the question must survive the rephrase
    subject = _names_of(it)[-1] if it.family == "kinship" else None
    if subject and subject not in t:
        return False
    return True
