"""IOI (indirect-object identification) prompts with ground-truth name-mover edges.

Canonical template: "When {A} and {B} went to the {place}, {B} gave a {obj} to"
-> correct next token is " {A}" (the indirect object / IO). The name-mover circuit
moves the IO name from its first-occurrence position to END; the ground-truth edge is
(END -> IO_pos). A and B are single-token first names (verified against the tokenizer).

Pure data construction + a position-finder that takes the tokenizer; no model needed,
so the position logic is unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass

NAMES = ["John", "Mary", "Tom", "James", "Anna", "Paul", "Sarah", "Mark",
         "Lucy", "David", "Emma", "Peter", "Laura", "Mike", "Susan", "Alan",
         "Kevin", "Julia", "Brian", "Karen"]
PLACES = ["store", "park", "school", "office", "garden", "station"]
OBJECTS = ["drink", "book", "ball", "ring", "letter", "snack"]


@dataclass(frozen=True)
class IOIItem:
    id: str
    text: str
    io_name: str       # A -- the answer (indirect object)
    s_name: str        # B -- the subject (repeated)
    place: str
    obj: str


def single_token_names(tok, names=NAMES):
    """Names that tokenize to a single id with a leading space (so positions are clean)."""
    out = []
    for nm in names:
        if len(tok(" " + nm)["input_ids"]) == 1:
            out.append(nm)
    return out


def build_ioi_items(tok, n=48, seed=0):
    """n seeded IOI items using only single-token names."""
    import random
    rng = random.Random(seed)
    names = single_token_names(tok)
    if len(names) < 2:
        raise ValueError("need >=2 single-token names for IOI")
    items = []
    for k in range(n):
        a, b = rng.sample(names, 2)
        place = rng.choice(PLACES)
        obj = rng.choice(OBJECTS)
        text = f"When {a} and {b} went to the {place}, {b} gave a {obj} to"
        items.append(IOIItem(f"ioi-{k:03d}", text, a, b, place, obj))
    return items


def ioi_positions(tok, item):
    """Return (io_pos, end_pos) token indices for an item.

    io_pos = first occurrence of the IO name token; end_pos = last token (the query
    position whose next-token prediction should be the IO name).
    """
    ids = tok(item.text)["input_ids"]
    io_id = tok(" " + item.io_name)["input_ids"][0]
    io_positions = [i for i, t in enumerate(ids) if t == io_id]
    if not io_positions:
        return None
    return io_positions[0], len(ids) - 1


def io_s_token_ids(tok, item):
    """Token ids of the IO and S names (with leading space), for the logit-margin readout."""
    return tok(" " + item.io_name)["input_ids"][0], tok(" " + item.s_name)["input_ids"][0]
