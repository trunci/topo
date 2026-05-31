import pytest

from src.data_ioi import (build_ioi_items, ioi_positions, io_s_token_ids,
                          single_token_names)


class FakeTok:
    """Minimal whitespace tokenizer: each space-delimited word -> a stable id.

    ' Name' -> single id (leading-space convention), so single-token names work.
    """
    def __init__(self):
        self.vocab = {}

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab) + 1)

    def __call__(self, text):
        # mimic GPT-2 leading-space tokenization: split, first word bare, rest " w"
        words = text.split(" ")
        ids = []
        for i, w in enumerate(words):
            if w == "":
                continue
            ids.append(self._id((" " if i > 0 else "") + w))
        return {"input_ids": ids}


def test_single_token_names_all_pass_fake_tok():
    tok = FakeTok()
    names = single_token_names(tok, ["John", "Mary"])
    assert names == ["John", "Mary"]


def test_positions_io_is_first_name_end_is_last():
    tok = FakeTok()
    items = build_ioi_items(tok, n=1, seed=0)
    it = items[0]
    ids = tok(it.text)["input_ids"]
    pos = ioi_positions(tok, it)
    assert pos is not None
    io_pos, end_pos = pos
    assert end_pos == len(ids) - 1
    # io_pos token equals the IO name's id, and it's the FIRST occurrence
    io_id = tok(" " + it.io_name)["input_ids"][0]
    assert ids[io_pos] == io_id
    assert io_pos == min(i for i, t in enumerate(ids) if t == io_id)
    # IO (A) is introduced before S's second mention; io_pos must be early
    assert io_pos < end_pos


def test_io_and_s_distinct_tokens():
    tok = FakeTok()
    it = build_ioi_items(tok, n=1, seed=1)[0]
    io_id, s_id = io_s_token_ids(tok, it)
    assert io_id != s_id


def test_template_has_io_before_to():
    tok = FakeTok()
    it = build_ioi_items(tok, n=1, seed=2)[0]
    assert it.text.startswith(f"When {it.io_name} and {it.s_name}")
    assert it.text.endswith(" to")
    assert it.io_name != it.s_name


def test_deterministic():
    tok = FakeTok()
    a = build_ioi_items(tok, n=5, seed=7)
    b = build_ioi_items(tok, n=5, seed=7)
    assert [i.text for i in a] == [i.text for i in b]
