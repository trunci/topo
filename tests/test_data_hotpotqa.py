from src.data_hotpotqa import _format_context, _gold_indices


TITLES = ["A", "B", "C", "D"]
SENTS = [["a1.", "a2."], ["b1."], ["c1.", "c2."], ["d1."]]


def test_gold_indices_picks_supporting_titles_in_context_order():
    assert _gold_indices(TITLES, ["C", "A"]) == [0, 2]


def test_gold_indices_ignores_titles_missing_from_context():
    assert _gold_indices(TITLES, ["B", "ZZZ"]) == [1]


def test_format_context_with_gold_subset():
    idx = _gold_indices(TITLES, ["D", "B"])
    out = _format_context([TITLES[i] for i in idx], [SENTS[i] for i in idx])
    assert out == "B: b1.\nD: d1."
    assert "A:" not in out and "C:" not in out
