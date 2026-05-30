import pytest


def test_kinship_texts_returns_prompts_and_gold():
    from src.eval_data import kinship_eval_items
    items = kinship_eval_items(n_per_family=3, seed=0)
    assert len(items) > 0
    for it in items:
        assert "prompt" in it and "gold" in it
        assert isinstance(it["prompt"], str) and isinstance(it["gold"], str)


@pytest.mark.slow
def test_wikitext_lines_returns_nonempty_text():
    from src.eval_data import wikitext_lines
    lines = wikitext_lines(n=5, min_chars=50)
    assert len(lines) == 5
    assert all(isinstance(t, str) and len(t) >= 50 for t in lines)
