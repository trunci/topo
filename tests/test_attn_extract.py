import numpy as np
import pytest

slow = pytest.mark.slow


@slow
def test_get_attentions_shape_and_range():
    from src.attn_extract import load_model, get_attentions
    model, tok, device = load_model()
    atts = get_attentions(model, tok, device, "Tom is Mary's father. Who is Mary's father?")
    # [layers, heads, n, n]
    assert atts.ndim == 4
    assert atts.shape[2] == atts.shape[3]
    # attention rows are probability distributions -> each row sums to ~1
    row_sums = atts.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-3)


@slow
def test_is_correct_runs_and_returns_bool_and_text():
    from src.attn_extract import load_model, is_correct
    model, tok, device = load_model()
    ok, ans = is_correct(model, tok, device, "Tom is Mary's father. Who is Mary's father?", "Tom")
    assert isinstance(ok, bool)
    assert isinstance(ans, str)
