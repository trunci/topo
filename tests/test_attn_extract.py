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


@slow
def test_load_named_loads_base_model():
    from src.attn_extract import load_named
    model, tok, device = load_named("Qwen/Qwen2.5-0.5B")
    assert model.config.num_hidden_layers == 24
    assert model.config.num_attention_heads == 14
    # base model should produce logits for a simple input
    import torch
    enc = tok("The capital of France is", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    assert out.logits.shape[0] == 1
