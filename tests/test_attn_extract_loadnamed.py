import pytest


@pytest.mark.slow
def test_load_named_loads_base_model():
    from src.attn_extract import load_named
    model, tok, device = load_named("Qwen/Qwen2.5-0.5B")
    assert model.config.num_attention_heads == 14
    assert len(model.model.layers) == 24
    assert device in ("mps", "cpu")
    # eager attention so output_attentions works
    assert model.config._attn_implementation == "eager"
