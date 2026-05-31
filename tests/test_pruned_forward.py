import numpy as np
import pytest

slow = pytest.mark.slow


def _all_keep_biases(n, H, L, device):
    import torch
    return {li: torch.zeros(H, n, n, device=device) for li in range(L)}


@slow
def test_all_keep_is_lossless():
    import torch
    from src.attn_extract import load_named
    from src.pruned_forward import MaskedModel
    model, tok, device = load_named("Qwen/Qwen2.5-0.5B")
    mm = MaskedModel(model)
    enc = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(device)
    n = enc["input_ids"].shape[1]
    H, L = model.config.num_attention_heads, model.config.num_hidden_layers
    # unmasked loss
    base = mm.loss(enc)
    # all-keep mask = zero bias -> identical loss
    masked = mm.loss(enc, biases=_all_keep_biases(n, H, L, device))
    assert abs(base - masked) < 1e-4


@slow
def test_aggressive_mask_changes_loss():
    import torch
    from src.attn_extract import load_named
    from src.pruned_forward import MaskedModel
    model, tok, device = load_named("Qwen/Qwen2.5-0.5B")
    mm = MaskedModel(model)
    enc = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(device)
    n = enc["input_ids"].shape[1]
    H, L = model.config.num_attention_heads, model.config.num_hidden_layers
    eye = torch.eye(n, device=device, dtype=torch.bool)
    biases = {li: torch.zeros(H, n, n, device=device) for li in range(L)}
    m0 = torch.full((H, n, n), float("-inf"), device=device)
    m0[:, eye] = 0.0
    biases[0] = m0  # layer 0: keep only the diagonal
    base = mm.loss(enc)
    masked = mm.loss(enc, biases=biases)
    assert abs(base - masked) > 1e-3


def test_keepsets_to_bias_keeps_diagonal_and_kept_edges():
    import torch
    from src.pruned_forward import keepsets_to_bias
    # 1 layer, 1 head, n=3; keep only edge {0,1}
    keepsets = {(0, 0): {frozenset({0, 1})}}
    biases = keepsets_to_bias(keepsets, n=3, H=1, L=1, device="cpu")
    b = biases[0][0]  # [n, n]
    # diagonal always kept (0.0)
    assert b[0, 0].item() == 0.0 and b[1, 1].item() == 2 * 0.0
    # kept edge {0,1}: both directions 0.0
    assert b[0, 1].item() == 0.0 and b[1, 0].item() == 0.0
    # non-kept edge {0,2}: -inf
    assert b[0, 2].item() == float("-inf")
    assert b[2, 0].item() == float("-inf")


def test_keepsets_to_bias_missing_head_keeps_all():
    import torch
    from src.pruned_forward import keepsets_to_bias
    # no entry for (0,0) -> that head keeps everything (all zeros)
    biases = keepsets_to_bias({}, n=3, H=1, L=1, device="cpu")
    assert torch.all(biases[0][0] == 0.0)


# ---- directed ablation_to_bias (Exp 6) --------------------------------------

def test_ablation_to_bias_only_named_directed_edges_ablated():
    import torch
    from src.pruned_forward import ablation_to_bias
    # ablate the directed edge (2 -> 0) only, in head 0
    abl = {(0, 0): {(2, 0)}}
    b = ablation_to_bias(abl, n=3, H=1, L=1, device="cpu")
    layer0 = b[0]
    assert layer0.shape == (1, 3, 3)
    # the named directed edge is ablated
    assert layer0[0, 2, 0].item() == float("-inf")
    # the reverse direction is NOT ablated (directed, not symmetric)
    assert layer0[0, 0, 2].item() == 0.0
    # everything else is kept (zero) -- only the named edge changes
    assert layer0[0, 1, 0].item() == 0.0
    assert layer0[0, 2, 1].item() == 0.0
    # exactly one -inf entry
    assert int(torch.isinf(layer0).sum().item()) == 1


def test_ablation_to_bias_never_ablates_diagonal():
    import torch
    from src.pruned_forward import ablation_to_bias
    # even if a self-loop is (perversely) requested, the diagonal stays kept
    abl = {(0, 0): {(1, 1), (2, 0)}}
    b = ablation_to_bias(abl, n=3, H=1, L=1, device="cpu")
    layer0 = b[0]
    assert layer0[0, 1, 1].item() == 0.0  # diagonal never ablated
    assert layer0[0, 2, 0].item() == float("-inf")


def test_ablation_to_bias_missing_head_keeps_all():
    import torch
    from src.pruned_forward import ablation_to_bias
    abl = {(0, 0): {(2, 0)}}
    b = ablation_to_bias(abl, n=3, H=2, L=1, device="cpu")
    # head 1 has no ablate-set -> all zeros (keep everything)
    assert b[0][1].sum().item() == 0.0
    # an unmasked head/edge stays zero
    assert b[0][0, 1, 0].item() == 0.0


# ---- GPT-2 masking (Exp 5) --------------------------------------------------

@slow
def test_gpt2_all_keep_is_lossless():
    import torch
    from src.attn_extract import load_named
    from src.pruned_forward import MaskedModel
    torch.set_num_threads(2)
    model, tok, device = load_named("gpt2", device="cpu")
    mm = MaskedModel(model)
    enc = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(device)
    n = enc["input_ids"].shape[1]
    H, L = model.config.n_head, model.config.n_layer
    base = mm.loss(enc)
    masked = mm.loss(enc, biases=_all_keep_biases(n, H, L, device))
    assert abs(base - masked) < 1e-4


@slow
def test_gpt2_aggressive_mask_changes_loss():
    import torch
    from src.attn_extract import load_named
    from src.pruned_forward import MaskedModel
    torch.set_num_threads(2)
    model, tok, device = load_named("gpt2", device="cpu")
    mm = MaskedModel(model)
    enc = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(device)
    n = enc["input_ids"].shape[1]
    H, L = model.config.n_head, model.config.n_layer
    eye = torch.eye(n, device=device, dtype=torch.bool)
    biases = {li: torch.zeros(H, n, n, device=device) for li in range(L)}
    m0 = torch.full((H, n, n), float("-inf"), device=device)
    m0[:, eye] = 0.0
    biases[0] = m0  # layer 0: keep only the diagonal
    base = mm.loss(enc)
    masked = mm.loss(enc, biases=biases)
    assert abs(base - masked) > 1e-3
