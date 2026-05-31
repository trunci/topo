"""Masked forward pass via monkeypatched eager attention.

Adds a per-(layer, head) additive bias (0.0 keep, -inf drop) to the attention
scores before softmax. The bias is combined with the model's own causal mask,
so causality is preserved; softmax then renormalizes over the kept keys. The
diagonal is always kept so no query row is fully masked (which would NaN).

Both supported families (Qwen2, used by Exp 1; GPT-2, used by Exp 5) feed their
eager kernel an *additive float* attention mask (0.0 keep, large-negative drop)
and do `attn_weights = attn_weights + attention_mask` before softmax. So in both
cases the per-head bias is simply ADDED to the incoming mask and the original
eager kernel is called unchanged. This keeps masking lossless when the bias is
all-zero and preserves the model's own causal mask. The diagonal is always kept
by keepsets_to_bias so no query row is fully masked (which would NaN).
"""
from __future__ import annotations

import torch
import transformers.models.qwen2.modeling_qwen2 as mq
import transformers.models.gpt2.modeling_gpt2 as mg

_ORIG_EAGER = mq.eager_attention_forward
_ORIG_GPT2_EAGER = mg.eager_attention_forward
_STATE = {"on": False, "bias": None}


def _add_bias(module, attention_mask):
    """Return attention_mask with the per-head additive edge bias added in."""
    am = attention_mask
    if _STATE["on"] and _STATE["bias"] is not None:
        b = _STATE["bias"].get(getattr(module, "_li", None))
        if b is not None:
            add = b.unsqueeze(0)  # [1, H, q, k]
            am = add if am is None else am + add
    return am


def _wrapped_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    am = _add_bias(module, attention_mask)
    return _ORIG_EAGER(module, query, key, value, am, scaling, dropout=dropout, **kw)


def _gpt2_wrapped_eager(module, query, key, value, attention_mask, scaling=None,
                        dropout=0.0, **kw):
    am = _add_bias(module, attention_mask)
    return _ORIG_GPT2_EAGER(module, query, key, value, am, scaling=scaling,
                            dropout=dropout, **kw)


# install the patches once at import time
mq.eager_attention_forward = _wrapped_eager
mg.eager_attention_forward = _gpt2_wrapped_eager


def keepsets_to_bias(keepsets, n, H, L, device):
    """Convert per-(layer, head) keep-sets into additive-bias tensors.

    keepsets: dict (layer, head) -> set[frozenset{i, j}]. A missing (layer, head)
    means keep everything (all-zero bias). Entry (q, k) is 0.0 if the undirected
    edge {q, k} is kept or q == k (diagonal always kept), else -inf.
    Returns dict layer -> tensor [H, n, n].
    """
    biases = {}
    for li in range(L):
        layer_bias = torch.zeros(H, n, n, device=device)
        for h in range(H):
            ks = keepsets.get((li, h), None)
            if ks is None:
                continue  # keep all -> zeros
            m = torch.full((n, n), float("-inf"), device=device)
            idx = torch.arange(n, device=device)
            m[idx, idx] = 0.0  # diagonal always kept
            for e in ks:
                i, j = tuple(e)
                m[i, j] = 0.0
                m[j, i] = 0.0
            layer_bias[h] = m
        biases[li] = layer_bias
    return biases


def _attn_modules(model):
    """Yield (index, attention_submodule) for Qwen2 or GPT-2 style models."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        for i, lyr in enumerate(inner.layers):       # Qwen2 etc.
            yield i, lyr.self_attn
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for i, blk in enumerate(model.transformer.h):  # GPT-2
            yield i, blk.attn
    else:
        raise ValueError(f"unsupported model architecture: {type(model).__name__}")


class MaskedModel:
    """Wraps a Qwen2- or GPT-2-style model to run forward passes with masking."""

    def __init__(self, model):
        self.model = model
        for i, attn in _attn_modules(model):
            attn._li = i

    @torch.no_grad()
    def loss(self, enc, biases=None) -> float:
        """Mean token cross-entropy loss for the given encoding.

        If biases is None, runs an ordinary (unmasked) forward pass.
        """
        if biases is None:
            _STATE["on"] = False
            _STATE["bias"] = None
            out = self.model(**enc, labels=enc["input_ids"])
            return float(out.loss.item())
        _STATE["on"] = True
        _STATE["bias"] = biases
        try:
            out = self.model(**enc, labels=enc["input_ids"])
            return float(out.loss.item())
        finally:
            _STATE["on"] = False
            _STATE["bias"] = None

    @torch.no_grad()
    def logits(self, enc, biases=None):
        """Return logits [1, seq, vocab] with optional masking."""
        if biases is None:
            _STATE["on"] = False
            _STATE["bias"] = None
            return self.model(**enc).logits
        _STATE["on"] = True
        _STATE["bias"] = biases
        try:
            return self.model(**enc).logits
        finally:
            _STATE["on"] = False
            _STATE["bias"] = None
