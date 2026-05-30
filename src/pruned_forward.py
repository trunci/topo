"""Masked forward pass via monkeypatched Qwen2 eager attention.

Adds a per-(layer, head) additive bias (0.0 keep, -inf drop) to the attention
scores before softmax. The bias is combined with the model's own causal mask,
so causality is preserved; softmax then renormalizes over the kept keys. The
diagonal is always kept so no query row is fully masked (which would NaN).
"""
from __future__ import annotations

import torch
import transformers.models.qwen2.modeling_qwen2 as mq

_ORIG_EAGER = mq.eager_attention_forward
_STATE = {"on": False, "bias": None}


def _wrapped_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    am = attention_mask
    if _STATE["on"] and _STATE["bias"] is not None:
        b = _STATE["bias"].get(getattr(module, "_li", None))
        if b is not None:
            add = b.unsqueeze(0)  # [1, H, q, k]
            am = add if am is None else am + add
    return _ORIG_EAGER(module, query, key, value, am, scaling, dropout=dropout, **kw)


# install the patch once at import time
mq.eager_attention_forward = _wrapped_eager


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


class MaskedModel:
    """Wraps a Qwen2 model to run forward passes with optional attention masking."""

    def __init__(self, model):
        self.model = model
        for i, lyr in enumerate(model.model.layers):
            lyr.self_attn._li = i

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
