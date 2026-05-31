"""Masked forward pass via monkeypatched eager attention.

Adds a per-(layer, head) additive bias (0.0 keep, -inf drop) to the attention
scores before softmax. The bias is combined with the model's own causal mask,
so causality is preserved; softmax then renormalizes over the kept keys. The
diagonal is always kept so no query row is fully masked (which would NaN).

Two model families are supported, patched independently at import time:

* Qwen2 (used by Exp 1) takes an *additive float* attention mask, so the bias is
  simply added to the incoming mask and forwarded to the original eager kernel.
* GPT-2 (used by Exp 5) takes a *boolean* causal mask and its installed eager
  kernel recomputes the raw scores after applying the mask (discarding any
  pre-softmax edits), so we cannot delegate. Instead the GPT-2 wrapper does a
  correct eager attention itself: scaled scores -> causal mask + additive bias
  -> softmax -> dropout -> value matmul, returning (attn_output, attn_weights)
  in the exact layout the model expects.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import transformers.models.qwen2.modeling_qwen2 as mq
import transformers.models.gpt2.modeling_gpt2 as mg

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


def _gpt2_wrapped_eager(module, query, key, value, attention_mask, head_mask=None,
                        scaling=None, dropout=0.0, **kw):
    """Correct eager attention for GPT-2 with an optional per-head additive bias.

    query/key/value: [batch, n_heads, q, d]. attention_mask (when not None) is a
    boolean causal mask [*, *, q, k] (True = attend). We build the additive
    causal bias ourselves so the per-head -inf edge bias survives to softmax.
    """
    if scaling is None:
        scaling = module.scaling

    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    q_len, k_len = query.shape[-2], key.shape[-2]
    neg = torch.finfo(attn_weights.dtype).min

    # additive causal mask (0 keep, neg drop)
    if attention_mask is not None:
        bool_mask = attention_mask[:, :, :, :k_len].to(torch.bool)
        attn_weights = attn_weights.masked_fill(~bool_mask, neg)
    elif getattr(module, "is_causal", False) and q_len > 1:
        causal = torch.ones(q_len, k_len, dtype=torch.bool,
                            device=query.device).tril()
        attn_weights = attn_weights.masked_fill(~causal[None, None], neg)

    # per-(layer,head) additive edge bias (0 keep, -inf drop)
    if _STATE["on"] and _STATE["bias"] is not None:
        b = _STATE["bias"].get(getattr(module, "_li", None))
        if b is not None:
            attn_weights = attn_weights + b.unsqueeze(0)  # [1, H, q, k]

    attn_weights = F.softmax(attn_weights, dim=-1)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    attn_weights = attn_weights.to(value.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)
    return attn_output, attn_weights


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
