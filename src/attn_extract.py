"""Load Qwen2.5-0.5B-Instruct and extract per-head attention matrices.

attn_implementation="eager" is REQUIRED: the default SDPA/flash path does not
return attention weights, so output_attentions would be None.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_named(model_name: str, device: str | None = None,
               dtype=None):
    """Load any causal-LM by name with eager attention (so attentions work).

    dtype defaults to float32 on CPU/MPS and bfloat16 on CUDA.
    Attention matrices are always cast to float32 before topology computation.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device).eval()
    return model, tok, device


def load_model(device: str | None = None):
    """Backwards-compatible loader for the instruct model used by the spike."""
    return load_named(MODEL_NAME, device=device)


def format_prompt(tok, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def get_attentions(model, tok, device, text: str) -> np.ndarray:
    """Return attentions as a numpy array [layers, heads, n, n]."""
    enc = tok(format_prompt(tok, text), return_tensors="pt").to(device)
    out = model(**enc, output_attentions=True)
    atts = [a[0].float().cpu().numpy() for a in out.attentions]  # each [heads, n, n]
    return np.stack(atts)


@torch.no_grad()
def is_correct(model, tok, device, text: str, gold: str, max_new_tokens: int = 12):
    enc = tok(format_prompt(tok, text), return_tensors="pt").to(device)
    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    ans = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return (gold.lower() in ans.lower()), ans
