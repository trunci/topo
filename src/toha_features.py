"""TOHA-style topological-divergence features (Experiment 16).

Implements the MTop-Div(R, P) score of Bazarova et al. (ACL 2026,
arXiv:2504.10063) for one attention head:

* tokens = prompt P (positions 0..p-1) + response R (positions p..n-1)
* complete undirected graph, edge weights 1 - w_ij (attention as similarity;
  w_ij = max of the two attention directions, only one of which exists under
  the causal mask)
* prompt-prompt edges are zeroed, which contracts P to a single component at
  filtration 0 -- so the H0 barcode sum equals the total weight of the minimal
  spanning forest attaching each response token to the prompt cluster
* normalized by |R| (TOHA's length normalization)

We therefore never need the full n x n matrix: only the response ROWS of each
head's attention. `get_response_rows` captures exactly that [heads, |R|, n]
slice per layer with the same low-memory hook trick as attn_extract.

High MTop-Div = response tokens attach to the prompt expensively = weak
grounding; TOHA reports this correlates with hallucination.
"""
from __future__ import annotations

import numpy as np
import torch

from src.attn_extract import _attn_layers, format_prompt


def mtop_div(resp_rows: np.ndarray, p: int) -> float:
    """MTop-Div for one head from its response-row attention slice.

    resp_rows: [R, n] attention rows for response positions p..n-1 (causal:
    row i may only attend to columns <= p + i).
    p: number of prompt tokens.

    Returns the total weight of the minimum spanning tree over the contracted
    graph {P-supernode} + response nodes, normalized by R.
    """
    R = resp_rows.shape[0]
    if R == 0:
        return 0.0
    # distance from each response token to the prompt supernode:
    # 1 - strongest attention onto any prompt column
    d_to_p = 1.0 - resp_rows[:, :p].max(axis=1)

    # response-response distances: undirected weight = max of both directions
    block = resp_rows[:, p:p + R]                # [R, R], lower-triangular-ish
    w = np.maximum(block, block.T)
    d_rr = 1.0 - w

    # dense Prim over R+1 nodes (node 0 = prompt supernode)
    D = np.empty((R + 1, R + 1))
    D[0, 0] = 0.0
    D[0, 1:] = d_to_p
    D[1:, 0] = d_to_p
    D[1:, 1:] = d_rr
    in_tree = np.zeros(R + 1, dtype=bool)
    in_tree[0] = True
    best = D[0].copy()
    best[0] = np.inf
    total = 0.0
    for _ in range(R):
        j = int(np.argmin(np.where(in_tree, np.inf, best)))
        total += float(best[j])
        in_tree[j] = True
        best = np.minimum(best, D[j])
    return total / R


@torch.no_grad()
def get_response_rows(model, tok, device, prompt_text: str,
                      generated_ids: torch.Tensor):
    """Teacher-forced forward over prompt + generated answer, capturing only
    the response rows of every head's attention.

    generated_ids: 1-D tensor of the generated answer token ids (no specials).
    Returns (rows_per_layer, p, n) where rows_per_layer is a list of
    [heads, R, n] float32 arrays.
    """
    enc = tok(format_prompt(tok, prompt_text), return_tensors="pt").to(device)
    p = int(enc["input_ids"].shape[1])
    ids = torch.cat([enc["input_ids"][0], generated_ids.to(device)])[None, :]
    n = int(ids.shape[1])

    captured: dict[int, np.ndarray] = {}
    handles = []

    def _make_hook(idx):
        def hook(module, inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                captured[idx] = out[1][0, :, p:, :].float().cpu().numpy()
                return (out[0], None) + tuple(out[2:])
            return out
        return hook

    layers = _attn_layers(model)
    for i, attn in enumerate(layers):
        handles.append(attn.register_forward_hook(_make_hook(i)))
    try:
        model(input_ids=ids, output_attentions=True)
    finally:
        for h in handles:
            h.remove()
    rows = [captured[i] for i in range(len(layers))]
    return rows, p, n


def toha_head_features(rows_per_layer, p: int) -> dict:
    """Per-head MTop-Div plus response-row first-order controls.

    Heads are flattened layer-major (layer0 h0..hH, layer1 ...), matching the
    exp15 `ph_*` convention.
    """
    mtd, ent = [], []
    eps = 1e-10
    for layer_rows in rows_per_layer:            # [heads, R, n]
        for h in range(layer_rows.shape[0]):
            A = layer_rows[h]
            mtd.append(mtop_div(A, p))
            ent.append(float(np.mean(-np.sum(A * np.log(A + eps), axis=-1))))
    mtd_a = np.array(mtd)
    return {
        "mtd_mean": float(mtd_a.mean()),
        "mtd_max": float(mtd_a.max()),
        "mtd_std": float(mtd_a.std()),
        "ctrl_resp_entropy": float(np.mean(ent)),
        "ph_mtd": [float(v) for v in mtd],
        "ph_resp_ent": [float(v) for v in ent],
    }
