"""Induction-head detection and cheap attention baselines.

Pure functions over attention arrays (numpy), so they are fast to unit-test with
no model dependency. The induction score is the standard repeated-random-sequence
detector: in the second copy of a repeated block, an induction head attends from
position i back to the token that followed the matching token in the first copy,
i.e. offset i - seq_len + 1.
"""
from __future__ import annotations

import numpy as np


def make_repeat_batch(vocab_size: int, seq_len: int, prefix_len: int,
                      n_seqs: int, seed: int = 0) -> np.ndarray:
    """Token-id batch of shape [n_seqs, prefix_len + 2*seq_len].

    Each row = [random prefix] + [random block S] + [same block S]. The two
    S-blocks are identical so induction heads have something to copy.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_seqs):
        prefix = rng.integers(0, vocab_size, size=prefix_len)
        block = rng.integers(0, vocab_size, size=seq_len)
        rows.append(np.concatenate([prefix, block, block]))
    return np.stack(rows).astype(np.int64)


def induction_score(A: np.ndarray, seq_len: int, prefix_len: int) -> float:
    """Mean attention from second-copy positions to the induction offset.

    A: [n, n] attention (rows=query, cols=key), causal. For each query position i
    in the second copy (prefix_len + seq_len .. n-1), the induction target key is
    i - seq_len + 1 (the token after the first-copy match). Returns the mean of
    A[i, i - seq_len + 1] over those positions (0 if none valid).
    """
    n = A.shape[0]
    start = prefix_len + seq_len
    vals = []
    for i in range(start, n):
        j = i - seq_len + 1
        if 0 <= j < n:
            vals.append(A[i, j])
    return float(np.mean(vals)) if vals else 0.0


def attention_distance(A: np.ndarray) -> float:
    """Mean over rows (with mass) of the attention-weighted token distance |i-j|."""
    n = A.shape[0]
    idx = np.arange(n)
    dists = []
    for i in range(n):
        row = A[i]
        s = row.sum()
        if s > 0:
            dists.append(float((row * np.abs(idx - i)).sum() / s))
    return float(np.mean(dists)) if dists else 0.0


def offdiag_mass(A: np.ndarray) -> float:
    """Mean over rows of the total off-diagonal attention mass."""
    off = A.copy().astype(float)
    np.fill_diagonal(off, 0.0)
    return float(off.sum(axis=1).mean())
