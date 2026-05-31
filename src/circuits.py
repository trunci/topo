"""Additional circuit detectors (previous-token, duplicate-token).

Pure numpy over an [n, n] causal attention matrix (rows=query, cols=key), so they
are fast to unit-test with no model dependency. These complement
`induction.induction_score` for the Experiment 4 generalization grid.
"""
from __future__ import annotations

import numpy as np


def previous_token_score(A: np.ndarray) -> float:
    """Mean over query rows i>=1 of A[i, i-1].

    A previous-token head puts its mass on the immediately preceding key, so the
    perfect pattern (mass exactly on i-1) scores ~1 and uniform attention scores
    low. Returns 0 if there are no valid query rows.
    """
    n = A.shape[0]
    vals = [A[i, i - 1] for i in range(1, n)]
    return float(np.mean(vals)) if vals else 0.0


def duplicate_token_score(A: np.ndarray, seq_len: int, prefix_len: int) -> float:
    """Mean over 2nd-copy query positions i of A[i, i-seq_len].

    For each query position i in the second copy (i >= prefix_len + seq_len), a
    duplicate-token head attends to the identical token in the first copy, at key
    i - seq_len. Returns the mean of A[i, i-seq_len] over those positions (0 if
    none valid). Distinct from induction (offset i-seq_len+1).
    """
    n = A.shape[0]
    start = prefix_len + seq_len
    vals = []
    for i in range(start, n):
        j = i - seq_len
        if 0 <= j < n:
            vals.append(A[i, j])
    return float(np.mean(vals)) if vals else 0.0
