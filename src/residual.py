"""First-order attention scalar(s) for the residual-information test (Exp 3).

`attention_entropy` is an ADDITIONAL cheap first-order baseline (beyond mean
attention distance and off-diagonal mass) against which we test whether H1
persistence has any residual predictive power for induction score.

Pure numpy on an attention array, so it is fast to unit-test with no model.
"""
from __future__ import annotations

import numpy as np


def attention_entropy(A: np.ndarray) -> float:
    """Mean over query rows of the Shannon entropy (nats) of each row distribution.

    Each row is renormalized to sum to 1 before its entropy is taken. Rows with
    zero total mass are skipped (they contribute nothing and would produce nan).
    A one-hot row has entropy 0; a uniform row over k keys has entropy ln(k).
    """
    A = np.asarray(A, dtype=float)
    entropies = []
    for row in A:
        s = row.sum()
        if s <= 0:
            continue
        p = row / s
        nz = p[p > 0]
        entropies.append(float(-(nz * np.log(nz)).sum()))
    return float(np.mean(entropies)) if entropies else 0.0
