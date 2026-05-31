import numpy as np
from src.circuits import previous_token_score, duplicate_token_score


def test_perfect_previous_token_pattern_scores_high():
    n = 8
    A = np.zeros((n, n))
    for i in range(1, n):
        A[i, i - 1] = 1.0
    assert previous_token_score(A) > 0.99


def test_previous_token_uniform_scores_low():
    n = 8
    A = np.ones((n, n))
    A = A / A.sum(axis=1, keepdims=True)
    assert previous_token_score(A) < 0.4


def test_previous_token_offby_one_not_confused():
    # Mass on i-2, not i-1: previous-token score should be ~0.
    n = 8
    A = np.zeros((n, n))
    for i in range(2, n):
        A[i, i - 2] = 1.0
    assert previous_token_score(A) < 1e-9


def test_perfect_duplicate_token_pattern_scores_high():
    # 2nd-copy queries i (i >= prefix+S) put all mass on i-S (the identical
    # token in the first copy).
    S, prefix = 4, 2
    n = prefix + 2 * S
    A = np.zeros((n, n))
    for i in range(prefix + S, n):
        A[i, i - S] = 1.0
    assert duplicate_token_score(A, seq_len=S, prefix_len=prefix) > 0.99


def test_duplicate_token_uniform_scores_low():
    S, prefix = 4, 2
    n = prefix + 2 * S
    A = np.ones((n, n))
    A = A / A.sum(axis=1, keepdims=True)
    assert duplicate_token_score(A, seq_len=S, prefix_len=prefix) < 0.4


def test_duplicate_token_offby_one_not_confused():
    # Mass on the induction offset i-S+1 (not the duplicate offset i-S):
    # duplicate-token score should be ~0.
    S, prefix = 4, 2
    n = prefix + 2 * S
    A = np.zeros((n, n))
    for i in range(prefix + S, n):
        A[i, i - S + 1] = 1.0
    assert duplicate_token_score(A, seq_len=S, prefix_len=prefix) < 1e-9
