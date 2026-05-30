import numpy as np
from src.induction import (
    make_repeat_batch, induction_score, attention_distance, offdiag_mass,
)


def test_perfect_induction_pattern_scores_high():
    n, S, prefix = 8, 4, 0
    A = np.zeros((n, n))
    for i in range(S, n):
        A[i, i - S + 1] = 1.0
    score = induction_score(A, seq_len=S, prefix_len=prefix)
    assert score > 0.99


def test_uniform_attention_scores_low():
    n, S, prefix = 8, 4, 0
    A = np.ones((n, n))
    A = A / A.sum(axis=1, keepdims=True)
    score = induction_score(A, seq_len=S, prefix_len=prefix)
    assert score < 0.4


def test_attention_distance_diagonal_is_zero():
    A = np.eye(5)
    assert attention_distance(A) == 0.0


def test_attention_distance_fixed_offset():
    n = 5
    A = np.zeros((n, n))
    for i in range(2, n):
        A[i, i - 2] = 1.0
    assert abs(attention_distance(A) - 2.0) < 1e-9


def test_offdiag_mass_counts_offdiagonal_only():
    A = np.array([[0.5, 0.5], [0.0, 1.0]])
    assert abs(offdiag_mass(A) - 0.25) < 1e-9


def test_make_repeat_batch_halves_are_identical():
    batch = make_repeat_batch(vocab_size=100, seq_len=5, prefix_len=2, n_seqs=3, seed=0)
    assert batch.shape == (3, 2 + 5 + 5)
    for row in batch:
        first = row[2:2 + 5]
        second = row[2 + 5:2 + 10]
        assert np.array_equal(first, second)


def test_make_repeat_batch_deterministic():
    a = make_repeat_batch(vocab_size=100, seq_len=5, prefix_len=2, n_seqs=3, seed=42)
    b = make_repeat_batch(vocab_size=100, seq_len=5, prefix_len=2, n_seqs=3, seed=42)
    assert np.array_equal(a, b)
