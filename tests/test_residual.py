import numpy as np
from src.residual import attention_entropy


def test_one_hot_rows_zero_entropy():
    # Each row puts all mass on one key -> entropy 0.
    A = np.eye(5)
    assert abs(attention_entropy(A)) < 1e-9


def test_uniform_row_max_entropy():
    # A single uniform row over k keys has Shannon entropy ln(k) nats.
    n = 4
    A = np.ones((n, n)) / n
    # mean over rows of ln(n)
    assert abs(attention_entropy(A) - np.log(n)) < 1e-9


def test_higher_entropy_for_flatter_distribution():
    peaked = np.array([[0.9, 0.1], [0.9, 0.1]])
    flat = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert attention_entropy(flat) > attention_entropy(peaked)


def test_renormalizes_unnormalized_rows():
    # Rows that do not sum to 1 are renormalized before entropy.
    A = np.array([[2.0, 2.0], [2.0, 2.0]])  # each row uniform after norm
    assert abs(attention_entropy(A) - np.log(2)) < 1e-9


def test_skips_zero_mass_rows():
    # A zero row contributes nothing (and does not produce nan).
    A = np.array([[0.0, 0.0], [0.5, 0.5]])
    # only the second row counts -> ln(2)
    assert abs(attention_entropy(A) - np.log(2)) < 1e-9
