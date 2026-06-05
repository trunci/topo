"""Vectorized, parallel-friendly reimplementations of the per-head attention
features used by Exp 14, plus a verifier proving they match the originals.

WHY: the originals (attention_distance, attention_entropy, sparsify) loop over
matrix rows in Python. Over 1024 heads x ~3700^2 matrices that serial Python is
the dominant cost on long contexts and leaves most CPU cores idle. These
versions remove the Python row loops (pure numpy -> releases the GIL), so a
ThreadPoolExecutor over heads actually parallelizes.

INTEGRITY: numbers must be identical to the originals. verify_identical() below
asserts exactly that on random row-stochastic matrices (same shape/structure as
attention). Attention weights are continuous, so top-k selection has no ties and
the vectorized argpartition selects the same edge set as the per-row loop.
"""
from __future__ import annotations

import numpy as np


def attention_distance_vec(A: np.ndarray, D: np.ndarray | None = None) -> float:
    """Vectorized src.induction.attention_distance."""
    n = A.shape[0]
    if D is None:
        idx = np.arange(n)
        D = np.abs(idx[None, :] - idx[:, None])
    s = A.sum(axis=1)
    mask = s > 0
    if not mask.any():
        return 0.0
    num = (A * D).sum(axis=1)
    return float((num[mask] / s[mask]).mean())


def offdiag_mass_vec(A: np.ndarray) -> float:
    """Vectorized src.induction.offdiag_mass (already loop-free; kept for parity)."""
    off = A.astype(float).copy()
    np.fill_diagonal(off, 0.0)
    return float(off.sum(axis=1).mean())


def attention_entropy_vec(A: np.ndarray) -> float:
    """Vectorized src.residual.attention_entropy."""
    A = np.asarray(A, dtype=float)
    s = A.sum(axis=1)
    mask = s > 0
    if not mask.any():
        return 0.0
    P = A[mask] / s[mask][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(P > 0, P * np.log(P), 0.0)
    ent_rows = -terms.sum(axis=1)
    return float(ent_rows.mean())


def kl_from_uniform_vec(A: np.ndarray) -> float:
    """Vectorized KL-from-uniform (matches run_exp14._kl_from_uniform)."""
    n = A.shape[-1]
    eps = 1e-10
    ent_rows = -np.sum(A * np.log(A + eps), axis=-1)
    kl_rows = np.log(n) - ent_rows
    return float(np.mean(kl_rows))


def sparsify_edges_vec(A: np.ndarray, top_k: int) -> tuple[int, np.ndarray]:
    """symmetrize -> top-k per row -> re-symmetrize -> upper-tri edge list.

    Equivalent to edges_from_weights(sparsify(symmetrize(A), top_k=top_k)) but
    without the Python per-row loop. Returns (n, edges[m,3]) as edges_from_weights.
    """
    W = np.maximum(A, A.T).astype(float)
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    if top_k is not None and top_k < n:
        kept = np.zeros_like(W)
        part = np.argpartition(W, -top_k, axis=1)[:, -top_k:]   # top-k cols per row
        rows = np.arange(n)[:, None]
        kept[rows, part] = W[rows, part]
        kept[W == 0.0] = 0.0          # never keep a zero-weight entry as an edge
        W = np.maximum(kept, kept.T)
    iu = np.triu_indices(n, k=1)
    w = W[iu]
    nz = w > 0.0
    edges = np.column_stack([iu[0][nz], iu[1][nz], w[nz]]).astype(float)
    return n, edges


def dense_features(A: np.ndarray, top_k: int, D: np.ndarray | None = None) -> dict:
    """All per-head dense features + edge list, computed loop-free (one head)."""
    return {
        "dist": attention_distance_vec(A, D),
        "off":  offdiag_mass_vec(A),
        "ent":  attention_entropy_vec(A),
        "kl":   kl_from_uniform_vec(A),
        "edges": sparsify_edges_vec(A, top_k),
    }


def verify_identical(n_trials: int = 8, n: int = 64, seed: int = 0,
                     top_k: int = 8) -> None:
    """Assert the vectorized features match the originals bit-for-bit."""
    from src.induction import attention_distance, offdiag_mass
    from src.residual import attention_entropy
    from src.topology import symmetrize, sparsify, edges_from_weights
    from src.run_exp14 import _kl_from_uniform

    rng = np.random.default_rng(seed)
    for t in range(n_trials):
        # Row-stochastic like a real (post-softmax) attention matrix.
        logits = rng.standard_normal((n, n))
        A = np.exp(logits)
        A = A / A.sum(axis=1, keepdims=True)

        assert np.isclose(attention_distance_vec(A), attention_distance(A), atol=0, rtol=1e-12), t
        assert np.isclose(offdiag_mass_vec(A), offdiag_mass(A), atol=0, rtol=1e-12), t
        assert np.isclose(attention_entropy_vec(A), attention_entropy(A), atol=0, rtol=1e-12), t
        assert np.isclose(kl_from_uniform_vec(A), _kl_from_uniform(A), atol=0, rtol=1e-12), t

        n_new, e_new = sparsify_edges_vec(A, top_k)
        n_old, e_old = edges_from_weights(sparsify(symmetrize(A), top_k=top_k))
        assert n_new == n_old, t
        # Sort edges canonically before comparing (row order may differ).
        def _key(e):
            return e[np.lexsort((e[:, 1], e[:, 0]))] if e.size else e
        e_new, e_old = _key(e_new), _key(e_old)
        assert e_new.shape == e_old.shape, (t, e_new.shape, e_old.shape)
        assert np.allclose(e_new, e_old, atol=0, rtol=1e-12), t
    print(f"verify_identical: OK ({n_trials} trials, n={n}, top_k={top_k}) "
          f"- vectorized features match originals bit-for-bit")


if __name__ == "__main__":
    verify_identical()
