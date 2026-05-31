"""Discrete Morse theory (Forman) on a head's flag complex.

Greedy reduction-based discrete gradient vector field: repeatedly pair a cell
that has exactly one remaining coface with that coface (a "reduction"); when no
such pair exists, mark a highest-dimension remaining cell critical. Critical
1-cells are H1 generators; vertex-edge matched pairs form a spanning forest.
The keep-set retains the topologically essential edges.
"""
from __future__ import annotations

import numpy as np

from src.topology import symmetrize, sparsify, build_simplex_tree


def _morse_match(simplices):
    """Greedy Morse matching. Returns (matched_pairs, critical_cells).

    simplices: list of tuples (each a sorted tuple of vertex ids).
    """
    sset = set(simplices)
    dim = {s: len(s) - 1 for s in simplices}
    # facets[s] = codim-1 faces of s that are themselves in the complex
    facets = {s: [tuple(sorted(set(s) - {v})) for v in s] if len(s) > 1 else []
              for s in simplices}
    cofaces = {s: [] for s in simplices}
    for s in simplices:
        for f in facets[s]:
            if f in cofaces:
                cofaces[f].append(s)

    remaining = set(simplices)
    matched = []
    critical = []

    def remaining_cofaces(a):
        return [c for c in cofaces[a] if c in remaining]

    while remaining:
        reduction = None
        for a in remaining:
            rc = remaining_cofaces(a)
            if len(rc) == 1:
                reduction = (a, rc[0])
                break
        if reduction is not None:
            a, b = reduction
            matched.append((a, b))
            remaining.discard(a)
            remaining.discard(b)
        else:
            c = max(remaining, key=lambda s: dim[s])
            critical.append(c)
            remaining.discard(c)

    return matched, critical


def morse_keep(A: np.ndarray, top_k: int | None = 8, weight_floor: float = 0.0,
               symmetrized: bool = False, min_budget_frac: float = 0.0) -> set:
    """Attention matrix -> set of kept edges (frozenset{i, j}) via discrete Morse.

    Keep: critical 1-cells (cycle generators) + vertex-edge matched pairs
    (spanning forest). Optionally top up to min_budget_frac of the edge count
    with the highest-weight not-yet-kept edges.
    """
    W = A.astype(float) if symmetrized else symmetrize(A)
    W = sparsify(W, top_k=top_k, weight_floor=weight_floor)

    st = build_simplex_tree(W)
    simplices = [tuple(sorted(s)) for s, _ in st.get_simplices()]

    matched, critical = _morse_match(simplices)

    keep = set()
    for a, b in matched:
        if len(a) == 1 and len(b) == 2:        # vertex-edge => spanning-forest edge
            keep.add(frozenset(b))
    for c in critical:
        if len(c) == 2:                         # critical 1-cell => cycle generator
            keep.add(frozenset(c))

    # all candidate edges present in the sparsified graph
    n = W.shape[0]
    iu = np.triu_indices(n, k=1)
    all_edges = [(int(i), int(j)) for i, j in zip(*iu) if W[i, j] > 0.0]

    if min_budget_frac > 0.0 and all_edges:
        target = int(np.ceil(min_budget_frac * len(all_edges)))
        if len(keep) < target:
            extras = sorted((e for e in all_edges if frozenset(e) not in keep),
                            key=lambda e: W[e[0], e[1]], reverse=True)
            for e in extras:
                if len(keep) >= target:
                    break
                keep.add(frozenset(e))

    return keep


def critical_cycle_edges(A: np.ndarray, top_k: int | None = 8,
                         weight_floor: float = 0.0,
                         symmetrized: bool = False) -> set:
    """Attention matrix -> set of critical 1-cells (cycle generators).

    Returns only the critical edges (frozenset{i, j}) from discrete Morse
    matching -- the topologically essential cycle generators (H1). Uses the same
    symmetrize + sparsify + build_simplex_tree + _morse_match pipeline as
    morse_keep, but keeps ONLY the critical 1-cells (not the spanning forest).
    """
    W = A.astype(float) if symmetrized else symmetrize(A)
    W = sparsify(W, top_k=top_k, weight_floor=weight_floor)

    st = build_simplex_tree(W)
    simplices = [tuple(sorted(s)) for s, _ in st.get_simplices()]

    _, critical = _morse_match(simplices)

    return {frozenset(c) for c in critical if len(c) == 2}


if __name__ == "__main__":
    # quick smoke
    W = np.zeros((4, 4))
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        W[i, j] = W[j, i] = 1.0
    print("4-cycle keep:", morse_keep(W, symmetrized=True))
