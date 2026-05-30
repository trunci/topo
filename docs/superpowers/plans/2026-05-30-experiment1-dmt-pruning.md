# Experiment 1 — Discrete Morse Theory Attention Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test the falsifiable pre-registered claim that pruning attention by discrete-Morse critical structure (DMT) preserves model quality better than magnitude / random / sliding-window pruning **at the same per-head edge budget**. Build the DMT keep-set algorithm, three budget-matched baselines, a verified attention-masking mechanism, a two-pass evaluation driver over WikiText-2 perplexity (base model) and kinship accuracy (instruct model), and an integrity-clean analysis pipeline where every reported number is computed by a script into JSON and the findings document sources numbers only from that JSON.

**Architecture:** Two pure-ish topology modules feed a model-aware evaluation driver. `morse.py` (discrete Morse matching → keep-set) and `keepsets.py` (budget-matched baselines) both consume a symmetrized+sparsified weight matrix and return `set[frozenset({i,j})]`, identical type so they are interchangeable. `pruned_forward.py` converts per-(layer,head) keep-sets into additive attention biases and runs a masked forward via a verified monkeypatch of Qwen2's module-level `eager_attention_forward`. `eval_data.py` supplies WikiText windows and reuses `data_gen.build_pairs` for kinship. `run_exp1.py` orchestrates: per example, DMT first (which sets the per-head budget k), then baselines matched to k, then masked forward → `results/exp1.parquet`. `compute_stats_exp1.py` reads the parquet → `results/exp1_stats.json` (single source of truth). `analyze_exp1.py` makes plots + prints a verdict from the JSON. `FINDINGS_exp1.md` quotes only the JSON.

**Tech Stack:** Python 3.13, `uv` env, PyTorch (MPS/CPU), `transformers==5.9.0`, `gudhi`, `numpy`/`scipy`/`pandas`/`pyarrow`, `statsmodels`, `matplotlib`, `datasets`, `pytest`. All deps are already installed.

**Critical verified facts baked into this plan (do NOT redesign these):**
1. The reduction-based discrete Morse matching below is verified correct on triangle / 4-cycle / 2-disjoint-edges / path. The naive "free face with unique coface" greedy DEADLOCKS on cycles and is WRONG — do not use it.
2. Masking via forward hooks does NOT work, and `ALL_ATTENTION_FUNCTIONS["eager"]` raises `KeyError 'eager'` in transformers 5.9. The WORKING mechanism is monkeypatching `transformers.models.qwen2.modeling_qwen2.eager_attention_forward` with an additive bias (0.0 keep / -inf drop) added BEFORE softmax. Verified: all-keep mask reproduces unmasked loss to <1e-4 (lossless); aggressive mask changes loss.

---

## Environment notes (read before any task)

- Run everything via `uv run ...`. Tests: `uv run pytest ...`.
- A harmless warning `VIRTUAL_ENV ... does not match ... will be ignored` may appear. Ignore it, or prefix any command with `unset VIRTUAL_ENV;`.
- For any command that **loads a model** (model tests, the run driver), prefix with the env loader so HF downloads authenticate with the token in `.env`:
  ```
  set -a; [ -f .env ] && . ./.env; set +a;
  ```
- Tests import modules via the `src.` package (e.g. `from src.morse import morse_keep`) — match the existing repo convention (`tests/test_topology.py` etc.). Implementation modules in `src/` use **relative** imports (`from . import topology`), matching the spike's `run_spike.py`/`analyze.py` style.
- The `slow` pytest marker is already registered in `pyproject.toml`. Mark model-loading tests `@pytest.mark.slow`.
- Models (both 24 layers × 14 heads): base = `Qwen/Qwen2.5-0.5B` (WikiText perplexity), instruct = `Qwen/Qwen2.5-0.5B-Instruct` (kinship). Both load with `torch_dtype=torch.float32, attn_implementation="eager"`, `.to(device).eval()`, device = mps if available else cpu.
- **Existing `attn_extract.load_model(device=None)` hardcodes `MODEL_NAME` (the instruct model) and does NOT take a model name.** This plan adds a `load_named(model_name, device=None)` helper to `attn_extract.py` (Task 0) and reuses it for both models — that is the cleanest option and keeps the existing `load_model` working for the spike.
- **Existing `attn_extract.format_prompt(tok, text)`** takes `(tok, text)` and returns the chat-templated string. **Existing `data_gen.Item`** has fields `id, family, hop, pair_id, context, question, prompt, gold` — the gold answer is `item.gold` and `item.prompt` is the RAW (un-chat-templated) `context + " " + question`. Use these exact names.
- Work on a branch. Before Task 0 run: `cd /Users/trunci/Desktop/res && git checkout -b exp1-dmt-pruning` (skip if already on it).
- All commit messages end with the required co-author trailer shown in each commit step.

---

## File Structure

```
src/
  attn_extract.py       # MODIFIED: add load_named(model_name, device=None)
  morse.py              # NEW: discrete Morse matching + keep-set (+ min_budget_frac top-up)
  keepsets.py           # NEW: budget-matched baselines: magnitude / random / window
  pruned_forward.py     # NEW: monkeypatch masking, keepsets_to_bias, masked_loss
  eval_data.py          # NEW: WikiText-2 slice loader + kinship items (reuse build_pairs)
  run_exp1.py           # NEW: orchestration → results/exp1.parquet
  compute_stats_exp1.py # NEW: parquet → results/exp1_stats.json (single source of truth)
  analyze_exp1.py       # NEW: plots + verdict (reads stats JSON)
tests/
  test_attn_extract_loadnamed.py
  test_morse.py
  test_keepsets.py
  test_pruned_forward.py
  test_eval_data.py
  test_run_exp1.py
  test_analyze_exp1.py
results/
  exp1.parquet          # produced by run_exp1
  exp1_stats.json       # produced by compute_stats_exp1
  exp1_ppl_by_method.png, exp1_dmt_vs_magnitude.png, exp1_sparsity_hist.png
FINDINGS_exp1.md        # numbers sourced ONLY from exp1_stats.json
```

---

## Task 0: Add `load_named` to `attn_extract.py`

The base model (WikiText) needs a loader by name; the existing `load_model` hardcodes the instruct model. Add a parametrized helper and keep `load_model` as a thin wrapper so the spike is unaffected.

**Files:** `src/attn_extract.py`, `tests/test_attn_extract_loadnamed.py`

**Steps:**

- [ ] 1. Write the failing test `tests/test_attn_extract_loadnamed.py`:

```python
import pytest


@pytest.mark.slow
def test_load_named_loads_base_model():
    from src.attn_extract import load_named
    model, tok, device = load_named("Qwen/Qwen2.5-0.5B")
    assert model.config.num_attention_heads == 14
    assert len(model.model.layers) == 24
    assert device in ("mps", "cpu")
    # eager attention so output_attentions works
    assert model.config._attn_implementation == "eager"
```

- [ ] 2. Run it and confirm it FAILS (function does not exist yet):

```
set -a; [ -f .env ] && . ./.env; set +a; unset VIRTUAL_ENV; uv run pytest tests/test_attn_extract_loadnamed.py -q -m slow
```

Expected: `ImportError: cannot import name 'load_named' from 'src.attn_extract'` → **FAIL**.

- [ ] 3. Edit `src/attn_extract.py`. Replace the existing `load_model` function (lines reproduced below) with the `load_named` helper plus a thin `load_model` wrapper. Find this block:

```python
def load_model(device: str | None = None):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,          # float32 for stable eager attention on MPS
        attn_implementation="eager",
    )
    model.to(device).eval()
    return model, tok, device
```

and replace it with:

```python
def load_named(model_name: str, device: str | None = None):
    """Load any causal-LM by name with float32 + eager attention (so attentions work)."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,          # float32 for stable eager attention on MPS
        attn_implementation="eager",
    )
    model.to(device).eval()
    return model, tok, device


def load_model(device: str | None = None):
    """Backwards-compatible loader for the instruct model used by the spike."""
    return load_named(MODEL_NAME, device=device)
```

- [ ] 4. Run it and confirm it PASSES (downloads the base model on first run):

```
set -a; [ -f .env ] && . ./.env; set +a; unset VIRTUAL_ENV; uv run pytest tests/test_attn_extract_loadnamed.py -q -m slow
```

Expected: `1 passed` → **PASS**.

- [ ] 5. Confirm the spike's existing test still passes (no regression in `load_model`):

```
set -a; [ -f .env ] && . ./.env; set +a; unset VIRTUAL_ENV; uv run pytest tests/test_attn_extract.py -q -m slow
```

Expected: `2 passed` → **PASS**.

- [ ] 6. Commit:

```
cd /Users/trunci/Desktop/res && git add src/attn_extract.py tests/test_attn_extract_loadnamed.py
git commit -m "$(cat <<'EOF'
Add load_named to attn_extract for loading the base model by name

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: `morse.py` — discrete Morse matching + keep-set

The discrete Morse matching is the **verified** algorithm below. The naive "free face with unique coface" greedy deadlocks on cycles and is wrong; use this one exactly. Verified results: triangle → keeps 2 of 3 edges (drops the redundant one); 4-cycle → keeps all 4; 2 disjoint edges → keeps both; path → keeps all path edges.

**Files:** `src/morse.py`, `tests/test_morse.py`

**Steps:**

- [ ] 1. Write the failing test `tests/test_morse.py`:

```python
import numpy as np
from src.morse import morse_keep


def _wmat(n, edges):
    """Symmetric weight matrix with the given undirected edges at weight 1.0."""
    W = np.zeros((n, n), dtype=float)
    for i, j in edges:
        W[i, j] = 1.0
        W[j, i] = 1.0
    return W


def test_four_cycle_keeps_all_edges():
    # 0-1-2-3-0 cycle: one independent loop, no triangles -> keep all 4 edges.
    W = _wmat(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    keep = morse_keep(W, top_k=8)
    assert keep == {
        frozenset({0, 1}),
        frozenset({1, 2}),
        frozenset({2, 3}),
        frozenset({3, 0}),
    }
    assert len(keep) == 4


def test_filled_triangle_keeps_two_edges():
    # Triangle 0-1-2, all three edges -> flag complex fills the 2-simplex.
    # DMT drops exactly one redundant edge (edge<->triangle match), keeps 2.
    W = _wmat(3, [(0, 1), (1, 2), (0, 2)])
    keep = morse_keep(W, top_k=8)
    assert len(keep) == 2
    assert keep.issubset(
        {frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})}
    )


def test_two_disjoint_edges_keeps_both():
    # 0-1 and 2-3, no shared vertices -> two tree edges, both kept.
    W = _wmat(4, [(0, 1), (2, 3)])
    keep = morse_keep(W, top_k=8)
    assert keep == {frozenset({0, 1}), frozenset({2, 3})}


def test_path_keeps_all_three_edges():
    # 0-1-2-3 path -> spanning tree, all 3 edges kept.
    W = _wmat(4, [(0, 1), (1, 2), (2, 3)])
    keep = morse_keep(W, top_k=8)
    assert keep == {frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})}


def test_min_budget_frac_tops_up():
    # Sparse graph: two strong disjoint edges plus a third lower-weight candidate.
    # Natural keep-set = the 2 strong edges. There are 3 candidate edges total.
    # With min_budget_frac=1.0 the keep-set must be topped up to all 3.
    W = np.zeros((5, 5), dtype=float)
    for (i, j, w) in [(0, 1, 1.0), (2, 3, 1.0), (3, 4, 0.4)]:
        W[i, j] = w
        W[j, i] = w
    base = morse_keep(W, top_k=8, min_budget_frac=0.0)
    full = morse_keep(W, top_k=8, min_budget_frac=1.0)
    assert len(full) >= len(base)
    assert len(full) == 3  # all candidate edges retained at the floor
    assert frozenset({3, 4}) in full
```

- [ ] 2. Run it and confirm it FAILS:

```
unset VIRTUAL_ENV; uv run pytest tests/test_morse.py -q
```

Expected: `ModuleNotFoundError: No module named 'src.morse'` → **FAIL** (5 errors).

- [ ] 3. Implement `src/morse.py` (complete code):

```python
"""Discrete Morse theory keep-set selection for attention graphs.

The matching is a reduction-based discrete Morse matching that is correct on
cycles (the naive 'free face with unique coface' greedy DEADLOCKS on cycles and
is WRONG). Verified on triangle / 4-cycle / 2-disjoint-edges / path.

Keep-set = spanning-forest (tree) edges [vertex<->edge matches]
           UNION critical 1-cells [unmatched cycle generators].
Edges matched upward to a triangle (edge<->triangle) are DROPPED.
"""
from __future__ import annotations

import numpy as np

from . import topology


def _candidate_edges(W: np.ndarray) -> list[frozenset]:
    """Undirected edges (i<j) with positive weight in the symmetric matrix W."""
    n = W.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                edges.append(frozenset({i, j}))
    return edges


def morse_keep(
    A: np.ndarray,
    top_k: int = 8,
    weight_floor: float = 0.0,
    symmetrized: bool = False,
    min_budget_frac: float = 0.0,
) -> set[frozenset]:
    """Return the DMT keep-set of undirected edges for attention matrix A.

    Mirrors topology.h1_features preprocessing: symmetrize (unless already done)
    then sparsify (top_k per row, weight_floor) BEFORE building the flag complex.

    If min_budget_frac > 0 and the natural keep-set holds fewer than
    min_budget_frac * (#candidate edges) edges, top up with the highest-weight
    not-yet-kept candidate edges until the floor is reached. Default 0.0 (off).

    Returns: set[frozenset({i, j})].
    """
    A = np.asarray(A, dtype=float)
    W = A if symmetrized else topology.symmetrize(A)
    W = topology.sparsify(W, top_k=top_k, weight_floor=weight_floor)

    st = topology.build_simplex_tree(W)
    S = [tuple(sorted(s)) for s, _ in st.get_simplices()]
    Sset = set(S)
    dim = {s: len(s) - 1 for s in S}

    facets = {}
    for s in S:
        if len(s) > 1:
            facets[s] = [tuple(sorted(set(s) - {v})) for v in s]
        else:
            facets[s] = []

    cofaces = {s: [] for s in S}
    for s in S:
        for f in facets[s]:
            if f in Sset:  # only facets that are themselves cells in the complex
                cofaces[f].append(s)

    remaining = set(S)
    matched: list[tuple] = []
    critical: list[tuple] = []

    while remaining:
        reduced = False
        for a in remaining:
            rc = [c for c in cofaces[a] if c in remaining]
            if len(rc) == 1:  # a has exactly one remaining coface -> reduce/pair
                b = rc[0]
                matched.append((a, b))
                remaining.discard(a)
                remaining.discard(b)
                reduced = True
                break
        if not reduced:  # stuck: mark a highest-dimension remaining cell critical
            c = max(remaining, key=lambda s: dim[s])
            critical.append(c)
            remaining.discard(c)

    keep: set[frozenset] = set()
    for (a, b) in matched:
        if len(a) == 1 and len(b) == 2:  # vertex<->edge: spanning-forest (tree) edge -> KEEP
            keep.add(frozenset(b))
        # len(a)==2 and len(b)==3: edge<->triangle redundant edge -> DROPPED
    for c in critical:
        if len(c) == 2:  # critical 1-cell: cycle generator -> KEEP
            keep.add(frozenset(c))

    if min_budget_frac > 0:
        candidates = _candidate_edges(W)
        floor = int(np.ceil(min_budget_frac * len(candidates)))
        if len(keep) < floor:
            extra = [e for e in candidates if e not in keep]

            def _w(e):
                i, j = sorted(e)
                return W[i, j]

            extra.sort(key=_w, reverse=True)  # highest weight first
            for e in extra:
                if len(keep) >= floor:
                    break
                keep.add(e)

    return keep
```

- [ ] 4. Run it and confirm it PASSES:

```
unset VIRTUAL_ENV; uv run pytest tests/test_morse.py -q
```

Expected: `5 passed` → **PASS**.

- [ ] 5. Commit:

```
cd /Users/trunci/Desktop/res && git add src/morse.py tests/test_morse.py
git commit -m "$(cat <<'EOF'
Add DMT keep-set selection (morse.py) with verified reduction matching

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `keepsets.py` — budget-matched baselines

All three take the symmetrized+sparsified weight matrix `W` and an integer budget `k`, and return `set[frozenset({i,j})]` of exactly `min(k, n_candidate_edges)` edges (candidate = `i<j` with `W>0`). Same return type as `morse_keep`, so they are interchangeable.

**Files:** `src/keepsets.py`, `tests/test_keepsets.py`

**Steps:**

- [ ] 1. Write the failing test `tests/test_keepsets.py`:

```python
import numpy as np
from src.keepsets import magnitude_keep, random_keep, window_keep


def _W():
    # candidate edges (i<j, w>0):
    #   (0,1)=0.9  (0,3)=0.2  (1,2)=0.7  (2,3)=0.5  (0,4)=0.1
    n = 5
    W = np.zeros((n, n), dtype=float)
    for (i, j, w) in [(0, 1, 0.9), (0, 3, 0.2), (1, 2, 0.7), (2, 3, 0.5), (0, 4, 0.1)]:
        W[i, j] = w
        W[j, i] = w
    return W


def test_magnitude_returns_true_top_k():
    W = _W()
    keep = magnitude_keep(W, 3)
    assert keep == {frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})}


def test_magnitude_caps_at_candidate_count():
    W = _W()
    keep = magnitude_keep(W, 100)
    assert len(keep) == 5  # only 5 candidate edges exist


def test_window_returns_smallest_index_gap():
    W = _W()
    # |i-j|: (0,1)=1 (1,2)=1 (2,3)=1 (0,3)=3 (0,4)=4
    # smallest-gap 3 edges are the three |i-j|==1 edges.
    keep = window_keep(W, 3)
    assert keep == {frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})}


def test_window_tie_break_takes_next_smallest_gap():
    W = _W()
    # 4 edges: the three gap-1 edges, then the next-smallest-gap edge (0,3) gap=3.
    keep = window_keep(W, 4)
    assert keep == {
        frozenset({0, 1}),
        frozenset({1, 2}),
        frozenset({2, 3}),
        frozenset({0, 3}),
    }


def test_random_is_reproducible_and_seed_sensitive():
    W = _W()
    a = random_keep(W, 3, seed=0)
    b = random_keep(W, 3, seed=0)
    c = random_keep(W, 3, seed=1)
    assert a == b
    assert len(a) == 3
    assert a != c  # different seed gives a different selection here


def test_random_caps_at_candidate_count():
    W = _W()
    keep = random_keep(W, 100, seed=0)
    assert len(keep) == 5
```

- [ ] 2. Run it and confirm it FAILS:

```
unset VIRTUAL_ENV; uv run pytest tests/test_keepsets.py -q
```

Expected: `ModuleNotFoundError: No module named 'src.keepsets'` → **FAIL**.

- [ ] 3. Implement `src/keepsets.py` (complete code):

```python
"""Budget-matched baseline edge keep-sets for attention pruning.

Each function takes the symmetrized+sparsified weight matrix W and an integer
budget k, and returns a set of exactly min(k, n_candidate_edges) undirected
edges as frozenset({i, j}). Candidate edges are i<j with W[i, j] > 0.
"""
from __future__ import annotations

import random as _random

import numpy as np


def _candidates(W: np.ndarray) -> list[tuple[int, int]]:
    """List of (i, j) with i<j and W[i, j] > 0."""
    n = W.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                out.append((i, j))
    return out


def magnitude_keep(W: np.ndarray, k: int) -> set[frozenset]:
    """The k candidate edges with the highest weight (descending; ties by (i, j))."""
    W = np.asarray(W, dtype=float)
    cands = _candidates(W)
    cands.sort(key=lambda e: (-W[e[0], e[1]], e[0], e[1]))
    return {frozenset({i, j}) for (i, j) in cands[: min(k, len(cands))]}


def random_keep(W: np.ndarray, k: int, seed: int) -> set[frozenset]:
    """k candidate edges sampled uniformly without replacement (deterministic by seed)."""
    W = np.asarray(W, dtype=float)
    cands = _candidates(W)
    rng = _random.Random(seed)
    take = min(k, len(cands))
    chosen = rng.sample(cands, take)
    return {frozenset({i, j}) for (i, j) in chosen}


def window_keep(W: np.ndarray, k: int) -> set[frozenset]:
    """k candidate edges with the smallest |i-j| (sliding-window prior).

    Ties broken by higher weight, then by (i, j) ascending order.
    """
    W = np.asarray(W, dtype=float)
    cands = _candidates(W)
    cands.sort(key=lambda e: (abs(e[0] - e[1]), -W[e[0], e[1]], e[0], e[1]))
    return {frozenset({i, j}) for (i, j) in cands[: min(k, len(cands))]}
```

- [ ] 4. Run it and confirm it PASSES:

```
unset VIRTUAL_ENV; uv run pytest tests/test_keepsets.py -q
```

Expected: `6 passed` → **PASS**.

- [ ] 5. Commit:

```
cd /Users/trunci/Desktop/res && git add src/keepsets.py tests/test_keepsets.py
git commit -m "$(cat <<'EOF'
Add budget-matched baselines (magnitude/random/window) in keepsets.py

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `pruned_forward.py` — masking mechanism (monkeypatch + masked_loss)

**Verified facts baked in:** Do NOT use forward hooks and do NOT use `ALL_ATTENTION_FUNCTIONS["eager"]` (raises `KeyError 'eager'` in transformers 5.9). The WORKING mechanism monkeypatches the module-level function `transformers.models.qwen2.modeling_qwen2.eager_attention_forward`. An additive bias (0.0 keep, -inf drop) is added to the model's own causal mask (`am + bias`) **before** softmax; softmax then renormalizes over surviving keys, so no manual renormalization is needed (the spec describes renormalization conceptually — note that this additive-bias-before-softmax mechanism handles it automatically). The diagonal (q==k) is always kept so no query row is ever all -inf (which would NaN). Verified: all-keep mask reproduces unmasked loss to <1e-4 (lossless guarantee); an aggressive mask changes the loss.

**Files:** `src/pruned_forward.py`, `tests/test_pruned_forward.py`

**Steps:**

- [ ] 1. Write the failing test `tests/test_pruned_forward.py` (model tests are slow):

```python
import math

import pytest

from src.attn_extract import load_named
from src.pruned_forward import (
    PrunedForward,
    keepsets_to_bias,
    masked_loss,
)

PROMPT = "The capital of France is Paris. The capital of Italy is Rome."


def _all_keep_keepsets(n, H, L):
    """Per-(layer,head) keep-set retaining every undirected pair -> bias all zeros."""
    full = {frozenset({i, j}) for i in range(n) for j in range(i + 1, n)}
    return {(l, h): set(full) for l in range(L) for h in range(H)}


@pytest.mark.slow
def test_all_keep_is_lossless():
    model, tok, device = load_named("Qwen/Qwen2.5-0.5B")
    pf = PrunedForward(model)
    enc = tok(PROMPT, return_tensors="pt").to(device)
    n = enc["input_ids"].shape[1]
    L = len(model.model.layers)
    H = model.config.num_attention_heads

    base = masked_loss(model, tok, enc, biases=None)
    keepsets = _all_keep_keepsets(n, H, L)
    bias = keepsets_to_bias(keepsets, n, H, L, device)
    masked = masked_loss(model, tok, enc, biases=bias)

    assert abs(base - masked) < 1e-4, (base, masked)
    pf.uninstall()


@pytest.mark.slow
def test_aggressive_mask_changes_loss():
    model, tok, device = load_named("Qwen/Qwen2.5-0.5B")
    pf = PrunedForward(model)
    enc = tok(PROMPT, return_tensors="pt").to(device)
    n = enc["input_ids"].shape[1]
    L = len(model.model.layers)
    H = model.config.num_attention_heads

    base = masked_loss(model, tok, enc, biases=None)

    # diagonal-only on layer 0 (drop every off-diagonal edge in every head of
    # layer 0), all-keep elsewhere.
    full = {frozenset({i, j}) for i in range(n) for j in range(i + 1, n)}
    keepsets = {}
    for l in range(L):
        for h in range(H):
            keepsets[(l, h)] = set() if l == 0 else set(full)
    bias = keepsets_to_bias(keepsets, n, H, L, device)
    masked = masked_loss(model, tok, enc, biases=bias)

    assert math.isfinite(masked)
    assert abs(base - masked) > 1e-3, (base, masked)
    pf.uninstall()
```

- [ ] 2. Run it and confirm it FAILS:

```
set -a; [ -f .env ] && . ./.env; set +a; unset VIRTUAL_ENV; uv run pytest tests/test_pruned_forward.py -q
```

Expected: `ModuleNotFoundError: No module named 'src.pruned_forward'` → **FAIL**.

- [ ] 3. Implement `src/pruned_forward.py` (complete code):

```python
"""Attention masking via monkeypatching Qwen2's eager_attention_forward.

Verified mechanism (transformers 5.9): the module-level function
transformers.models.qwen2.modeling_qwen2.eager_attention_forward is replaced
with a wrapper that adds a per-layer additive bias [H, q, k] (0.0 keep / -inf
drop) to the attention scores BEFORE softmax. The bias is combined with the
model's own causal mask (am + bias), so causality is preserved and softmax
renormalizes over surviving keys automatically. The diagonal (q==k) is always
kept so no query row becomes all -inf (which would NaN).

Hooks and ALL_ATTENTION_FUNCTIONS["eager"] do NOT work here (the latter raises
KeyError 'eager' in transformers 5.9).
"""
from __future__ import annotations

import torch
import transformers.models.qwen2.modeling_qwen2 as mq

# Module-global patch state. STATE["bias"] maps layer_idx -> tensor [H, q, k].
STATE = {"on": False, "bias": None}

_ORIG = mq.eager_attention_forward
_INSTALLED = False


def _wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    am = attention_mask
    if STATE["on"] and STATE["bias"] is not None:
        b = STATE["bias"].get(getattr(module, "_li", None))
        if b is not None:
            # b is [H, q, k]; broadcast over batch -> [1, H, q, k]
            am = b.unsqueeze(0) if am is None else am + b.unsqueeze(0)
    return _ORIG(module, query, key, value, am, scaling, dropout=dropout, **kw)


def _install():
    global _INSTALLED
    if not _INSTALLED:
        mq.eager_attention_forward = _wrapped
        _INSTALLED = True


def _uninstall():
    global _INSTALLED
    if _INSTALLED:
        mq.eager_attention_forward = _ORIG
        _INSTALLED = False
    STATE["on"] = False
    STATE["bias"] = None


class PrunedForward:
    """Installs the monkeypatch and tags each attention module with its layer index.

    Usage:
        pf = PrunedForward(model)
        pf.set_bias(bias_dict); pf.enable()
        ... forward ...
        pf.disable()        # restore unmasked behavior (patch stays installed)
        pf.uninstall()      # fully restore the original function
    """

    def __init__(self, model):
        self.model = model
        for i, lyr in enumerate(model.model.layers):
            lyr.self_attn._li = i
        _install()

    def set_bias(self, bias):
        STATE["bias"] = bias

    def enable(self):
        STATE["on"] = True

    def disable(self):
        STATE["on"] = False

    def uninstall(self):
        _uninstall()


def keepsets_to_bias(keepsets_per_layer_head, n, H, L, device):
    """Convert per-(layer,head) keep-sets into a bias dict layer_idx -> Tensor[H, n, n].

    keepsets_per_layer_head: dict[(layer_idx, head_idx)] -> set[frozenset({q, k})].
    Entry (q, k) is 0.0 if frozenset({q, k}) is in that head's keep-set OR q == k
    (diagonal always kept), else -inf.
    """
    neg_inf = float("-inf")
    bias = {}
    for l in range(L):
        b = torch.full((H, n, n), neg_inf, dtype=torch.float32, device=device)
        for h in range(H):
            ks = keepsets_per_layer_head.get((l, h), set())
            for q in range(n):
                b[h, q, q] = 0.0  # diagonal always kept (prevents all-(-inf) rows)
            for e in ks:
                i, j = tuple(e)
                b[h, i, j] = 0.0
                b[h, j, i] = 0.0
        bias[l] = b
    return bias


def masked_loss(model, tok, enc, biases) -> float:
    """Mean next-token cross-entropy loss over the sequence under the given biases.

    enc: tokenizer output dict with input_ids (and attention_mask) already on device.
    biases: dict layer_idx -> Tensor[H, n, n], or None for unmasked.
    Returns the mean token loss (a Python float).

    Calls PrunedForward(model) itself (idempotent: re-tagging layer indices and
    re-installing the patch are both safe), so callers may use it standalone.
    """
    pf = PrunedForward(model)
    try:
        if biases is None:
            pf.disable()
        else:
            pf.set_bias(biases)
            pf.enable()
        input_ids = enc["input_ids"]
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=input_ids)
        return float(out.loss.item())
    finally:
        pf.disable()
```

> **Implementer note:** `model(..., labels=input_ids)` makes HF compute the standard shifted causal-LM mean token loss internally — exactly the metric we want, and it keeps the lossless guarantee exact.

- [ ] 4. Run it and confirm it PASSES (downloads the base model on first run; may take a minute):

```
set -a; [ -f .env ] && . ./.env; set +a; unset VIRTUAL_ENV; uv run pytest tests/test_pruned_forward.py -q -m slow
```

Expected: `2 passed` → **PASS**. (Ignore any `VIRTUAL_ENV ... will be ignored` line.)

- [ ] 5. Commit:

```
cd /Users/trunci/Desktop/res && git add src/pruned_forward.py tests/test_pruned_forward.py
git commit -m "$(cat <<'EOF'
Add attention masking via Qwen2 eager_attention_forward monkeypatch

Verified lossless on all-keep (<1e-4) and effective on aggressive masks.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `eval_data.py` — WikiText slice + kinship items

WikiText-2 windows for perplexity (base model); kinship items reuse `data_gen.build_pairs` (instruct model).

**Files:** `src/eval_data.py`, `tests/test_eval_data.py`

**Steps:**

- [ ] 1. Write the failing test `tests/test_eval_data.py`:

```python
import pytest

from src.eval_data import wikitext_windows, kinship_items


@pytest.mark.slow
def test_wikitext_windows_returns_nonempty_text():
    # downloads wikitext-2 on first run
    texts = wikitext_windows(n=5)
    assert isinstance(texts, list)
    assert len(texts) == 5
    assert all(isinstance(t, str) and len(t.strip()) > 0 for t in texts)


def test_kinship_items_returns_items():
    items = kinship_items(n_per_family=3)
    assert isinstance(items, list)
    assert len(items) > 0
    first = items[0]
    # data_gen.Item fields
    assert hasattr(first, "prompt")
    assert hasattr(first, "gold")
    assert hasattr(first, "hop")
```

- [ ] 2. Run it and confirm it FAILS:

```
unset VIRTUAL_ENV; uv run pytest tests/test_eval_data.py -q
```

Expected: `ModuleNotFoundError: No module named 'src.eval_data'` → **FAIL**.

- [ ] 3. Implement `src/eval_data.py` (complete code):

```python
"""Evaluation data: WikiText-2 windows (perplexity) and kinship items (accuracy)."""
from __future__ import annotations

from . import data_gen


def wikitext_windows(n: int = 50, min_chars: int = 80) -> list[str]:
    """Return up to n non-empty WikiText-2 test lines (each >= min_chars chars).

    Section headings (lines starting with '=') are skipped. The caller tokenizes
    and truncates to a max token length (e.g. 64) downstream.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    out = []
    for row in ds:
        t = row["text"].strip()
        if len(t) >= min_chars and not t.startswith("="):
            out.append(t)
            if len(out) >= n:
                break
    return out


def kinship_items(n_per_family: int = 15, seed: int = 0):
    """Reuse data_gen.build_pairs to produce kinship/ordering query Items."""
    return data_gen.build_pairs(n_per_family=n_per_family, seed=seed)
```

- [ ] 4. Run it and confirm it PASSES. Run the fast test first, then the slow one:

```
unset VIRTUAL_ENV; uv run pytest tests/test_eval_data.py::test_kinship_items_returns_items -q
set -a; [ -f .env ] && . ./.env; set +a; unset VIRTUAL_ENV; uv run pytest tests/test_eval_data.py -q -m slow
```

Expected: first `1 passed`; second `1 passed` → **PASS**.

- [ ] 5. Commit:

```
cd /Users/trunci/Desktop/res && git add src/eval_data.py tests/test_eval_data.py
git commit -m "$(cat <<'EOF'
Add eval_data: WikiText-2 windows and kinship items

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `run_exp1.py` — two-pass orchestration → `results/exp1.parquet`

**Decisions baked in (instructions left these as choices):**
- **Model loaders:** WikiText perplexity on the BASE model `attn_extract.load_named("Qwen/Qwen2.5-0.5B")`; kinship on the INSTRUCT model `attn_extract.load_named("Qwen/Qwen2.5-0.5B-Instruct")`.
- **Prompt text:** WikiText windows are plain text (NO chat template — base model). Kinship prompts go through `attn_extract.format_prompt(tok, item.prompt)` (chat template — instruct model, matching the spike's extraction path).
- **Kinship metric (simplest correct — avoids masked autoregressive generation):** teacher-forced gold-answer-token loss under the mask, PLUS a single-step greedy `correct` flag = (argmax of the logits at the final prompt position equals the gold answer's first token id). This gives both a loss and an accuracy signal in one masked forward. Documented here and in FINDINGS.

**Two-pass per example, per method ∈ {unpruned, dmt, magnitude, random, window}:**
1. Pass 1: `attn_extract.get_attentions(model, tok, device, text)` → numpy `[L, H, n, n]` (note: `get_attentions` internally re-applies `format_prompt` only for the kinship instruct path via the same `text` we pass; for WikiText we pass plain text and `format_prompt` would chat-wrap it — see the implementer note below for why we extract attentions directly instead of via `get_attentions` to keep the WikiText token sequence identical between Pass 1 and Pass 2).
2. **DMT first** (it sets the per-head budget): per head, `morse_keep(att[l,h], top_k=8, symmetrized=False)` → keepset; record `k_lh = len(keepset)`.
   - unpruned: keepset = all candidate edges (loss measured with `biases=None`).
   - baselines: per head use the **same** `k_lh` from DMT on that example+head, then `magnitude/random/window` on `W_lh = sparsify(symmetrize(att[l,h]))`.
3. Build biases via `keepsets_to_bias`, run Pass 2 masked forward, record loss. WikiText: `ppl = exp(loss)`.

**Files:** `src/run_exp1.py`, `tests/test_run_exp1.py`

**Steps:**

- [ ] 1. Write the failing smoke test `tests/test_run_exp1.py` (slow; tiny config via `run()` args):

```python
import pytest
import pandas as pd

from src.run_exp1 import run


@pytest.mark.slow
def test_run_writes_parquet_with_schema(tmp_path):
    out = tmp_path / "exp1.parquet"
    # tiny config: 1 wikitext window, 1 pair per family, short sequences
    run(out_path=str(out), n_wikitext=1, kinship_n_per_family=1, max_tokens=24)
    df = pd.read_parquet(out)
    expected_cols = {
        "example_id", "domain", "method", "loss", "ppl",
        "correct", "mean_sparsity", "seq_len",
    }
    assert expected_cols.issubset(set(df.columns))
    methods = {"unpruned", "dmt", "magnitude", "random", "window"}
    assert methods.issubset(set(df["method"].unique()))
    assert set(df["domain"].unique()).issubset({"wikitext", "kinship"})
    wt = df[df["domain"] == "wikitext"]
    assert (wt["ppl"] > 0).all()
    dmt = df[df["method"] == "dmt"]
    assert ((dmt["mean_sparsity"] >= 0) & (dmt["mean_sparsity"] <= 1)).all()
```

- [ ] 2. Run it and confirm it FAILS:

```
set -a; [ -f .env ] && . ./.env; set +a; unset VIRTUAL_ENV; uv run pytest tests/test_run_exp1.py -q
```

Expected: `ModuleNotFoundError: No module named 'src.run_exp1'` → **FAIL**.

- [ ] 3. Implement `src/run_exp1.py` (complete code):

```python
"""Experiment 1 orchestration: DMT vs budget-matched baselines on two domains.

For each example and method in {unpruned, dmt, magnitude, random, window}:
  Pass 1: extract per-(layer,head) attention.
  DMT first sets the per-head budget k_lh; baselines match it exactly.
  Pass 2: masked forward -> loss (and ppl for wikitext).

WikiText perplexity on the BASE model; kinship accuracy on the INSTRUCT model.
Writes results/exp1.parquet.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import torch

from . import attn_extract, eval_data, keepsets as _keepsets_mod, morse, topology
from .pruned_forward import PrunedForward, keepsets_to_bias, masked_loss

# ----- run configuration (top-of-file constants; override via run() args) -----
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_WIKITEXT = 50
KINSHIP_N_PER_FAMILY = 15
MAX_TOKENS = 64
TOP_K = 8
RANDOM_SEED = 0
METHODS = ["unpruned", "dmt", "magnitude", "random", "window"]


def _attentions_for_ids(model, input_ids):
    """Run a forward with output_attentions and return numpy [L, H, n, n].

    We extract attentions directly from the SAME input_ids used for the masked
    forward, so Pass 1 and Pass 2 see an identical token sequence. (We do NOT use
    attn_extract.get_attentions here because it re-applies the chat template to a
    raw string, which would not match a pre-tokenized WikiText window.)
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True)
    atts = [a[0].float().cpu().numpy() for a in out.attentions]  # each [H, n, n]
    return np.stack(atts)


def _all_candidate_keepset(W):
    """Keep-set of every candidate edge (i<j, W>0) for the unpruned all-keep bias."""
    n = W.shape[0]
    return {frozenset({i, j}) for i in range(n) for j in range(i + 1, n) if W[i, j] > 0}


def _per_head_W(att, l, h, top_k):
    """Symmetrized+sparsified weight matrix for one (layer, head)."""
    return topology.sparsify(topology.symmetrize(att[l, h]), top_k=top_k)


def _build_keepsets(att, method, top_k, seed, dmt_budgets=None):
    """Return (dict[(l,h)] -> keepset, list of (n_kept, n_candidates) per head).

    For baselines, dmt_budgets[(l,h)] supplies the matched budget k_lh.
    """
    L, H = att.shape[0], att.shape[1]
    out = {}
    spar = []
    for l in range(L):
        for h in range(H):
            W = _per_head_W(att, l, h, top_k)
            n_cand = int((W > 0).sum() // 2)
            if method == "unpruned":
                ks = _all_candidate_keepset(W)
            elif method == "dmt":
                ks = morse.morse_keep(W, top_k=top_k, symmetrized=True)
            elif method == "magnitude":
                ks = _keepsets_mod.magnitude_keep(W, dmt_budgets[(l, h)])
            elif method == "random":
                ks = _keepsets_mod.random_keep(W, dmt_budgets[(l, h)], seed)
            elif method == "window":
                ks = _keepsets_mod.window_keep(W, dmt_budgets[(l, h)])
            else:
                raise ValueError(method)
            out[(l, h)] = ks
            spar.append((len(ks), n_cand))
    return out, spar


def _mean_sparsity(spar):
    """Mean over heads of (n_kept / n_candidates); heads with no candidates -> 1.0."""
    fracs = [(k / c) if c > 0 else 1.0 for (k, c) in spar]
    return float(np.mean(fracs)) if fracs else 1.0


def _eval_example(model, enc, att, domain, example_id, gold_id, top_k, seed):
    """Run all methods on a single example; return a list of row dicts."""
    n = enc["input_ids"].shape[1]
    L, H = att.shape[0], att.shape[1]
    rows = []

    # DMT first -> per-head budgets
    dmt_ks, dmt_spar = _build_keepsets(att, "dmt", top_k, seed)
    budgets = {lh: len(ks) for lh, ks in dmt_ks.items()}

    pf = PrunedForward(model)
    for method in METHODS:
        if method == "unpruned":
            _, spar = _build_keepsets(att, "unpruned", top_k, seed)
            pf.disable()
            biases = None
        else:
            if method == "dmt":
                ks, spar = dmt_ks, dmt_spar
            else:
                ks, spar = _build_keepsets(att, method, top_k, seed, dmt_budgets=budgets)
            biases = keepsets_to_bias(ks, n, H, L, model.device if hasattr(model, "device") else enc["input_ids"].device)

        if domain == "wikitext":
            loss = masked_loss(model, None, enc, biases=biases)
            rows.append({
                "example_id": example_id, "domain": "wikitext", "method": method,
                "loss": loss, "ppl": math.exp(loss),
                "correct": float("nan"), "mean_sparsity": _mean_sparsity(spar),
                "seq_len": n,
            })
        else:  # kinship: teacher-forced gold-token loss + single-step greedy accuracy
            if biases is None:
                pf.disable()
            else:
                pf.set_bias(biases)
                pf.enable()
            with torch.no_grad():
                logits = model(input_ids=enc["input_ids"]).logits
            pf.disable()
            last = logits[0, -1]  # next-token distribution after the prompt
            logprobs = torch.log_softmax(last, dim=-1)
            gold_loss = float(-logprobs[gold_id].item())
            pred = int(torch.argmax(last).item())
            rows.append({
                "example_id": example_id, "domain": "kinship", "method": method,
                "loss": gold_loss, "ppl": float("nan"),
                "correct": float(pred == gold_id),
                "mean_sparsity": _mean_sparsity(spar), "seq_len": n,
            })
    return rows


def _gold_token_id(tok, answer):
    """First token id of the space-prefixed answer string."""
    ids = tok(" " + answer.strip(), add_special_tokens=False)["input_ids"]
    return ids[0] if ids else tok.eos_token_id


def run(
    out_path: str = "results/exp1.parquet",
    n_wikitext: int = N_WIKITEXT,
    kinship_n_per_family: int = KINSHIP_N_PER_FAMILY,
    max_tokens: int = MAX_TOKENS,
    top_k: int = TOP_K,
    seed: int = RANDOM_SEED,
) -> str:
    rows = []

    # --- WikiText perplexity on the BASE model (plain text, no chat template) ---
    base_model, base_tok, device = attn_extract.load_named(BASE_MODEL)
    texts = eval_data.wikitext_windows(n=n_wikitext)
    for ei, text in enumerate(texts):
        enc = base_tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
        if enc["input_ids"].shape[1] < 3:
            continue
        att = _attentions_for_ids(base_model, enc["input_ids"])
        rows += _eval_example(base_model, enc, att, "wikitext", ei, None, top_k, seed)
    del base_model

    # --- Kinship on the INSTRUCT model (chat-templated prompts) ---
    inst_model, inst_tok, device = attn_extract.load_named(INSTRUCT_MODEL)
    items = eval_data.kinship_items(n_per_family=kinship_n_per_family, seed=seed)
    for ei, item in enumerate(items):
        prompt = attn_extract.format_prompt(inst_tok, item.prompt)
        enc = inst_tok(prompt, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
        if enc["input_ids"].shape[1] < 3:
            continue
        gold_id = _gold_token_id(inst_tok, item.gold)
        att = _attentions_for_ids(inst_model, enc["input_ids"])
        rows += _eval_example(inst_model, enc, att, "kinship", ei, gold_id, top_k, seed)
    del inst_model

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_parquet(out_path)
    return out_path


if __name__ == "__main__":
    p = run()
    print(f"wrote {p}")
```

> **Implementer notes:**
> - The `keepsets` module is imported as `_keepsets_mod` so it is never shadowed by a local variable named `keepsets`.
> - `masked_loss(model, None, enc, biases=biases)` passes `tok=None` because `masked_loss` does not use the tokenizer (it only needs `enc["input_ids"]`). This matches the Task 3 signature `masked_loss(model, tok, enc, biases)`.
> - `model.device` may not exist on all model objects; the code falls back to the input tensor's device.

- [ ] 4. Run the smoke test and confirm it PASSES (loads both models; allow a few minutes):

```
set -a; [ -f .env ] && . ./.env; set +a; unset VIRTUAL_ENV; uv run pytest tests/test_run_exp1.py -q -m slow
```

Expected: `1 passed` → **PASS**.

- [ ] 5. Commit:

```
cd /Users/trunci/Desktop/res && git add src/run_exp1.py tests/test_run_exp1.py
git commit -m "$(cat <<'EOF'
Add run_exp1 two-pass orchestration (DMT sets budget; baselines match)

Kinship metric = teacher-forced gold-token loss + single-step greedy accuracy.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `compute_stats_exp1.py` + `analyze_exp1.py` (stats JSON, plots, verdict)

`compute_stats_exp1.py` is the **single source of truth** for every reported number (integrity requirement, mirroring the existing `src/compute_stats.py` pattern from the spike). `analyze_exp1.py` makes plots and prints the verdict, reading the same JSON.

**Files:** `src/compute_stats_exp1.py`, `src/analyze_exp1.py`, `tests/test_analyze_exp1.py`

**Steps:**

- [ ] 1. Write the failing test `tests/test_analyze_exp1.py` (fast — synthetic parquet, no model):

```python
import json

import numpy as np
import pandas as pd

from src.compute_stats_exp1 import compute_stats
from src.analyze_exp1 import analyze


def _fake_parquet(path):
    rng = np.random.default_rng(0)
    rows = []
    methods = ["unpruned", "dmt", "magnitude", "random", "window"]
    # wikitext: make dmt clearly best (lowest loss) so the verdict is WINS
    base = {"unpruned": 2.0, "dmt": 2.1, "magnitude": 2.4, "random": 2.8, "window": 2.6}
    for ei in range(8):
        for m in methods:
            loss = base[m] + rng.normal(0, 0.02)
            rows.append({
                "example_id": ei, "domain": "wikitext", "method": m,
                "loss": loss, "ppl": float(np.exp(loss)),
                "correct": float("nan"), "mean_sparsity": 0.4, "seq_len": 40,
            })
    for ei in range(6):
        for m in methods:
            rows.append({
                "example_id": ei, "domain": "kinship", "method": m,
                "loss": 1.0, "ppl": float("nan"),
                "correct": 1.0 if m in ("unpruned", "dmt") else 0.0,
                "mean_sparsity": 0.4, "seq_len": 30,
            })
    pd.DataFrame(rows).to_parquet(path)


def test_compute_stats_writes_json(tmp_path):
    pq = tmp_path / "exp1.parquet"
    out = tmp_path / "exp1_stats.json"
    _fake_parquet(pq)
    stats = compute_stats(str(pq), str(out))
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == stats
    assert "wikitext_mean_ppl_by_method" in stats
    assert set(stats["wikitext_mean_ppl_by_method"]).issuperset(
        {"unpruned", "dmt", "magnitude", "random", "window"}
    )
    assert "wilcoxon_dmt_vs" in stats
    assert "kinship_accuracy_by_method" in stats
    assert "verdict" in stats
    assert stats["verdict"] == "WINS"  # dmt beats all three baselines here


def test_analyze_runs_and_returns_verdict(tmp_path):
    pq = tmp_path / "exp1.parquet"
    sj = tmp_path / "exp1_stats.json"
    _fake_parquet(pq)
    compute_stats(str(pq), str(sj))
    verdict = analyze(str(pq), str(sj), outdir=str(tmp_path))
    assert verdict in ("WINS", "NULL/LOSS")
    assert (tmp_path / "exp1_ppl_by_method.png").exists()
    assert (tmp_path / "exp1_dmt_vs_magnitude.png").exists()
    assert (tmp_path / "exp1_sparsity_hist.png").exists()
```

- [ ] 2. Run it and confirm it FAILS:

```
unset VIRTUAL_ENV; uv run pytest tests/test_analyze_exp1.py -q
```

Expected: `ModuleNotFoundError: No module named 'src.compute_stats_exp1'` → **FAIL**.

- [ ] 3. Implement `src/compute_stats_exp1.py` (complete code):

```python
"""Compute every reported Experiment-1 number from the parquet into a JSON.

SINGLE SOURCE OF TRUTH for reported numbers (integrity rule): FINDINGS_exp1.md
must quote only values produced here, never hand-typed. Mirrors src/compute_stats.py.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASELINES = ["magnitude", "random", "window"]


def _paired_losses(df, method_a, method_b):
    """Aligned per-example loss arrays for two methods (inner join on example_id)."""
    a = df[df["method"] == method_a].set_index("example_id")["loss"]
    b = df[df["method"] == method_b].set_index("example_id")["loss"]
    idx = a.index.intersection(b.index)
    return a.loc[idx].to_numpy(), b.loc[idx].to_numpy()


def compute_stats(parquet_path: str, out_path: str) -> dict:
    df = pd.read_parquet(parquet_path)
    wt = df[df["domain"] == "wikitext"]
    kin = df[df["domain"] == "kinship"]

    stats = {}

    stats["wikitext_mean_ppl_by_method"] = {
        m: float(wt[wt["method"] == m]["ppl"].mean()) for m in wt["method"].unique()
    }
    stats["wikitext_mean_loss_by_method"] = {
        m: float(wt[wt["method"] == m]["loss"].mean()) for m in wt["method"].unique()
    }

    # Paired Wilcoxon of per-example loss: dmt vs each baseline (WikiText)
    wilcox = {}
    for b in BASELINES:
        da, db = _paired_losses(wt, "dmt", b)
        if len(da) >= 1 and np.any(da - db != 0):
            try:
                stat, p = wilcoxon(da, db)
                stat, p = float(stat), float(p)
            except ValueError:
                stat, p = float("nan"), float("nan")
        else:
            stat, p = float("nan"), float("nan")
        wilcox[b] = {
            "statistic": stat,
            "pvalue": p,
            "dmt_mean_loss": float(np.mean(da)) if len(da) else float("nan"),
            "baseline_mean_loss": float(np.mean(db)) if len(db) else float("nan"),
            "dmt_lower": bool(np.mean(da) < np.mean(db)) if len(da) else False,
        }
    stats["wilcoxon_dmt_vs"] = wilcox

    stats["kinship_accuracy_by_method"] = {
        m: float(kin[kin["method"] == m]["correct"].mean()) for m in kin["method"].unique()
    }
    stats["kinship_mean_gold_loss_by_method"] = {
        m: float(kin[kin["method"] == m]["loss"].mean()) for m in kin["method"].unique()
    }

    dmt_wt = wt[wt["method"] == "dmt"]["mean_sparsity"]
    stats["dmt_mean_sparsity"] = float(dmt_wt.mean()) if len(dmt_wt) else float("nan")

    # Verdict: DMT WINS iff its WikiText mean PPL < magnitude AND < random AND < window
    ppl = stats["wikitext_mean_ppl_by_method"]
    dmt_ppl = ppl.get("dmt", float("inf"))
    wins = all(dmt_ppl < ppl.get(b, float("inf")) for b in BASELINES)
    stats["verdict"] = "WINS" if wins else "NULL/LOSS"

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    compute_stats("results/exp1.parquet", "results/exp1_stats.json")
    print("WROTE results/exp1_stats.json")
```

- [ ] 4. Implement `src/analyze_exp1.py` (complete code):

```python
"""Experiment-1 plots and verdict. Reads the stats JSON; never recomputes
reported numbers independently of compute_stats_exp1.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def analyze(parquet_path: str, stats_path: str, outdir: str = "results") -> str:
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_parquet(parquet_path)
    with open(stats_path) as f:
        stats = json.load(f)

    # 1. bar chart of mean PPL by method (WikiText)
    ppl = stats["wikitext_mean_ppl_by_method"]
    methods = [m for m in ["unpruned", "dmt", "magnitude", "random", "window"] if m in ppl]
    fig, ax = plt.subplots()
    ax.bar(methods, [ppl[m] for m in methods])
    ax.set_ylabel("mean perplexity")
    ax.set_title("WikiText-2 mean PPL by method")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "exp1_ppl_by_method.png"))
    plt.close(fig)

    # 2. scatter of per-example DMT vs magnitude loss (WikiText)
    wt = df[df["domain"] == "wikitext"]
    a = wt[wt["method"] == "dmt"].set_index("example_id")["loss"]
    b = wt[wt["method"] == "magnitude"].set_index("example_id")["loss"]
    idx = a.index.intersection(b.index)
    fig, ax = plt.subplots()
    ax.scatter(b.loc[idx], a.loc[idx])
    if len(idx):
        lim = [min(a.loc[idx].min(), b.loc[idx].min()), max(a.loc[idx].max(), b.loc[idx].max())]
        ax.plot(lim, lim, "--", color="gray")
    ax.set_xlabel("magnitude loss")
    ax.set_ylabel("dmt loss")
    ax.set_title("Per-example loss: DMT vs magnitude (below line = DMT better)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "exp1_dmt_vs_magnitude.png"))
    plt.close(fig)

    # 3. histogram of DMT mean_sparsity (WikiText)
    fig, ax = plt.subplots()
    ax.hist(wt[wt["method"] == "dmt"]["mean_sparsity"], bins=20)
    ax.set_xlabel("DMT mean sparsity (fraction of candidate edges kept)")
    ax.set_ylabel("count")
    ax.set_title("DMT sparsity distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "exp1_sparsity_hist.png"))
    plt.close(fig)

    verdict = stats["verdict"]
    print("=== Experiment 1 verdict ===")
    print(f"WikiText mean PPL: {stats['wikitext_mean_ppl_by_method']}")
    print(f"Kinship accuracy: {stats['kinship_accuracy_by_method']}")
    print(f"Wilcoxon DMT vs baselines: {stats['wilcoxon_dmt_vs']}")
    print(f"VERDICT: {verdict}")
    return verdict


if __name__ == "__main__":
    analyze("results/exp1.parquet", "results/exp1_stats.json")
```

- [ ] 5. Run the test and confirm it PASSES:

```
unset VIRTUAL_ENV; uv run pytest tests/test_analyze_exp1.py -q
```

Expected: `2 passed` → **PASS**.

- [ ] 6. Commit:

```
cd /Users/trunci/Desktop/res && git add src/compute_stats_exp1.py src/analyze_exp1.py tests/test_analyze_exp1.py
git commit -m "$(cat <<'EOF'
Add compute_stats_exp1 (source-of-truth JSON) and analyze_exp1 (plots/verdict)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Full run + `FINDINGS_exp1.md` (numbers sourced ONLY from JSON)

**INTEGRITY REQUIREMENT (do not violate — carried over from a prior failure on this project):** Every number in `FINDINGS_exp1.md` must be copied from `results/exp1_stats.json`. Run the pipeline, READ the actual JSON output, then write the findings using those exact values. Never hand-type or estimate a number.

**Files:** `FINDINGS_exp1.md` (plus generated `results/exp1.parquet`, `results/exp1_stats.json`, three PNGs)

**Steps:**

- [ ] 1. Run the full experiment with the default (real) config. Loads both models; may take several minutes:

```
set -a; [ -f .env ] && . ./.env; set +a; unset VIRTUAL_ENV; uv run python -m src.run_exp1
```

Expected final line: `wrote results/exp1.parquet`.

- [ ] 2. Compute the stats JSON, then the plots + verdict:

```
unset VIRTUAL_ENV; uv run python -m src.compute_stats_exp1
unset VIRTUAL_ENV; uv run python -m src.analyze_exp1
```

Expected: `WROTE results/exp1_stats.json`, then a printed verdict block ending with `VERDICT: WINS` or `VERDICT: NULL/LOSS`.

- [ ] 3. READ the actual JSON before writing anything:

```
unset VIRTUAL_ENV; uv run python -c "import json; print(json.dumps(json.load(open('results/exp1_stats.json')), indent=2))"
```

Confirm the file lists `wikitext_mean_ppl_by_method`, `wilcoxon_dmt_vs`, `kinship_accuracy_by_method`, `dmt_mean_sparsity`, and `verdict`.

- [ ] 4. Write `FINDINGS_exp1.md` using **only** the values printed in step 3. Use this template and substitute the real numbers from the JSON (invent nothing):

```markdown
# Experiment 1 Findings: DMT Attention Pruning

**Date:** 2026-05-30
**Verdict:** <copy stats["verdict"]>

All numbers below are copied verbatim from `results/exp1_stats.json`
(produced by `src/compute_stats_exp1.py`). No number here is hand-computed.

## Setup
- Base model: Qwen/Qwen2.5-0.5B (WikiText-2 perplexity).
- Instruct model: Qwen/Qwen2.5-0.5B-Instruct (kinship).
- Methods: unpruned, dmt, magnitude, random, window.
- DMT sets each head's edge budget; baselines match it exactly per head.
- Masking: additive bias (0 keep / -inf drop) before softmax via Qwen2
  eager_attention_forward monkeypatch; verified lossless on all-keep.
- Kinship metric: teacher-forced gold-token loss + single-step greedy accuracy.

## WikiText-2 mean perplexity by method
<one line per method from stats["wikitext_mean_ppl_by_method"]>

## Paired Wilcoxon (per-example loss): DMT vs each baseline
<for each baseline: statistic, pvalue, dmt_mean_loss, baseline_mean_loss, dmt_lower
 from stats["wilcoxon_dmt_vs"]>

## Kinship accuracy by method
<one line per method from stats["kinship_accuracy_by_method"]>

## DMT sparsity
- Mean DMT sparsity (fraction of candidate edges kept): <stats["dmt_mean_sparsity"]>

## Plots
- results/exp1_ppl_by_method.png
- results/exp1_dmt_vs_magnitude.png
- results/exp1_sparsity_hist.png

## Conclusion
<Plain statement matching the pre-registered claim. If verdict == WINS: DMT's
WikiText mean PPL is lower than magnitude AND it beats random and window at equal
per-head budget. If NULL/LOSS: state plainly that DMT did not beat the baselines
— a real, publishable negative (topology diagnostic but not prescriptive). No spin.>
```

- [ ] 5. Commit the findings and artifacts:

```
cd /Users/trunci/Desktop/res && git add FINDINGS_exp1.md results/exp1.parquet results/exp1_stats.json results/exp1_ppl_by_method.png results/exp1_dmt_vs_magnitude.png results/exp1_sparsity_hist.png
git commit -m "$(cat <<'EOF'
Add Experiment 1 findings and artifacts (numbers sourced from stats JSON)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

> **Note:** `.gitignore` ignores `results/*.parquet` and `results/*.png`. Force-add them with `git add -f` if you want them tracked, or commit only `FINDINGS_exp1.md` + `results/exp1_stats.json` and leave the heavy artifacts gitignored (matching the spike's convention, which tracked only the findings note). Either is acceptable; prefer the spike's convention (findings + JSON tracked, parquet/PNG gitignored) — in that case the add line is just `git add FINDINGS_exp1.md && git add -f results/exp1_stats.json`.

- [ ] 6. Final verification — run the full fast test suite (skip slow) to confirm nothing regressed:

```
unset VIRTUAL_ENV; uv run pytest -q -m "not slow"
```

Expected: all fast tests pass (morse, keepsets, kinship-items, analyze, plus the pre-existing spike fast tests for topology and data_gen).

---

## Self-Review notes (spec requirement → task mapping)

- **DMT keep-set = critical 1-cells (essential cycles) ∪ Morse spanning forest (tree edges); DROP edges paired upward into a triangle** (spec "Method" steps 2-3) → Task 1 `morse_keep`, verified by `test_filled_triangle_keeps_two_edges` (drops the redundant edge), `test_four_cycle_keeps_all_edges` (critical cycle generators kept), `test_path`/`test_two_disjoint_edges` (tree edges kept). The reduction-based matching (not the naive deadlocking greedy) is used exactly as the verified algorithm specifies.
- **After symmetrize + sparsify, reuse topology.py; build flag complex via build_simplex_tree** (spec "Method" intro, step 1) → Task 1 `morse_keep` symmetrizes (unless `symmetrized=True`) and sparsifies via `topology.symmetrize`/`topology.sparsify`, then `topology.build_simplex_tree`, mirroring `h1_features`.
- **`morse_keep(A, top_k, weight_floor)` interface returning `set[frozenset({i,j})]`** (spec "Output interface") → Task 1 signature `morse_keep(A, top_k=8, weight_floor=0.0, symmetrized=False, min_budget_frac=0.0)` returns `set[frozenset]`. Extra `symmetrized`/`min_budget_frac` params are additive (defaults preserve the spec interface).
- **`min_budget_frac` floor knob, default small/off, top up to floor with next-highest-weight edges** (spec "Risk and mitigation") → Task 1 `min_budget_frac` (default 0.0 = off), verified by `test_min_budget_frac_tops_up`.
- **Budget-matched baselines: magnitude top-k, random seeded, window nearest-diagonal; all return the same `set[frozenset]` type** (spec "Fair comparison") → Task 2 `magnitude_keep(W,k)`/`random_keep(W,k,seed)`/`window_keep(W,k)`, each capped at candidate count; tests cover top-k correctness, window gap + tie-break, random reproducibility/seed-sensitivity, and capping.
- **Two-pass mask-and-rerun; unpruned control through the identical path; diagonal always kept; harness lossless** (spec "Pruning mechanism", "Testing strategy") → Task 3 monkeypatch (`am + bias` before softmax) + `keepsets_to_bias` (diagonal forced 0.0); unpruned uses `biases=None` (and all-keep verified equivalent). Lossless (<1e-4) and effectiveness (>1e-3) verified by the two slow tests. Note added that softmax handles renormalization, replacing the spec's manual post-softmax renormalize+zero with the verified additive-bias-before-softmax mechanism.
- **WikiText-2 perplexity on the base model; kinship accuracy on the instruct model; both 24L×14H, float32, eager** (spec "Reuse and scope guardrails") → Task 0 `load_named`; Task 5 loads base for WikiText (`ppl=exp(loss)`) and instruct for kinship; both eager/float32 via `load_named`.
- **Reuse attn_extract, topology, data_gen unchanged; small WikiText slice + kinship only; MMLU/MuSiQue/etc deferred** (spec "Reuse and scope guardrails") → Task 4 reuses `data_gen.build_pairs`; Task 5 reuses `attn_extract`/`topology`. Only `attn_extract` is touched, and only additively (`load_named`), leaving the spike untouched. No extra datasets.
- **DMT decides k per (layer,head,example); baselines keep exactly k; random seed deterministic** (spec "Fair comparison") → Task 5 two-pass driver computes DMT first, passes `budgets` into baseline construction; `random_keep` is seeded.
- **Deliverable parquet: per example × method with loss, ppl, accuracy, mean sparsity** (spec "Deliverables") → Task 5 schema `example_id, domain, method, loss, ppl (nan for kinship), correct (nan for wikitext), mean_sparsity, seq_len`, asserted by `test_run_exp1`.
- **Compute target: minutes; small config** (spec "Reuse and scope guardrails") → Task 5 top-of-file constants (`N_WIKITEXT=50`, `KINSHIP_N_PER_FAMILY=15`, `MAX_TOKENS=64`), overridable via `run()`; smoke test uses a tiny config.
- **Analysis: per-method PPL & accuracy, paired DMT-vs-baseline Wilcoxon with effect direction, plots (PPL bar, DMT-vs-magnitude scatter, sparsity histogram), clear win/null verdict** (spec "Pre-registered claim", "Deliverables") → Task 6 `compute_stats_exp1` (means, Wilcoxon, `dmt_lower` direction, verdict) + `analyze_exp1` (three plots, printed verdict).
- **Pre-registered win condition: DMT WikiText PPL < magnitude AND beats random AND window; honest null reported straight** (spec "Pre-registered claim and null") → Task 6 verdict logic `WINS` iff DMT PPL < all three baselines, else `NULL/LOSS`; verified by `test_compute_stats_writes_json` (WINS case). Task 7 FINDINGS template forbids spin on the null.
- **Integrity: all reported numbers computed by a script from the parquet, none hand-typed (lesson from the spike's integrity failure)** (spec "Deliverables") → Task 6 `compute_stats_exp1.py` is the single source of truth (mirrors existing `src/compute_stats.py`); Task 7 explicitly reads the JSON before writing FINDINGS and forbids hand-typed numbers; `test_compute_stats_writes_json` asserts the on-disk JSON equals the returned dict.

**Signature consistency check (cross-task):**
- `load_named(model_name, device=None)` — Task 0; called Task 3/4/5 tests and Task 5 driver. ✓
- `morse_keep(A, top_k=8, weight_floor=0.0, symmetrized=False, min_budget_frac=0.0)` — Task 1; called Task 5 with `top_k=top_k, symmetrized=True` (on the already-symmetrized+sparsified `W`). ✓
- `magnitude_keep(W, k)`, `random_keep(W, k, seed)`, `window_keep(W, k)` — Task 2; called Task 5 via `_keepsets_mod` (no shadow). ✓
- `keepsets_to_bias(keepsets_per_layer_head, n, H, L, device)` — Task 3; called Task 5 and Task 3 test. ✓
- `masked_loss(model, tok, enc, biases)` — Task 3; called Task 5 (`tok=None`) and Task 3 test. ✓
- `PrunedForward(model)` with `set_bias/enable/disable/uninstall` — Task 3; used Task 3 test and Task 5 kinship path. ✓
- `wikitext_windows(n=50, min_chars=80)`, `kinship_items(n_per_family=15, seed=0)` — Task 4; called Task 5. ✓
- `run(out_path, n_wikitext, kinship_n_per_family, max_tokens, top_k, seed)` — Task 5; called Task 5 test + Task 7. ✓
- `compute_stats(parquet_path, out_path)` / `analyze(parquet_path, stats_path, outdir)` — Task 6; called Task 6 test + Task 7. ✓
- `Item` fields used: `item.prompt` (raw, chat-templated downstream via `format_prompt(tok, item.prompt)`), `item.gold` (gold answer string), `item.hop` (test assertion). ✓ matches `data_gen.Item`.
- `format_prompt(tok, text)` — existing 2-arg signature; called Task 5 as `format_prompt(inst_tok, item.prompt)`. ✓
```
