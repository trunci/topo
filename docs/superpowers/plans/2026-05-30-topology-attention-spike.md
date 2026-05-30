# Topology-of-Attention De-Risk Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal pipeline that extracts attention from Qwen2.5-0.5B-Instruct, turns each attention head into a flag complex, computes H₁ persistent homology, and tests whether H₁ features track reasoning hop-count on controlled minimal pairs — answering the project's make-or-break question.

**Architecture:** Five small, single-responsibility modules under `src/`: `topology.py` (pure attention-matrix → H₁-features, the heart, fully TDD'd), `data_gen.py` (matched minimal pairs), `attn_extract.py` (model loading + attention extraction), `run_spike.py` (orchestration → parquet), `analyze.py` (paired stats + plots + verdict). Topology runs on CPU; the model runs on MPS in float32.

**Tech Stack:** Python 3.13, `uv` env, PyTorch 2.6 (MPS), `transformers`, `gudhi`, `numpy`/`scipy`/`pandas`/`pyarrow`, `statsmodels`, `matplotlib`, `pytest`.

**Critical correctness fact (verified):** `gudhi.SimplexTree.compute_persistence` must be called with `persistence_dim_max=True`, or H₁ classes living in the complex's maximal dimension (e.g. an unfilled cycle) are silently dropped.

---

## File Structure

- `pyproject.toml` — uv project + deps
- `src/topology.py` — `symmetrize`, `sparsify`, `build_simplex_tree`, `h1_intervals`, `betti1_at`, `h1_features`
- `src/data_gen.py` — `Item` dataclass, `build_pairs`
- `src/attn_extract.py` — `load_model`, `format_prompt`, `get_attentions`, `is_correct`
- `src/run_spike.py` — `main` orchestration → `results/spike.parquet`
- `src/analyze.py` — `main` paired stats + plots + verdict → `results/`
- `tests/test_topology.py` — known-homology graphs
- `tests/test_data_gen.py` — pair structure sanity
- `results/` — outputs (gitignored except the findings note)
- `FINDINGS.md` — written by hand after the run

---

## Task 0: Project setup

**Files:**
- Create: `pyproject.toml` (via `uv`)
- Create: `.gitignore`
- Create: `src/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Initialize uv project and add dependencies**

Run:
```bash
cd /Users/trunci/Desktop/res
uv init --no-workspace --name topo-attn --python 3.13
uv add torch transformers "gudhi" numpy scipy pandas pyarrow scikit-learn matplotlib statsmodels
uv add --dev pytest
```
Expected: `pyproject.toml` and `uv.lock` created; venv at `.venv`.

- [ ] **Step 2: Create package dirs and `.gitignore`**

Create `src/__init__.py` (empty) and `tests/__init__.py` (empty).

Create `.gitignore`:
```
.venv/
__pycache__/
*.pyc
results/*.parquet
results/*.npy
results/*.png
.cache/
hf_cache/
```

- [ ] **Step 3: Remove the uv starter file if present**

Run: `rm -f main.py hello.py`
Expected: no stray starter module.

- [ ] **Step 4: Verify the toolchain imports**

Run: `uv run python -c "import torch, transformers, gudhi, numpy, scipy, pandas, statsmodels; print('ok', torch.backends.mps.is_available())"`
Expected: `ok True`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/__init__.py tests/__init__.py
git commit -m "chore: scaffold uv project and dependencies"
```

---

## Task 1: Topology module (the heart, TDD)

**Files:**
- Create: `src/topology.py`
- Test: `tests/test_topology.py`

- [ ] **Step 1: Write failing tests for known-homology graphs**

Create `tests/test_topology.py`:
```python
import numpy as np
from src.topology import (
    symmetrize, sparsify, build_simplex_tree, h1_intervals, betti1_at, h1_features,
)


def _weight_matrix(n, edges):
    W = np.zeros((n, n), dtype=float)
    for i, j in edges:
        W[i, j] = 1.0
        W[j, i] = 1.0
    return W


def test_square_has_one_cycle():
    W = _weight_matrix(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    st = build_simplex_tree(W)
    intervals = h1_intervals(st)
    assert betti1_at(intervals, 0.5) == 1


def test_filled_triangle_has_no_cycle():
    # All three edges present -> expansion(2) fills the 2-simplex -> no H1.
    W = _weight_matrix(3, [(0, 1), (1, 2), (2, 0)])
    st = build_simplex_tree(W)
    intervals = h1_intervals(st)
    assert betti1_at(intervals, 0.5) == 0


def test_two_disjoint_squares_have_two_cycles():
    W = _weight_matrix(8, [(0, 1), (1, 2), (2, 3), (3, 0),
                           (4, 5), (5, 6), (6, 7), (7, 4)])
    st = build_simplex_tree(W)
    intervals = h1_intervals(st)
    assert betti1_at(intervals, 0.5) == 2


def test_path_has_no_cycle():
    W = _weight_matrix(4, [(0, 1), (1, 2), (2, 3)])
    st = build_simplex_tree(W)
    intervals = h1_intervals(st)
    assert betti1_at(intervals, 0.5) == 0


def test_symmetrize_takes_elementwise_max():
    A = np.array([[0.0, 0.7], [0.2, 0.0]])
    W = symmetrize(A)
    assert W[0, 1] == 0.7 and W[1, 0] == 0.7


def test_sparsify_keeps_top_k_per_node_symmetric():
    A = np.array([
        [0.0, 0.9, 0.1, 0.05],
        [0.9, 0.0, 0.8, 0.2],
        [0.1, 0.8, 0.0, 0.7],
        [0.05, 0.2, 0.7, 0.0],
    ])
    W = sparsify(A, top_k=1)
    # Result must stay symmetric and keep only strong edges.
    assert np.allclose(W, W.T)
    assert W[0, 1] > 0  # node 0's strongest edge survives


def test_h1_features_on_square_reports_one_cycle():
    W = _weight_matrix(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    feats = h1_features(W, thresholds=(0.5,), top_k=8)
    assert feats["n_cycles"] == 1
    assert feats["betti1_t0.5"] == 1
    assert feats["max_persistence"] >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_topology.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.topology'`

- [ ] **Step 3: Implement `src/topology.py`**

Create `src/topology.py`:
```python
"""Attention matrix -> flag complex -> H1 persistent homology features.

Pure functions only (no model/torch dependency) so this module is fast to test.
"""
from __future__ import annotations

import numpy as np
import gudhi


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Elementwise-max symmetrization of a (causal) attention matrix."""
    return np.maximum(A, A.T)


def sparsify(W: np.ndarray, top_k: int | None = None, weight_floor: float = 0.0) -> np.ndarray:
    """Zero the diagonal, drop weak edges, keep top_k per node, re-symmetrize.

    top_k is applied per row, which breaks symmetry, so we re-symmetrize with max
    afterward (an edge survives if it is in either endpoint's top_k).
    """
    W = W.astype(float).copy()
    np.fill_diagonal(W, 0.0)
    if weight_floor > 0.0:
        W[W < weight_floor] = 0.0
    if top_k is not None:
        n = W.shape[0]
        kept = np.zeros_like(W)
        for i in range(n):
            row = W[i]
            nz = np.flatnonzero(row)
            if nz.size == 0:
                continue
            k = min(top_k, nz.size)
            top = nz[np.argpartition(row[nz], -k)[-k:]]
            kept[i, top] = row[top]
        W = np.maximum(kept, kept.T)
    return W


def build_simplex_tree(W: np.ndarray) -> gudhi.SimplexTree:
    """Build the flag (clique) complex with filtration f(edge) = 1 - weight.

    Vertices enter at 0; strong edges (weight near 1) enter early (near 0).
    Expanded to dimension 2 so triangles can fill in and kill 1-cycles.
    """
    n = W.shape[0]
    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i], filtration=0.0)
    iu = np.triu_indices(n, k=1)
    for i, j in zip(*iu):
        w = W[i, j]
        if w > 0.0:
            st.insert([int(i), int(j)], filtration=float(1.0 - w))
    st.expansion(2)
    return st


def h1_intervals(st: gudhi.SimplexTree) -> np.ndarray:
    """Return the dimension-1 persistence intervals as an (m, 2) array.

    persistence_dim_max=True is REQUIRED: without it, H1 classes living in the
    complex's maximal dimension (e.g. an unfilled cycle) are silently dropped.
    """
    st.compute_persistence(persistence_dim_max=True)
    intervals = st.persistence_intervals_in_dimension(1)
    return np.asarray(intervals, dtype=float).reshape(-1, 2)


def betti1_at(intervals: np.ndarray, t: float) -> int:
    """Number of H1 classes alive at filtration value t."""
    if intervals.size == 0:
        return 0
    births, deaths = intervals[:, 0], intervals[:, 1]
    return int(np.sum((births <= t) & (t < deaths)))


def h1_features(
    A: np.ndarray,
    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
    top_k: int | None = 8,
    weight_floor: float = 0.0,
    symmetrized: bool = False,
) -> dict:
    """Full attention-matrix -> H1 feature dict.

    Set symmetrized=True when A is already a symmetric weight matrix (tests).
    """
    W = A.astype(float) if symmetrized else symmetrize(A)
    W = sparsify(W, top_k=top_k, weight_floor=weight_floor)
    intervals = h1_intervals(build_simplex_tree(W))
    finite = intervals[np.isfinite(intervals[:, 1])] if intervals.size else intervals
    persist = (finite[:, 1] - finite[:, 0]) if finite.size else np.array([])
    feats = {
        "n_cycles": int(intervals.shape[0]),
        "total_persistence": float(persist.sum()) if persist.size else 0.0,
        "max_persistence": float(persist.max()) if persist.size else 0.0,
    }
    for t in thresholds:
        feats[f"betti1_t{t}"] = betti1_at(intervals, t)
    return feats
```

Note: `test_h1_features_on_square_reports_one_cycle` passes a symmetric `W` but calls `h1_features(W, ...)` without `symmetrized=True`; `symmetrize` of an already-symmetric matrix is a no-op (max with its own transpose), so the test is correct.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_topology.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topology.py tests/test_topology.py
git commit -m "feat: H1 persistent-homology features from attention matrices (TDD)"
```

---

## Task 2: Minimal-pair data generation (TDD)

**Files:**
- Create: `src/data_gen.py`
- Test: `tests/test_data_gen.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data_gen.py`:
```python
from src.data_gen import build_pairs, Item


def test_build_pairs_returns_items():
    items = build_pairs(n_per_family=5, seed=0)
    assert len(items) > 0
    assert all(isinstance(it, Item) for it in items)


def test_each_pair_has_one_1hop_and_one_2hop():
    items = build_pairs(n_per_family=5, seed=0)
    by_pair = {}
    for it in items:
        by_pair.setdefault(it.pair_id, []).append(it)
    for pid, group in by_pair.items():
        hops = sorted(it.hop for it in group)
        assert hops == [1, 2], f"pair {pid} has hops {hops}"


def test_pair_members_share_context_and_differ_in_question():
    items = build_pairs(n_per_family=5, seed=0)
    by_pair = {}
    for it in items:
        by_pair.setdefault(it.pair_id, []).append(it)
    for group in by_pair.values():
        one = next(it for it in group if it.hop == 1)
        two = next(it for it in group if it.hop == 2)
        # Shared leading context (everything up to the question) must match.
        assert one.context == two.context
        assert one.prompt != two.prompt


def test_gold_answers_appear_in_context():
    items = build_pairs(n_per_family=5, seed=0)
    for it in items:
        assert it.gold in it.context


def test_seed_is_deterministic():
    a = build_pairs(n_per_family=5, seed=42)
    b = build_pairs(n_per_family=5, seed=42)
    assert [it.prompt for it in a] == [it.prompt for it in b]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data_gen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data_gen'`

- [ ] **Step 3: Implement `src/data_gen.py`**

Create `src/data_gen.py`:
```python
"""Controlled minimal pairs: matched 1-hop / 2-hop questions over identical context."""
from __future__ import annotations

import random
from dataclasses import dataclass

NAMES = [
    "Tom", "Mary", "Sue", "Anna", "Beth", "Cara", "David", "Emma", "Frank",
    "Grace", "Henry", "Iris", "Jack", "Kara", "Liam", "Nina", "Owen", "Pia",
    "Quinn", "Rosa", "Sam", "Tina", "Umar", "Vera", "Will", "Xena", "Yara", "Zane",
]


@dataclass(frozen=True)
class Item:
    id: str
    family: str
    hop: int           # 1 or 2
    pair_id: str
    context: str       # identical within a pair
    question: str
    prompt: str        # context + " " + question
    gold: str


def _kinship_pair(pid: str, names: list[str]) -> list[Item]:
    a, b, c = names
    ctx = f"{a} is {b}'s father. {b} is {c}'s father."
    q2 = f"Who is {c}'s grandfather?"
    q1 = f"Who is {c}'s father?"
    return [
        Item(f"{pid}-h2", "kinship", 2, pid, ctx, q2, f"{ctx} {q2}", a),
        Item(f"{pid}-h1", "kinship", 1, pid, ctx, q1, f"{ctx} {q1}", b),
    ]


def _ordering_pair(pid: str, names: list[str]) -> list[Item]:
    a, b, c = names
    ctx = f"{a} is taller than {b}. {b} is taller than {c}."
    q2 = f"Among {a}, {b}, and {c}, who is the tallest?"
    q1 = f"Who is taller, {a} or {b}?"
    return [
        Item(f"{pid}-h2", "ordering", 2, pid, ctx, q2, f"{ctx} {q2}", a),
        Item(f"{pid}-h1", "ordering", 1, pid, ctx, q1, f"{ctx} {q1}", a),
    ]


_BUILDERS = {"kinship": _kinship_pair, "ordering": _ordering_pair}


def build_pairs(n_per_family: int = 30, seed: int = 0) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []
    for family, builder in _BUILDERS.items():
        for k in range(n_per_family):
            names = rng.sample(NAMES, 3)
            pid = f"{family}-{k:03d}"
            items.extend(builder(pid, names))
    return items
```

Note: in the ordering family both 1-hop and 2-hop gold answers are `a`. That is acceptable: the topology comparison is per-matched-pair on the *prompts*, and `a` legitimately appears in the context for both, satisfying `test_gold_answers_appear_in_context`. The questions still differ (direct comparison vs. transitive chaining), which is the variable under test.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data_gen.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data_gen.py tests/test_data_gen.py
git commit -m "feat: controlled 1-hop/2-hop minimal pairs (TDD)"
```

---

## Task 3: Attention extraction

**Files:**
- Create: `src/attn_extract.py`
- Test: `tests/test_attn_extract.py`

This task loads a real ~1GB model, so its test is marked slow and downloads weights on first run.

- [ ] **Step 1: Write a slow integration test**

Create `tests/test_attn_extract.py`:
```python
import numpy as np
import pytest

slow = pytest.mark.slow


@slow
def test_get_attentions_shape_and_range():
    from src.attn_extract import load_model, get_attentions
    model, tok, device = load_model()
    atts = get_attentions(model, tok, device, "Tom is Mary's father. Who is Mary's father?")
    # [layers, heads, n, n]
    assert atts.ndim == 4
    assert atts.shape[2] == atts.shape[3]
    # attention rows are probability distributions -> each row sums to ~1
    row_sums = atts.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-3)


@slow
def test_is_correct_runs_and_returns_bool_and_text():
    from src.attn_extract import load_model, is_correct
    model, tok, device = load_model()
    ok, ans = is_correct(model, tok, device, "Tom is Mary's father. Who is Mary's father?", "Tom")
    assert isinstance(ok, bool)
    assert isinstance(ans, str)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_attn_extract.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.attn_extract'`

- [ ] **Step 3: Implement `src/attn_extract.py`**

Create `src/attn_extract.py`:
```python
"""Load Qwen2.5-0.5B-Instruct and extract per-head attention matrices.

attn_implementation="eager" is REQUIRED: the default SDPA/flash path does not
return attention weights, so output_attentions would be None.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


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


def format_prompt(tok, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def get_attentions(model, tok, device, text: str) -> np.ndarray:
    """Return attentions as a numpy array [layers, heads, n, n]."""
    enc = tok(format_prompt(tok, text), return_tensors="pt").to(device)
    out = model(**enc, output_attentions=True)
    atts = [a[0].float().cpu().numpy() for a in out.attentions]  # each [heads, n, n]
    return np.stack(atts)


@torch.no_grad()
def is_correct(model, tok, device, text: str, gold: str, max_new_tokens: int = 12):
    enc = tok(format_prompt(tok, text), return_tensors="pt").to(device)
    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    ans = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return (gold.lower() in ans.lower()), ans
```

- [ ] **Step 4: Register the `slow` marker and run the test**

Add to `pyproject.toml` under a new `[tool.pytest.ini_options]` section:
```toml
[tool.pytest.ini_options]
markers = ["slow: downloads/loads the real model"]
```

Run: `uv run pytest tests/test_attn_extract.py -v -m slow`
Expected: PASS (2 passed) — first run downloads ~1GB of weights.

- [ ] **Step 5: Commit**

```bash
git add src/attn_extract.py tests/test_attn_extract.py pyproject.toml
git commit -m "feat: Qwen2.5-0.5B attention extraction and correctness check"
```

---

## Task 4: Orchestration → parquet

**Files:**
- Create: `src/run_spike.py`
- Create: `results/` (directory; created by the script)

- [ ] **Step 1: Implement `src/run_spike.py`**

Create `src/run_spike.py`:
```python
"""Run the spike: generate pairs, extract attention, compute H1 features per head."""
from __future__ import annotations

import os
import pandas as pd

from src.data_gen import build_pairs
from src.attn_extract import load_model, get_attentions, is_correct
from src.topology import h1_features

THRESHOLDS = (0.3, 0.5, 0.7)
TOP_K = 8


def main(n_per_family: int = 30, seed: int = 0, out_path: str = "results/spike.parquet"):
    os.makedirs("results", exist_ok=True)
    items = build_pairs(n_per_family=n_per_family, seed=seed)
    model, tok, device = load_model()

    rows = []
    for idx, it in enumerate(items):
        ok, ans = is_correct(model, tok, device, it.prompt, it.gold)
        atts = get_attentions(model, tok, device, it.prompt)  # [L, H, n, n]
        n_layers, n_heads, n, _ = atts.shape
        for layer in range(n_layers):
            for head in range(n_heads):
                feats = h1_features(
                    atts[layer, head], thresholds=THRESHOLDS, top_k=TOP_K
                )
                rows.append({
                    "item_id": it.id, "pair_id": it.pair_id, "family": it.family,
                    "hop": it.hop, "correct": ok, "seq_len": n,
                    "layer": layer, "head": head, **feats,
                })
        print(f"[{idx+1}/{len(items)}] {it.id} correct={ok} ans={ans!r}")

    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"\nWrote {len(df)} rows to {out_path}")
    print(f"Overall task accuracy: {df.groupby('item_id')['correct'].first().mean():.2%}")
    return df


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test with a tiny run**

Run: `uv run python -c "from src.run_spike import main; df = main(n_per_family=2); print(df.shape); print(sorted(df.columns))"`
Expected: a DataFrame with rows = 2 families × 2 pairs-each... = `2*2*2 = 8 items × n_layers × n_heads`, columns including `layer, head, hop, correct, n_cycles, total_persistence, max_persistence, betti1_t0.3, betti1_t0.5, betti1_t0.7`. Prints overall accuracy.

- [ ] **Step 3: Commit**

```bash
git add src/run_spike.py
git commit -m "feat: orchestrate spike run into results dataframe"
```

---

## Task 5: Analysis, plots, and verdict

**Files:**
- Create: `src/analyze.py`

- [ ] **Step 1: Implement `src/analyze.py`**

Create `src/analyze.py`:
```python
"""Evaluate the pre-registered criteria from results/spike.parquet."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

METRIC = "total_persistence"   # primary H1 summary for the paired test
ALPHA = 0.05


def paired_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (layer, head): paired Wilcoxon of metric, 2-hop vs 1-hop."""
    out = []
    for (layer, head), g in df.groupby(["layer", "head"]):
        piv = g.pivot_table(index="pair_id", columns="hop", values=metric)
        piv = piv.dropna()
        if 1 not in piv.columns or 2 not in piv.columns or len(piv) < 5:
            continue
        diff = piv[2].values - piv[1].values
        if np.allclose(diff, 0):
            p, stat = 1.0, 0.0
        else:
            stat, p = wilcoxon(piv[2].values, piv[1].values)
        # matched-pairs effect size: mean signed diff normalized by its SD
        eff = diff.mean() / (diff.std() + 1e-9)
        out.append({"layer": layer, "head": head, "n_pairs": len(piv),
                    "mean_diff": diff.mean(), "effect": eff, "p": p})
    res = pd.DataFrame(out)
    if len(res):
        res["p_adj"] = multipletests(res["p"], alpha=ALPHA, method="fdr_bh")[1]
        res["sig"] = res["p_adj"] < ALPHA
    return res


def nontriviality(df: pd.DataFrame) -> float:
    """Fraction of (head, example) observations with a non-trivial H1 cycle."""
    has_cycle = (df["n_cycles"] > 0) & (df["max_persistence"] > 0.05)
    return float(has_cycle.mean())


def main(in_path: str = "results/spike.parquet"):
    df = pd.read_parquet(in_path)

    # --- Criterion 1: non-triviality ---
    frac = nontriviality(df)
    crit1 = frac > 0.10
    print(f"[Criterion 1] non-trivial-H1 fraction = {frac:.2%} -> {'PASS' if crit1 else 'FAIL'}")

    # --- Criterion 2: discrimination ---
    res = paired_table(df, METRIC)
    n_sig = int(res["sig"].sum()) if len(res) else 0
    crit2 = n_sig > 0
    print(f"[Criterion 2] significant (layer,head) after BH = {n_sig} -> {'PASS' if crit2 else 'FAIL'}")
    if len(res):
        print("\nTop discriminating heads:")
        print(res.sort_values("p_adj").head(10).to_string(index=False))

    # --- Criterion 3: layer clustering (plausibility) ---
    if n_sig > 0:
        sig_layers = res.loc[res["sig"], "layer"]
        n_layers = df["layer"].max() + 1
        print(f"\n[Criterion 3] sig heads layer median = {sig_layers.median():.1f} "
              f"of {n_layers} (mid/late => plausible)")

    # --- Plots ---
    if len(res):
        n_layers = int(df["layer"].max() + 1)
        n_heads = int(df["head"].max() + 1)
        grid = np.full((n_layers, n_heads), np.nan)
        for _, r in res.iterrows():
            grid[int(r.layer), int(r.head)] = -np.log10(max(r.p_adj, 1e-12))
        plt.figure(figsize=(8, 6))
        plt.imshow(grid, aspect="auto", cmap="viridis")
        plt.colorbar(label="-log10(p_adj)")
        plt.xlabel("head"); plt.ylabel("layer")
        plt.title(f"2-hop vs 1-hop discrimination ({METRIC})")
        plt.tight_layout(); plt.savefig("results/heatmap.png", dpi=120)
        print("\nSaved results/heatmap.png")

        # distribution plot for the single most discriminating head
        best = res.sort_values("p_adj").iloc[0]
        g = df[(df.layer == best.layer) & (df.head == best.head)]
        plt.figure(figsize=(6, 4))
        for hop, sub in g.groupby("hop"):
            plt.hist(sub[METRIC], bins=20, alpha=0.5, label=f"{hop}-hop")
        plt.legend(); plt.xlabel(METRIC)
        plt.title(f"L{int(best.layer)}H{int(best.head)} (p_adj={best.p_adj:.1e})")
        plt.tight_layout(); plt.savefig("results/best_head_dist.png", dpi=120)
        print("Saved results/best_head_dist.png")

    # --- Verdict ---
    if crit1 and crit2:
        verdict = "GREEN: topology is non-trivial AND tracks hop-count. Build Experiment 1."
    elif crit1:
        verdict = "YELLOW: topology is real but not reasoning-linked. Pivot to runtime diagnostic."
    else:
        verdict = "RED: topology is trivial. Stop or rethink the premise."
    print(f"\n=== VERDICT === {verdict}")
    return res


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test analyze on the tiny run from Task 4**

Run: `uv run python -c "from src.run_spike import main as run; run(n_per_family=6); from src.analyze import main as an; an()"`
Expected: prints the three criteria lines and a VERDICT line; writes `results/heatmap.png` and (if any signal) `results/best_head_dist.png`. With only 6 pairs/family the verdict is not meaningful yet — this only confirms the code path runs end-to-end.

- [ ] **Step 3: Commit**

```bash
git add src/analyze.py
git commit -m "feat: paired stats, plots, and pre-registered verdict"
```

---

## Task 6: Full run and findings note

**Files:**
- Create: `FINDINGS.md`

- [ ] **Step 1: Run the full spike**

Run: `uv run python -m src.run_spike`
Expected: progress lines for `2 × 30 = 60 pairs = 120 items`; writes `results/spike.parquet`; prints overall task accuracy. (If accuracy is very low, note it — it affects interpretation.)

- [ ] **Step 2: Run the analysis**

Run: `uv run python -m src.analyze`
Expected: three criteria lines, top heads table, plots, and a VERDICT.

- [ ] **Step 3: Write `FINDINGS.md`**

Record, with the actual numbers from Step 2: overall task accuracy; non-trivial-H1 fraction (criterion 1); number of significant heads after BH and the top-10 table (criterion 2); the layer-clustering observation (criterion 3); the verdict; and a short "what this means for the project / next step" paragraph mapping the verdict to the green/yellow/red action from the spec. Reference the two plot files.

- [ ] **Step 4: Commit**

```bash
git add FINDINGS.md
git commit -m "docs: spike findings and verdict"
```

---

## Self-Review notes

- **Spec coverage:** non-triviality (Task 5 `nontriviality` + criterion 1), discrimination (Task 5 `paired_table` + Wilcoxon + BH, criterion 2), layer clustering (criterion 3), persistent-homology-over-filtration method (Task 1), minimal pairs (Task 2), Qwen2.5-0.5B + MPS + eager (Task 3), parquet + plots + findings note (Tasks 4–6), TDD on topology with the known-homology fixtures (Task 1). All spec sections map to a task.
- **Type consistency:** feature keys (`n_cycles`, `total_persistence`, `max_persistence`, `betti1_t{threshold}`) are produced in `h1_features` (Task 1) and consumed unchanged in `run_spike` and `analyze`. `Item` fields (`id, family, hop, pair_id, context, question, prompt, gold`) defined in Task 2 and used in Task 4. `get_attentions` returns `[L, H, n, n]` consumed as such in Task 4.
- **Deferred by design (spec out-of-scope):** discrete Morse matching, pruning, failure probes, 7B/GPU — intentionally not in this plan.
