# Experiment 2 — Topology ↔ Induction Heads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether attention heads with distinctive H₁ topology coincide with induction heads (an independently-known circuit) in GPT-2 small, with a pre-registered GREEN/YELLOW/RED verdict and machine-generated findings.

**Architecture:** `induction.py` (pure functions: repeated-random batches, per-head induction score, distance baselines) feeds `run_exp2.py`, which loads GPT-2 via the existing `attn_extract.load_named`, extracts attentions, and per head computes induction score + H₁ persistence (reusing `topology.h1_features`) + distance baselines → `results/exp2.parquet`. `compute_stats_exp2.py` reads the parquet → `results/exp2_stats.json` (single source of truth: Spearman, Mann–Whitney, baseline comparison, verdict). `write_findings_exp2.py` emits `FINDINGS_exp2.md` mechanically from the JSON — no hand-typed numbers.

**Tech Stack:** Python 3.13, `uv`, PyTorch (CPU), `transformers`, `gudhi`, numpy/scipy/pandas/pyarrow, matplotlib, pytest. All installed.

**Verified facts baked in:**
1. `attn_extract.load_named("gpt2", device="cpu")` works; GPT-2 small = 12 layers × 12 heads; `output_attentions=True` returns 12 tensors of shape `[1, 12, n, n]` (eager). Use `device="cpu"` for safety (a previous MPS run crashed the machine).
2. `topology.h1_features(A, thresholds=(0.3,0.5,0.7), top_k=8, ...)` returns a dict including `total_persistence` (the primary signal). Reuse unchanged.

---

## Environment notes (read before any task)

- Run everything via `uv run ...`. A harmless `VIRTUAL_ENV ... will be ignored` warning may appear; ignore it or prefix with `unset VIRTUAL_ENV;`.
- For any command that loads the model, prefix with: `set -a; [ -f .env ] && . ./.env; set +a;` so HF downloads authenticate.
- Tests import via the `src.` package (e.g. `from src.induction import induction_score`), matching the repo convention. Implementation modules in `src/` import siblings as `from src.x import y` (matches existing `run_exp1.py`).
- All commits end with the co-author trailer shown in each step.
- Do NOT modify spike or Exp1 code; reuse `topology.py` and `attn_extract.py` as-is.

## File Structure

```
src/
  induction.py             # NEW: repeat batches, induction score, distance baselines
  run_exp2.py              # NEW: GPT-2 -> per-head induction + H1 + distance -> results/exp2.parquet
  compute_stats_exp2.py    # NEW: parquet -> results/exp2_stats.json (source of truth)
  write_findings_exp2.py   # NEW: JSON -> FINDINGS_exp2.md (no hand-typed numbers)
  analyze_exp2.py          # NEW: scatter plot from parquet + stats
tests/
  test_induction.py        # NEW
  test_run_exp2.py         # NEW (light smoke, slow)
  test_compute_stats_exp2.py  # NEW (synthetic parquet, fast)
results/
  exp2.parquet, exp2_stats.json, exp2_scatter.png   # produced
FINDINGS_exp2.md           # produced by write_findings_exp2
```

---

## Task 1: `induction.py` — induction score + baselines (TDD)

**Files:** `src/induction.py`, `tests/test_induction.py`

- [ ] **Step 1: Write the failing test** `tests/test_induction.py`:

```python
import numpy as np
from src.induction import (
    make_repeat_batch, induction_score, attention_distance, offdiag_mass,
)


def test_perfect_induction_pattern_scores_high():
    # seq layout: prefix_len=0, S=4, total n=8. Second-copy position i (i in 4..7)
    # should attend to i - S + 1 = i - 3 (the token after the first-copy match).
    n, S, prefix = 8, 4, 0
    A = np.zeros((n, n))
    for i in range(S, n):
        A[i, i - S + 1] = 1.0
    # rows that are all zero (first copy) -> leave; induction_score only reads 2nd copy
    score = induction_score(A, seq_len=S, prefix_len=prefix)
    assert score > 0.99


def test_uniform_attention_scores_low():
    n, S, prefix = 8, 4, 0
    A = np.ones((n, n))
    A = A / A.sum(axis=1, keepdims=True)
    score = induction_score(A, seq_len=S, prefix_len=prefix)
    assert score < 0.4  # chance-ish, well below the induction pattern


def test_attention_distance_diagonal_is_zero():
    A = np.eye(5)
    assert attention_distance(A) == 0.0


def test_attention_distance_fixed_offset():
    # all mass at distance 2
    n = 5
    A = np.zeros((n, n))
    for i in range(2, n):
        A[i, i - 2] = 1.0
    # rows 0,1 have no mass; distance averages over rows with mass
    assert abs(attention_distance(A) - 2.0) < 1e-9


def test_offdiag_mass_counts_offdiagonal_only():
    A = np.array([[0.5, 0.5], [0.0, 1.0]])
    # off-diagonal entries: A[0,1]=0.5, A[1,0]=0.0 -> total 0.5, averaged over 2 rows = 0.25
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
```

- [ ] **Step 2: Run it, confirm FAIL**

Run: `unset VIRTUAL_ENV; uv run pytest tests/test_induction.py -q`
Expected: `ModuleNotFoundError: No module named 'src.induction'`.

- [ ] **Step 3: Implement** `src/induction.py`:

```python
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
    n = A.shape[0]
    off = A.copy().astype(float)
    np.fill_diagonal(off, 0.0)
    return float(off.sum(axis=1).mean())
```

- [ ] **Step 4: Run it, confirm PASS**

Run: `unset VIRTUAL_ENV; uv run pytest tests/test_induction.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
cd /Users/trunci/Desktop/res && git add src/induction.py tests/test_induction.py
git commit -m "$(cat <<'EOF'
feat: induction-head score and attention baselines (TDD)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `run_exp2.py` — per-head measurements → parquet

**Files:** `src/run_exp2.py`, `tests/test_run_exp2.py`

- [ ] **Step 1: Write the failing smoke test** `tests/test_run_exp2.py`:

```python
import pytest
import pandas as pd


@pytest.mark.slow
def test_run_exp2_writes_per_head_parquet(tmp_path):
    from src.run_exp2 import run
    out = tmp_path / "exp2.parquet"
    # tiny config: short blocks, few sequences
    run(out_path=str(out), seq_len=8, prefix_len=2, n_seqs=2, top_k=8)
    df = pd.read_parquet(out)
    # GPT-2 small = 12 layers x 12 heads = 144 rows
    assert len(df) == 144
    expected = {"layer", "head", "induction_score", "h1_persistence",
                "attn_distance", "offdiag_mass"}
    assert expected.issubset(set(df.columns))
    assert df["induction_score"].notna().all()
```

- [ ] **Step 2: Run it, confirm FAIL**

Run: `unset VIRTUAL_ENV; uv run pytest tests/test_run_exp2.py -q`
Expected: `ModuleNotFoundError: No module named 'src.run_exp2'`.

- [ ] **Step 3: Implement** `src/run_exp2.py`:

```python
"""Experiment 2: per-head induction score vs H1 topology in GPT-2 small.

Loads GPT-2 (CPU), runs repeated-random sequences, and for each (layer, head)
records induction score, H1 total persistence (reusing topology.h1_features),
and cheap distance baselines -> results/exp2.parquet (one row per head).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch

from src.attn_extract import load_named
from src.topology import h1_features
from src.induction import (
    make_repeat_batch, induction_score, attention_distance, offdiag_mass,
)

MODEL = "gpt2"
SEQ_LEN = 25
PREFIX_LEN = 5
N_SEQS = 8
TOP_K = 8
SEED = 0


def run(out_path="results/exp2.parquet", seq_len=SEQ_LEN, prefix_len=PREFIX_LEN,
        n_seqs=N_SEQS, top_k=TOP_K, seed=SEED):
    os.makedirs("results", exist_ok=True)
    model, tok, device = load_named(MODEL, device="cpu")
    L = model.config.n_layer
    H = model.config.n_head
    vocab = model.config.vocab_size

    batch = make_repeat_batch(vocab, seq_len, prefix_len, n_seqs, seed=seed)
    input_ids = torch.tensor(batch, dtype=torch.long, device=device)

    # accumulate per-head sums across sequences, then average
    n = input_ids.shape[1]
    sums = {(l, h): {"ind": 0.0, "h1": 0.0, "dist": 0.0, "off": 0.0}
            for l in range(L) for h in range(H)}

    with torch.no_grad():
        for s in range(input_ids.shape[0]):
            out = model(input_ids[s:s + 1], output_attentions=True)
            for l in range(L):
                att = out.attentions[l][0].float().cpu().numpy()  # [H, n, n]
                for h in range(H):
                    A = att[h]
                    sums[(l, h)]["ind"] += induction_score(A, seq_len, prefix_len)
                    sums[(l, h)]["h1"] += h1_features(A, top_k=top_k)["total_persistence"]
                    sums[(l, h)]["dist"] += attention_distance(A)
                    sums[(l, h)]["off"] += offdiag_mass(A)
            print(f"[exp2] sequence {s + 1}/{input_ids.shape[0]} done (n={n})", flush=True)

    nseq = input_ids.shape[0]
    rows = []
    for (l, h), d in sums.items():
        rows.append({
            "layer": l, "head": h,
            "induction_score": d["ind"] / nseq,
            "h1_persistence": d["h1"] / nseq,
            "attn_distance": d["dist"] / nseq,
            "offdiag_mass": d["off"] / nseq,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"[exp2] wrote {len(df)} head rows to {out_path}", flush=True)
    return df


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run the smoke test, confirm PASS**

Run: `set -a; [ -f .env ] && . ./.env; set +a; unset VIRTUAL_ENV; uv run pytest tests/test_run_exp2.py -q -m slow`
Expected: `1 passed` (downloads GPT-2 on first run; CPU; ~a minute).

- [ ] **Step 5: Commit**

```bash
cd /Users/trunci/Desktop/res && git add src/run_exp2.py tests/test_run_exp2.py
git commit -m "$(cat <<'EOF'
feat: Experiment 2 per-head induction vs H1 topology run (GPT-2, CPU)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `compute_stats_exp2.py` — stats + verdict (TDD)

**Files:** `src/compute_stats_exp2.py`, `tests/test_compute_stats_exp2.py`

- [ ] **Step 1: Write the failing test** `tests/test_compute_stats_exp2.py`:

```python
import json
import numpy as np
import pandas as pd
from src.compute_stats_exp2 import compute_stats


def _synthetic(path, coupled=True):
    # 144 heads; induction score random in [0,1]; persistence tracks it if coupled.
    rng = np.random.default_rng(0)
    ind = rng.random(144)
    if coupled:
        h1 = ind * 2.0 + rng.normal(0, 0.05, 144)   # strongly tracks induction
    else:
        h1 = rng.random(144)                          # unrelated
    dist = rng.random(144)                            # weak/irrelevant baseline
    df = pd.DataFrame({
        "layer": np.repeat(np.arange(12), 12),
        "head": np.tile(np.arange(12), 12),
        "induction_score": ind, "h1_persistence": h1,
        "attn_distance": dist, "offdiag_mass": rng.random(144),
    })
    df.to_parquet(path)


def test_coupled_data_gives_green(tmp_path):
    pq = tmp_path / "exp2.parquet"; out = tmp_path / "exp2_stats.json"
    _synthetic(pq, coupled=True)
    stats = compute_stats(str(pq), str(out))
    assert stats["spearman_rho"] > 0
    assert stats["spearman_p"] < 0.05
    assert stats["mannwhitney_p"] < 0.05
    assert stats["verdict"].startswith("GREEN")
    assert json.loads(out.read_text()) == stats


def test_uncoupled_data_gives_red(tmp_path):
    pq = tmp_path / "exp2.parquet"; out = tmp_path / "exp2_stats.json"
    _synthetic(pq, coupled=False)
    stats = compute_stats(str(pq), str(out))
    assert stats["verdict"].startswith("RED")
```

- [ ] **Step 2: Run it, confirm FAIL**

Run: `unset VIRTUAL_ENV; uv run pytest tests/test_compute_stats_exp2.py -q`
Expected: `ModuleNotFoundError: No module named 'src.compute_stats_exp2'`.

- [ ] **Step 3: Implement** `src/compute_stats_exp2.py`:

```python
"""Compute Experiment-2 statistics from the parquet into a JSON (source of truth).

Pre-registered (see spec):
  1. discrimination: Spearman(induction, H1 persistence) > 0 and p < 0.05
  2. separation: top-k induction heads have higher H1 persistence (Mann-Whitney)
  3. beats baseline: |rho(persistence,induction)| >= |rho(distance,induction)| - 0.05
Verdict: GREEN if 1&2 and not dominated by distance; YELLOW if 1&2 but distance
as good/better; RED if 1 or 2 fails.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

TOP_K = 10
ALPHA = 0.05
BASELINE_SLACK = 0.05


def compute_stats(parquet_path: str, out_path: str) -> dict:
    df = pd.read_parquet(parquet_path)
    ind = df["induction_score"].to_numpy()
    h1 = df["h1_persistence"].to_numpy()
    dist = df["attn_distance"].to_numpy()

    rho, p = spearmanr(ind, h1)
    drho, dp = spearmanr(ind, dist)

    # separation: top-k by induction score vs the rest, compare H1 persistence
    order = np.argsort(ind)[::-1]
    top_idx = order[:TOP_K]
    rest_idx = order[TOP_K:]
    top_h1 = h1[top_idx]
    rest_h1 = h1[rest_idx]
    if len(rest_h1) > 0 and len(top_h1) > 0:
        u_stat, mw_p = mannwhitneyu(top_h1, rest_h1, alternative="greater")
    else:
        u_stat, mw_p = float("nan"), float("nan")

    crit1 = bool(rho > 0 and p < ALPHA)
    crit2 = bool(mw_p < ALPHA)
    not_dominated = bool(abs(rho) >= abs(drho) - BASELINE_SLACK)

    if crit1 and crit2 and not_dominated:
        verdict = "GREEN: topology corresponds to induction heads beyond a trivial distance baseline."
    elif crit1 and crit2:
        verdict = "YELLOW: topology correlates with induction but a distance baseline does as well or better."
    else:
        verdict = "RED: no significant topology-induction correspondence."

    stats = {
        "n_heads": int(len(df)),
        "top_k": TOP_K,
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(mw_p),
        "top_mean_h1": float(np.mean(top_h1)),
        "rest_mean_h1": float(np.mean(rest_h1)),
        "baseline_distance_rho": float(drho),
        "baseline_distance_p": float(dp),
        "topology_not_dominated_by_distance": not_dominated,
        "top_induction_heads": [
            f"L{int(df.iloc[i]['layer'])}H{int(df.iloc[i]['head'])} "
            f"ind={ind[i]:.3f} h1={h1[i]:.3f}" for i in top_idx
        ],
        "verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    compute_stats("results/exp2.parquet", "results/exp2_stats.json")
    print("WROTE results/exp2_stats.json")
```

- [ ] **Step 4: Run it, confirm PASS**

Run: `unset VIRTUAL_ENV; uv run pytest tests/test_compute_stats_exp2.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
cd /Users/trunci/Desktop/res && git add src/compute_stats_exp2.py tests/test_compute_stats_exp2.py
git commit -m "$(cat <<'EOF'
feat: Experiment 2 stats and verdict (Spearman + Mann-Whitney + baseline) (TDD)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `analyze_exp2.py` + `write_findings_exp2.py`

**Files:** `src/analyze_exp2.py`, `src/write_findings_exp2.py`

- [ ] **Step 1: Implement** `src/analyze_exp2.py` (plot only; no asserted numbers):

```python
"""Experiment 2 plot: induction score vs H1 persistence, top induction heads marked."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze(parquet_path="results/exp2.parquet", out_png="results/exp2_scatter.png",
            top_k=10):
    df = pd.read_parquet(parquet_path)
    ind = df["induction_score"].to_numpy()
    h1 = df["h1_persistence"].to_numpy()
    top = np.argsort(ind)[::-1][:top_k]
    mask = np.zeros(len(df), dtype=bool)
    mask[top] = True

    plt.figure(figsize=(6, 5))
    plt.scatter(ind[~mask], h1[~mask], alpha=0.5, label="other heads")
    plt.scatter(ind[mask], h1[mask], color="crimson", label=f"top-{top_k} induction")
    plt.xlabel("induction score")
    plt.ylabel("H1 total persistence")
    plt.title("GPT-2 small: induction vs attention-graph topology")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"wrote {out_png}")


if __name__ == "__main__":
    analyze()
```

- [ ] **Step 2: Implement** `src/write_findings_exp2.py`:

```python
"""Generate FINDINGS_exp2.md mechanically from results/exp2_stats.json.

Every number is read from the JSON; prose is fixed here. No value is hand-typed.
"""
from __future__ import annotations

import json


def _f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def build(s: dict) -> str:
    heads = "\n".join("- " + h for h in s["top_induction_heads"])
    return f"""# Experiment 2 Findings: Topology vs Induction Heads

**Date:** 2026-05-30
**Verdict:** {s["verdict"]}

Every number here is generated by `src/write_findings_exp2.py` from
`results/exp2_stats.json` (produced by `src/compute_stats_exp2.py`). No value is
hand-typed — a guard adopted after earlier fabrication incidents in this project.

## Question
Do attention heads with distinctive H1 topology coincide with induction heads, a circuit
detected independently of any topology? Model: GPT-2 small, {s["n_heads"]} heads.

## Pre-registered tests
- **Discrimination** — Spearman(induction score, H1 persistence): rho = {_f(s["spearman_rho"])},
  p = {_f(s["spearman_p"], 5)}.
- **Separation** — top-{s["top_k"]} induction heads vs the rest, H1 persistence
  (Mann-Whitney, greater): U = {_f(s["mannwhitney_u"], 1)}, p = {_f(s["mannwhitney_p"], 5)}.
  Mean H1 persistence: top induction heads {_f(s["top_mean_h1"])} vs rest {_f(s["rest_mean_h1"])}.
- **Beats trivial baseline** — Spearman(induction, attention distance): rho =
  {_f(s["baseline_distance_rho"])}, p = {_f(s["baseline_distance_p"], 5)}. Topology not
  dominated by distance: {s["topology_not_dominated_by_distance"]}.

## Top induction heads (by induction score)
{heads}

## Reading
The verdict field above is machine-checked against the pre-registered rule (GREEN: topology
corresponds to induction beyond the distance baseline; YELLOW: correlates but the trivial
baseline does as well; RED: no correspondence). Whatever the outcome, it is reported as-is.

## Plot
- results/exp2_scatter.png — induction score vs H1 persistence, top induction heads marked.

## Caveats
- Single model (GPT-2 small), CPU, modest batch.
- Induction is one circuit; correspondence (or its absence) here does not generalize to all
  circuits (IOI etc. untested).

## Reproduce
```bash
uv run python -m src.run_exp2              # results/exp2.parquet
uv run python -m src.compute_stats_exp2    # results/exp2_stats.json (source of truth)
uv run python -m src.write_findings_exp2   # this file
uv run python -m src.analyze_exp2          # results/exp2_scatter.png
```
"""


def main(stats_path="results/exp2_stats.json", out_path="FINDINGS_exp2.md"):
    with open(stats_path) as f:
        s = json.load(f)
    with open(out_path, "w") as f:
        f.write(build(s))
    print(f"wrote {out_path} from {stats_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify both import cleanly**

Run: `unset VIRTUAL_ENV; uv run python -c "import src.analyze_exp2, src.write_findings_exp2; print('imports OK')"`
Expected: `imports OK`.

- [ ] **Step 4: Commit**

```bash
cd /Users/trunci/Desktop/res && git add src/analyze_exp2.py src/write_findings_exp2.py
git commit -m "$(cat <<'EOF'
feat: Experiment 2 scatter plot and machine-generated findings writer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Full run + findings (STOP after this — do not over-run)

**Files:** produces `results/exp2.parquet`, `results/exp2_stats.json`, `results/exp2_scatter.png`, `FINDINGS_exp2.md`

- [ ] **Step 1: Run the experiment (CPU, single process, with progress)**

Run: `set -a; [ -f .env ] && . ./.env; set +a; unset VIRTUAL_ENV; uv run python -m src.run_exp2`
Expected: per-sequence progress lines, then `[exp2] wrote 144 head rows to results/exp2.parquet`.

- [ ] **Step 2: Compute stats, generate plot + findings**

Run:
```bash
unset VIRTUAL_ENV; uv run python -m src.compute_stats_exp2
unset VIRTUAL_ENV; uv run python -m src.analyze_exp2
unset VIRTUAL_ENV; uv run python -m src.write_findings_exp2
```

- [ ] **Step 3: READ the JSON and confirm the findings match before trusting**

Run: `unset VIRTUAL_ENV; uv run python -c "import json; print(json.dumps(json.load(open('results/exp2_stats.json')), indent=2))"`
Confirm `n_heads == 144` and that `FINDINGS_exp2.md` shows the same rho / p / verdict.

- [ ] **Step 4: Commit**

```bash
cd /Users/trunci/Desktop/res && git add FINDINGS_exp2.md && git add -f results/exp2_stats.json results/exp2_scatter.png
git commit -m "$(cat <<'EOF'
Add Experiment 2 results and findings (machine-generated from stats JSON)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Final fast-test check**

Run: `unset VIRTUAL_ENV; uv run pytest -q -m "not slow"`
Expected: all fast tests pass (induction, compute_stats_exp2, plus pre-existing).

---

## Self-Review notes (spec → task mapping)

- Induction score (independent detector) → Task 1 `induction_score`, tested for ≈1 on a perfect pattern, low on uniform.
- Repeated-random batch → Task 1 `make_repeat_batch`, tested identical halves + determinism.
- Distance baseline → Task 1 `attention_distance` + `offdiag_mass`, tested on known matrices.
- H1 persistence signal (reuse topology) → Task 2 calls `topology.h1_features(...)["total_persistence"]`.
- GPT-2 small, CPU, 144 heads → Task 2 `run_exp2` (verified load_named("gpt2") works, 12×12).
- Pre-registered criteria 1/2/3 + verdict → Task 3 `compute_stats_exp2` (Spearman, Mann-Whitney top-k=10, baseline slack 0.05), tested GREEN on coupled / RED on uncoupled synthetic data.
- Single source of truth JSON → Task 3 writes `exp2_stats.json`; test asserts on-disk == returned.
- No hand-typed numbers → Task 4 `write_findings_exp2` emits FINDINGS from JSON; Task 5 reads JSON before trusting.
- Scatter plot deliverable → Task 4 `analyze_exp2`.
- Safety (CPU, one model, progress) → Task 2 uses device="cpu", per-sequence flushed prints.

**Signature consistency:** `make_repeat_batch(vocab_size, seq_len, prefix_len, n_seqs, seed)`, `induction_score(A, seq_len, prefix_len)`, `attention_distance(A)`, `offdiag_mass(A)` — defined Task 1, called Task 2. `run(out_path, seq_len, prefix_len, n_seqs, top_k, seed)` — Task 2, called by test + Task 5. `compute_stats(parquet_path, out_path)` — Task 3, called by test + Task 5. `analyze(parquet_path, out_png, top_k)` / `build(stats)` / `main(...)` — Task 4. All consistent.
```
