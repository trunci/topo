# Design: Experiment 1 — DMT Attention Pruning vs Baselines

**Date:** 2026-05-30
**Status:** Approved (pending spec review)
**Parent proposal:** `research.md` — Experiment 1.
**Predecessor:** the de-risk spike (`FINDINGS.md`, verdict GREEN) established that attention
flag-complexes carry non-trivial H₁ that tracks reasoning. This experiment tests whether that
topology is not just *diagnostic* but *prescriptive* for pruning.

## Purpose

Answer one question: **does pruning attention by discrete-Morse critical structure preserve
model quality better than magnitude / random / sliding-window pruning at the same per-head
sparsity?**

The spike used persistent homology as a *probe* (measuring topology). This experiment builds
the proposal's actual headline method — **Forman discrete Morse theory**: a discrete gradient
vector field whose unpaired *critical cells* are the topologically essential structure — and
uses it to decide which attention edges to keep.

## Pre-registered claim and null

Decided before running.

- **Primary win condition:** at DMT's own per-head edge budget, the DMT-pruned model has
  **lower WikiText-2 perplexity than magnitude pruning**, and also beats random and
  sliding-window. Secondary axis: DMT accuracy on kinship minimal pairs is **≥ magnitude**.
- **Honest null:** if DMT ties or loses to magnitude on perplexity, that is a real and
  publishable negative result — topology would be diagnostic (spike) but not prescriptive for
  pruning. Reported straight, no goalpost-moving.

Statistics: paired comparison across examples (DMT vs each baseline), Wilcoxon signed-rank
with effect size; perplexity compared per-example via per-token loss.

## Decisions (locked during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Scope | De-risk the method first | WikiText PPL + kinship accuracy + 3 baselines; defer MMLU/MuSiQue/HotpotQA until DMT shows a win |
| Mechanism | Two-pass mask-and-rerun | Capture attention, compute keep-set, re-run with masking hook; identical treatment for every method = fair |
| Keep-set | Critical edges + Morse spanning forest | Preserve H₁ generators (essential cycles) and graph connectivity; drop triangle-filled edges |
| Sparsity match | DMT sets the per-head budget | Baselines keep *exactly* DMT's kept-edge count k per head → "equivalent sparsity" is exact, per-head; DMT cannot win by keeping more |

## Method — discrete Morse matching (`src/morse.py`, the new heart)

For each (layer, head) attention matrix, after symmetrize + sparsify (reuse `topology.py`):

1. Build the flag complex (reuse `topology.build_simplex_tree`).
2. Compute a **discrete gradient vector field** by greedy coreduction (Forman): repeatedly
   find a free face (a cell that is a face of exactly one unpaired coface) and pair them.
   Cells left unpaired at the end are **critical cells**. Critical 1-cells are H₁ generators.
3. **Keep-set** (edges to retain) =
   - all **critical 1-cells** (the essential cycles), PLUS
   - the **vertex↔edge matched pairs** that form a Morse spanning forest (retains
     connectivity of the original graph),
   and DROP edges paired *upward* into a triangle (topologically redundant).
4. Return the kept edge set and `k = len(keep_set)`.

Output interface: `morse_keep(A, top_k, weight_floor) -> set[frozenset({i,j})]`.

### Risk and mitigation (flagged)

The keep-set may be very sparse (critical cells + spanning forest can be a small fraction of
edges), pushing the whole comparison into an extreme-sparsity regime where every method is
degraded and differences are noise. **Mitigation built in from the start:** `morse_keep`
accepts an optional `min_budget_frac` (floor on k as a fraction of the sparsified edge count);
if DMT's natural keep-set is below the floor, top up with the next-highest-weight critical-
adjacent edges to reach the floor. Default floor is small; we inspect DMT's natural sparsity
first and only engage the knob if needed. Either way the budget DMT uses is what the baselines
are matched to, so fairness is preserved.

## Fair comparison — budget-matched baselines (`src/keepsets.py`)

DMT decides k per (layer, head, example). Each baseline then keeps **exactly k** edges for
that same head:

- `magnitude_keep(A, k)` — top-k symmetrized edges by attention weight.
- `random_keep(A, k, seed)` — k edges sampled uniformly from candidate edges (seeded for
  reproducibility; seed derived from layer/head/example index so it is deterministic).
- `window_keep(A, k)` — the k edges nearest the diagonal (|i−j| smallest), the sliding-window
  prior.

All return `set[frozenset({i,j})]`, same type as `morse_keep`, so they are interchangeable.

## Pruning mechanism — two-pass mask-and-rerun (`src/pruned_forward.py`)

1. **Pass 1:** forward with `output_attentions=True`; capture attention per (layer, head).
2. Compute each method's keep-set from pass-1 attention.
3. **Pass 2:** re-run the model with a forward pre-hook on each attention module that, for
   each head, **zeroes attention entries not in the keep-set and renormalizes each query row
   to sum to 1** (rows that become all-zero fall back to attending to self/diagonal to avoid
   NaNs). Read the metric from this pass.

Every method — including an **unpruned control** (keep-set = all edges) — goes through the
identical two-pass path, so measured differences are due to the pruning choice, not the
harness. Metrics per example: mean per-token cross-entropy loss (→ perplexity) on WikiText
text, and greedy-decode kinship accuracy.

Implementation note: the hook must mask the **post-softmax** attention probabilities and
renormalize, matching how the spike read attention (post-softmax, rows sum to 1). The keep-set
is symmetric (undirected edges); applied to the causal matrix, an entry (q,k) is kept if its
undirected edge is in the keep-set and q≥k (causal). Diagonal (self-attention) is always kept.

## Components

- `src/morse.py` — discrete gradient field, critical cells, `morse_keep`. **Primary TDD target.**
- `src/keepsets.py` — `magnitude_keep`, `random_keep`, `window_keep`.
- `src/pruned_forward.py` — masking hook, renormalization, two-pass driver.
- `src/eval_data.py` — WikiText-2 slice loader (HF `datasets`), plus reuse `data_gen` kinship pairs.
- `src/run_exp1.py` — orchestrate {unpruned, dmt, magnitude, random, window} × examples → `results/exp1.parquet`.
- `src/analyze_exp1.py` — per-method PPL & accuracy, paired DMT-vs-baseline tests, plots, verdict.
- `tests/test_morse.py`, `tests/test_keepsets.py`, `tests/test_pruned_forward.py`.

## Testing strategy

TDD on `morse.py` against graphs with known Morse structure:
- Single triangle (filled): 1 critical 0-cell, 0 critical 1-cells (β₁=0); keep-set connects all 3 vertices.
- 4-cycle (unfilled): exactly 1 critical 1-cell (the loop generator); keep-set is connected and retains the cycle.
- Two disjoint edges: 2 critical 0-cells (2 components); keep-set = both edges.
- Path graph: 0 critical 1-cells; keep-set = all path edges (spanning tree).

`keepsets.py`: each returns exactly k edges; magnitude returns the true top-k; window returns
the k nearest-diagonal; random is deterministic under fixed seed.

`pruned_forward.py`: after masking+renormalization, every query row sums to ~1 and all
non-kept (and non-causal) entries are 0; the unpruned control reproduces the pass-1 loss
(within fp tolerance), proving the harness itself is lossless.

## Reuse and scope guardrails (YAGNI)

- Reuse `attn_extract` (model, attention), `topology` (symmetrize/sparsify/build_simplex_tree),
  `data_gen` (kinship pairs) unchanged.
- Datasets: a small WikiText-2 slice + kinship pairs only. MMLU / MuSiQue / HotpotQA /
  2WikiMultihop deferred until DMT shows a win (matches the staged philosophy of the spike).
- **Two models, by metric:** WikiText-2 perplexity is measured on the **base** model
  `Qwen/Qwen2.5-0.5B` (more standard for language-modeling PPL); kinship accuracy uses the
  **instruct** model `Qwen/Qwen2.5-0.5B-Instruct` (chat-formatted, matches the spike). Both
  are the same 24L×14H architecture, MPS, float32, eager attention. The two-pass pruning
  harness is model-agnostic and runs identically on each.
- Compute target: a few hundred WikiText windows + 120 kinship prompts × ~5 methods, minutes
  to low-tens-of-minutes on the M3 Pro. No GPU.

## Deliverables

`results/exp1.parquet` (per example × method: loss, ppl, accuracy, mean sparsity), plots
(PPL-by-method bar with CIs; DMT-vs-magnitude per-example scatter; sparsity histogram), and a
findings note evaluating the pre-registered claim with a clear win / null verdict. All
reported numbers computed by a script from the parquet — none hand-typed (lesson carried over
from the spike's integrity failure).
