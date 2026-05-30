# Design: Topology-of-Attention De-Risk Spike

**Date:** 2026-05-30
**Status:** Approved (pending spec review)
**Parent proposal:** `research.md` — *Discrete Morse Theory for Attention Sparsification and Mechanistic Interpretability in LLMs*

## Purpose

The full proposal rests on one untested empirical assumption: that the flag
complex of a transformer's attention graph has **non-trivial topology (H₁ ≠ 0)
that tracks reasoning**. If β₁ is ~0 everywhere, or carries no relationship to
reasoning, the project is dead on arrival.

This spike answers that single make-or-break question **cheaply and first**,
before any pruning or discrete-Morse infrastructure is built. It is the
de-risking stage of an explicitly staged commitment.

## Scope

In scope:
- A minimal, model-agnostic pipeline: attention extraction → flag complex →
  persistent homology → H₁ features.
- A controlled experiment testing whether H₁ features track reasoning hop-count.
- A pre-registered verdict that decides whether and how to proceed.

Explicitly **out of scope** for this spike (deferred until after a green verdict):
- Discrete Morse matching / critical cells (the proposal's headline method).
- Attention pruning and the sparsification baselines (Experiment 1).
- Failure-prediction probes (Experiment 3).
- 7B-scale models and GPU rental.

## Decisions (locked during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Sequencing | De-risk spike first | Test the riskiest assumption before building infra |
| Model | Qwen2.5-0.5B-Instruct | ~1GB, fast on MPS, genuinely capable of simple multi-hop reasoning |
| Test data | Controlled synthetic minimal pairs | Isolates hop-count; controls length/vocab; verifiable |
| Topology method | Persistent homology over a weight filtration | Robust to threshold; yields β₁ AND persistence features; matches proposal |
| Compute | Local, Apple M3 Pro (18GB, MPS) | The expensive part is topology (CPU), not the model |

## Pre-registered success criteria

Decided **before** running, to prevent post-hoc rationalization.

1. **Non-triviality** — H₁ is non-trivial (β₁ > 0 with non-negligible
   persistence) in a substantial fraction of (layer, head) pairs on these inputs.
2. **Discrimination** — at least some (layer, head) pairs show a statistically
   significant difference in an H₁ summary (β₁ and/or total H₁ persistence)
   between the multi-hop and single-hop members of matched pairs. Paired Wilcoxon
   signed-rank test, reported with effect size, Benjamini–Hochberg corrected
   across the (layer, head) family.
3. **Plausibility (bonus)** — discriminating heads cluster in mid/late layers
   rather than scattering randomly.

**Verdict rule:**
- **Green** (criteria 1 AND 2 hold) → build Experiment 1.
- **Yellow** (only 1 holds) → topology is real but not reasoning-linked; pivot
  toward the runtime-diagnostic framing (fallback #1 in the proposal).
- **Red** (neither holds) → stop or rethink the premise.

## Data — controlled minimal pairs

Module: `data_gen.py`. Programmatically generate matched pairs that share an
identical context and differ **only** in the question, isolating hop-count while
controlling length and vocabulary.

Example (kinship family):
- 2-hop: "Tom is Mary's father. Mary is Sue's father. Who is Sue's grandfather?" → `Tom`
- 1-hop control: "Tom is Mary's father. Mary is Sue's father. Who is Sue's father?" → `Mary`

Relation families: kinship, transitive ordering ("taller than"), location chains.
Target ~60–100 matched pairs across families. Each item records `{id, family,
hop, prompt, gold_answer}`. Model correctness is logged per item so analysis can
optionally restrict to the subset the model actually solves.

## Method — attention → persistent homology

Module: `topology.py` (the methodological heart; pure and unit-tested).

Per example, run the model with `output_attentions=True`, producing an attention
tensor per layer of shape `[n_heads, n, n]`. For each (layer, head) matrix `A`
(rows = query, cols = key; causal/lower-triangular):

1. **Symmetrize:** `W_ij = max(A_ij, A_ji)`.
2. **Sparsify:** keep top-k edges per node (and/or a weight floor) to keep the
   flag-complex expansion tractable.
3. **Filtration:** vertices enter at filtration 0; each edge enters at
   `f = 1 − W` so that strong attention edges appear earliest. Build the
   flag/clique complex with `gudhi.SimplexTree` followed by `expansion(2)`
   (dimension 2 is sufficient to capture and kill H₁ cycles).
4. **Persistence:** compute persistence; extract the H₁ (dimension-1) diagram.
   Features per (layer, head): β₁ at a small set of fixed thresholds, total H₁
   persistence (Σ death − birth), max H₁ persistence, number of H₁ classes.

## Components

Small, isolated, independently testable units:

- `data_gen.py` — generate and verify matched minimal pairs.
- `attn_extract.py` — load Qwen2.5-0.5B-Instruct, run an example, return
  attentions as numpy; disk-cache by prompt hash.
- `topology.py` — pure function: attention matrix → H₁ feature dict. Primary TDD
  target.
- `run_spike.py` — orchestrate: generate data → extract attentions → compute
  topology per (layer, head) → assemble a results dataframe (parquet).
- `analyze.py` — paired Wilcoxon stats, effect sizes, Benjamini–Hochberg
  correction, layer×head heatmaps, multi-vs-single distribution plots, verdict.
- `tests/` — unit tests.

## Testing strategy

TDD on `topology.py` before trusting it on real attention. Validate the gudhi
pipeline against graphs with known homology:
- 4-cycle (square, unfilled) → β₁ = 1.
- Filled triangle (2-simplex present) → β₁ = 0.
- Two disjoint cycles → β₁ = 2.
- Tree / path graph → β₁ = 0.

Plus sanity tests on `data_gen.py` (pairs share context, differ only in
question; gold answers correct by construction).

## Environment & deliverables

- `uv`-managed project. Dependencies: `transformers`, `gudhi`, `numpy`, `scipy`,
  `pandas`, `scikit-learn`, `matplotlib`, `statsmodels`. `torch` 2.6 already
  installed; runs on MPS.
- Git repo for the work (initialized).
- Deliverables: results dataframe (parquet), plots (β₁ layer×head heatmap,
  multi-vs-single distributions), and a short findings note that evaluates the
  three pre-registered criteria and states the green/yellow/red verdict.

## Cost estimate

~336 attention matrices per example (24 layers × 14 heads) × ~100 examples ≈ 34k
persistence computations, each on a small sparsified graph (milliseconds).
Expected total runtime: minutes on the M3 Pro. No GPU required.

## Risks & mitigations

- **Flag-complex blowup** on dense attention → mitigated by top-k sparsification
  and capping expansion at dimension 2.
- **Model fails the task**, confounding "topology tracks reasoning" with "model
  can't reason" → mitigated by minimal pairs sharing context, logging
  correctness, and optional restriction to solved items.
- **Threshold sensitivity** → mitigated by using persistence (filtration) rather
  than a single threshold.
- **Multiple comparisons** across (layer, head) → mitigated by Benjamini–Hochberg
  correction.
