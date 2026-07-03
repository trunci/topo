# 6 — Directed, generation-time, length-invariant topology

## Question

Every topology failure in the audit traces to two structural sins of the
pipeline: max-symmetrization (destroys attention's directionality — who
attends to whom is causal information) and extensive statistics (H1 totals
scale with sequence length; ρ = 0.947 with length on real text). Both are
fixable *by construction*: path homology / Dowker filtrations operate on
directed graphs natively, and per-node-normalized or rank-based filtrations
are length-invariant by design. Does a methodologically clean topology see
what the broken one cannot?

## Design

- Implement directed H1 (path homology on the causal attention digraph;
  MTop-Div-style attachment already generalizes) + rank-filtration variant
  (attention percentile per row rather than raw weight → length-invariant).
- Re-score the audit's decided battlegrounds, cheapest first:
  1. induction non-redundancy (Exp 2–4 data, $0 — attention matrices
     recomputable on CPU for GPT-2);
  2. synthetic failure prediction (Exp 13 setting, laptop-scale);
  3. real QA (Exp 14/15/16 items, one A100 session) — the one that counts.
- Pre-registered: each battleground keeps its original verdict rule; the
  question is whether any RED flips.

## Honest prior

The Exp 16 result lowered this idea's odds: generation-time attachment with a
*divergence* construction already failed on real QA, so "the metric was the
problem" has taken a hit. The live hypothesis is narrower: directionality +
length-invariance jointly matter for the *descriptive* claims (§4) and the
synthetic regime, and might sharpen the 1.5B entropy-orthogonal signal into
something interpretable.

## Why it matters (attention/context SOTA)

Less applied than ideas 2–3; the audience is attention-structure methods.
Its best outcome is a tools contribution: a topology-for-attention library
whose failure modes are engineered out, released with the audit as its
evaluation standard.

## Cost

Implementation-heavy (path homology is real math-engineering, ~3–4 days).
Compute light: mostly CPU; **~$5** GPU for the real-QA pass.
