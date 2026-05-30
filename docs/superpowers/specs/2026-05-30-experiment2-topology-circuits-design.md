# Design: Experiment 2 — Topology ↔ Known Circuits (Induction Heads)

**Date:** 2026-05-30
**Status:** Approved (pending spec review)
**Parent proposal:** `research.md` — claim #2 (mechanistic correspondence).
**Predecessors:** spike (`FINDINGS.md`, GREEN — topology tracks reasoning) and
Experiment 1 (`FINDINGS_exp1.md`, NULL — DMT pruning does not beat baselines). The
consolidated story so far (`WRITEUP.md`): attention topology is **diagnostic** of
reasoning but **not prescriptive** for pruning. This experiment tests whether the
diagnostic is *mechanistically meaningful* — whether topologically distinctive heads
coincide with an independently-known circuit.

## Purpose

Answer one question: **do the attention heads with distinctive H₁ topology coincide with
induction heads** — a circuit identified independently of any topology?

Induction heads are the cleanest documented circuit and have a standard, automatable
detector. They give us a "ground truth" set of heads that owes nothing to our topology
pipeline, so a correspondence would be real evidence that topology predicts circuits.

## Pre-registered criteria (decided before running)

Across all 144 GPT-2-small heads, using each head's induction score (independent) and its
H₁ persistence on the same repeated-sequence inputs:

1. **Discrimination:** Spearman(induction score, H₁ persistence) > 0 and p < 0.05.
2. **Separation:** the top-k induction heads have significantly higher H₁ persistence than
   the remaining heads (Mann–Whitney U, p < 0.05). k is fixed before running (k = 10).
3. **Beats trivial baseline:** H₁ persistence predicts induction at least as well as a
   cheap attention-distance scalar (mean |i−j| weighted by attention). Operationalized as:
   |Spearman(persistence, induction)| ≥ |Spearman(distance, induction)| − 0.05 (i.e.
   topology is not clearly dominated by the trivial metric).

**Verdict rule:**
- **GREEN:** criteria 1 AND 2 hold, AND topology is not dominated by the distance baseline
  (criterion 3).
- **YELLOW:** criteria 1 AND 2 hold but the distance baseline does it as well or better
  (topology correlates but adds nothing over a trivial metric).
- **RED:** criterion 1 or 2 fails (no correspondence).

All thresholds (p < 0.05, k = 10, the −0.05 slack) are fixed here, before running.

## Method

1. **Model:** GPT-2 small (`gpt2`, 124M, 12 layers × 12 heads = 144 heads), loaded via the
   existing `attn_extract.load_named` with eager attention + float32, on **CPU**.
2. **Induction score (independent ground truth):** the standard repeated-random-sequence
   detector. Build batches of token sequences of the form `[prefix ; rand(S) ; rand(S)]`
   where the two `rand(S)` halves are identical random tokens. A head's induction score is
   the mean attention weight from each position `i` in the second copy to position
   `i − S + 1` (the token that immediately followed the matching token in the first copy),
   averaged over positions and sequences. Computed purely from attention, with no topology.
3. **Topology signal:** on the *same* repeated sequences, run the spike pipeline
   (`topology.h1_features`, unchanged) per head and take H₁ **total persistence** as the
   primary signal (mirrors the spike's primary metric).
4. **Distance baseline:** per head, mean attention distance `Σ_ij A_ij · |i − j|` and
   off-diagonal mass `Σ_{i≠j} A_ij`, both cheap scalars, used as the trivial competitor in
   criterion 3.

All four quantities are computed per head from the same forward passes.

## Components

Small units, heavy reuse:

- `src/induction.py` — NEW. `make_repeat_batch(...)` (repeated-random token batches);
  `induction_score(attn, seq_len, prefix_len)` (per-head induction score from an attention
  array); `attention_distance(attn)` and `offdiag_mass(attn)` (baseline scalars). Pure
  numpy/torch on attention arrays; no model dependency, so unit-testable.
- Reuse `src/topology.py` (`h1_features`) unchanged.
- Reuse `src/attn_extract.py` (`load_named`) — GPT-2 loads through the same path.
- `src/run_exp2.py` — NEW. Load GPT-2; run the repeat batch with `output_attentions`;
  per (layer, head) compute induction score, H₁ persistence, distance, off-diagonal mass →
  `results/exp2.parquet` (one row per head).
- `src/compute_stats_exp2.py` — NEW. Spearman (persistence vs induction), Mann–Whitney
  (top-k vs rest), distance-baseline Spearman, verdict → `results/exp2_stats.json`
  (single source of truth).
- `src/write_findings_exp2.py` — NEW. Emit `FINDINGS_exp2.md` mechanically from the JSON.
- Tests: `tests/test_induction.py` — induction_score ≈ 1 on a hand-built attention matrix
  with a perfect induction pattern, ≈ 0 on uniform attention; distance/off-diagonal scalars
  on known matrices.

## Testing strategy

TDD on `induction.py` (pure functions, fast, no model):
- A synthetic attention array where the only mass is exactly on the induction offset
  (position `i` → `i − S + 1`) yields induction_score ≈ 1.0.
- A uniform-attention array yields a low induction score (≈ chance).
- `attention_distance` of a diagonal-only matrix is 0; of a matrix with all mass at
  distance d is d.
- `make_repeat_batch` produces sequences whose two halves are token-identical.

`run_exp2.py` gets a light smoke check (tiny batch, assert parquet schema: one row per head,
columns `layer, head, induction_score, h1_persistence, attn_distance, offdiag_mass`).
`compute_stats_exp2.py` is unit-tested on a synthetic parquet where persistence is
constructed to track induction, asserting the JSON contains the Spearman/Mann–Whitney
fields and the correct GREEN verdict; the test also asserts the on-disk JSON equals the
returned dict.

## Safety & integrity (lessons from this session)

- **GPT-2 small only, CPU, one model in memory** — far lighter than the Qwen float32 runs
  that exhausted memory and restarted the machine. No parallel heavy work during the run.
  Stream per-head/per-batch progress so the run is observable.
- **No hand-typed numbers.** `FINDINGS_exp2.md` is generated by `write_findings_exp2.py`
  from `results/exp2_stats.json`; every value is read from disk. This guard exists because
  result numbers were fabricated repeatedly earlier in this project.
- Pre-registered criteria above are fixed before running; the verdict is a machine-checked
  field in the JSON.

## Scope guardrails (YAGNI)

- **Induction heads only** — the single cleanest circuit. No IOI, no path-patching, no
  Qwen cross-check; those are follow-ons only if this lands.
- Single model. CPU. A few hundred small persistence computations — minutes.
- Reuse `topology.py` and `attn_extract.py` as-is; do not modify spike/Exp1 code.

## Deliverables

`results/exp2.parquet` (per-head: induction score, H₁ persistence, distance, off-diagonal
mass), `results/exp2_stats.json` (Spearman, Mann–Whitney, baseline comparison, verdict), a
scatter plot (induction score vs H₁ persistence, induction heads highlighted), and
`FINDINGS_exp2.md` generated from the JSON with the GREEN/YELLOW/RED verdict.
