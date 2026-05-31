# Design: Experiment 5 — Causal test + cycle inspection for induction

**Date:** 2026-05-31
**Status:** Pre-registered (all design decisions fixed before any real-data compute).
**Parent:** WRITEUP.md "next evolutions". Builds on Exp 3/4, which showed H1 persistence is a
**non-redundant correlate** of the *induction* circuit (model-robust: GREEN in gpt2 AND
distilgpt2; circuit-specific: RED for previous-token / duplicate-token). Everything so far is
**correlational**. This experiment escalates to two harder claims, both restricted to the
circuit where topology earns its keep (induction):

- **Evolution 2 (cycle inspection) — descriptive/mechanistic.** *What* attention edges form
  the H1 cycles in induction heads? Hypothesis: they span the induction gap.
- **Evolution 1 (causal) — functional.** Do those cycle edges *causally matter* for induction
  behavior more than matched control edges?

E2 is run first because the cycle edges it identifies are the treatment set for E1.

## Shared substrate

- Model: **gpt2** (CPU, eager, float32). distilgpt2 optional only if time permits; gpt2 is
  the primary. NO MPS, one model in memory, `torch.set_num_threads(2)`.
- Inputs: repeated-random sequences `[prefix; block S; block S]` via
  `induction.make_repeat_batch` (n_seqs<=8, seq_len S=25, prefix_len=5) — same as Exp 2/3/4.
- "Cycle edges" of a head = the **critical 1-cells** from discrete Morse matching of that
  head's symmetrized+sparsified attention graph. New helper `morse.critical_cycle_edges(A,
  top_k=8, ...)` reusing the existing `_morse_match` (critical cells with len==2). These are
  the project's canonical "topologically essential cycle generators".
- "Induction heads" = the top-K heads by `induction.induction_score` (K=10, fixed), pooled
  across the batch (mean score per head), computed once and reused by both E2 and E1.

---

## Evolution 2 — Cycle inspection (descriptive)

For each induction head, on each sequence, collect its critical cycle edges `{i,j}` and their
**gap** `|i-j|`. Also collect cycle-edge gaps for a comparison pool of **non-induction heads**
(the bottom-K by induction score, K=10).

Pre-registered hypothesis & test:
- The induction copy distance is `S` (= seq_len): an induction head attends from 2nd-copy
  position `i` to ~`i-S+1`. If cycles encode that relation, induction-head cycle edges should
  concentrate near gap `S` (within +/-2), far more than non-induction-head cycle edges.
- **Metric:** fraction of cycle edges with `|gap - S| <= 2`, for induction vs non-induction
  heads. **Test:** Mann-Whitney on per-head fractions (induction > non-induction), p<0.05.
- Verdict E2: **GREEN** if induction-head cycle edges concentrate at the induction gap
  significantly more than non-induction heads; **RED** otherwise. Also report the raw gap
  histograms (counts per gap bucket) in the JSON so the pattern is inspectable, whatever the
  test says.

This is descriptive: a RED here just means "cycles are not obviously the copy edges", which is
still informative and does not invalidate E1.

---

## Evolution 1 — Causal ablation (functional)

**Readout (behavioral, NOT the head's own score — avoids circularity):** the model's mean
next-token cross-entropy loss on **second-copy positions** (positions `prefix_len + S .. n-1`,
where induction is what lets the model predict the repeat). Lower loss = working induction.
Compute via `MaskedModel.logits(enc, biases)` -> per-position CE on second-copy positions.

**Treatment vs controls (all mask an EQUAL number of edges, per head, then renormalize via the
existing additive-bias masking; readout = increase in second-copy loss vs the unmasked run):**
- **treatment = cycle**: ablate the head's critical cycle edges (count = m per head).
- **control-random**: ablate m random edges from that head's candidate edges (seeded).
- **control-magnitude**: ablate the m highest-attention-weight edges (matches/【exceeds】 the
  treatment on magnitude, so a positive treatment-vs-magnitude result cannot be "you just
  removed strong edges").

Masking mechanism: reuse `pruned_forward`. To ablate edge-set E in head h while keeping
everything else, pass `keepsets[(layer,h)] = all_candidate_edges(h) - E`; heads not in the
dict keep all (zero bias). Ablate **only within the K induction heads**, all of them together,
per sequence. The diagonal is always kept by `keepsets_to_bias` (no all-masked rows).

Per sequence we get four second-copy losses: unmasked, cycle, random, magnitude. Define the
**damage** of a condition = loss(condition) - loss(unmasked).

Pre-registered verdict E1 (paired across the n_seqs sequences; Wilcoxon signed-rank):
- **GREEN (causal):** damage(cycle) > damage(magnitude) AND damage(cycle) > damage(random),
  both with Wilcoxon p<0.05 and positive median difference. I.e. ablating the *topological*
  cycle edges hurts induction behavior MORE than removing the same number of highest-magnitude
  edges — topology selects functionally-special edges that magnitude does not.
- **PARTIAL:** damage(cycle) > damage(random) (p<0.05) but NOT reliably > damage(magnitude).
  (Cycle edges matter, but not beyond what magnitude already captures.)
- **RED:** cycle ablation does not significantly exceed random. (No causal signal.)

A RED/PARTIAL is a fully acceptable, honest outcome — and it would sharpen the Exp 1 NULL
(topology not better for pruning) by showing whether it's also not better for the *specific*
induction behavior. Report whatever the data says.

Sanity check to log (not a verdict): damage(cycle) and damage(magnitude) should both be > 0
(ablating real attention should hurt); if they are ~0, the ablation/readout is too weak and
that must be flagged rather than interpreted.

---

## Components (heavy reuse; small new code)

- `src/morse.py` — ADD `critical_cycle_edges(A, top_k=8, weight_floor=0.0, symmetrized=False)`
  returning `set[frozenset({i,j})]` (critical 1-cells). Reuse `_morse_match`. TDD: on a
  4-cycle it returns exactly one critical edge; on a filled triangle, none; on a path, none.
  Do NOT change `morse_keep`'s behavior (existing tests must stay green).
- `src/induction.py` — reuse; ADD nothing unless needed (`second_copy_loss` lives in run_exp5).
- `src/run_exp5.py` — NEW. Build the batch; pick top-K / bottom-K induction heads; (E2) collect
  cycle-edge gaps per head; (E1) for each sequence compute unmasked + 3 ablation second-copy
  losses over the induction-head set. Write a per-sequence/per-head tidy parquet
  `results/exp5.parquet` (two logical tables or two parquets: `exp5_cycles.parquet` for E2
  gaps, `exp5_ablation.parquet` for E1 losses). CPU, one model, flushed progress.
- `src/compute_stats_exp5.py` — NEW. E2 Mann-Whitney + gap histograms; E1 paired Wilcoxons +
  median damages + the two sub-verdicts + an overall summary -> `results/exp5_stats.json`
  (single source of truth). TDD on synthetic data (a constructed parquet where cycle damage >
  magnitude -> GREEN; where equal -> RED; on-disk JSON == returned dict).
- `src/write_findings_exp5.py` — NEW. Emit `FINDINGS_exp5.md` mechanically from the JSON
  (E2 verdict + gap table; E1 verdict + damage table). No hand-typed numbers.

## Integrity & safety (HARD)

- CPU only; one model; `torch.set_num_threads(2)`; n_seqs<=8, S=25. No MPS, no parallel heavy
  jobs. (Two machine crashes earlier came from MPS/large/parallel runs.)
- **No hand-typed numbers** anywhere: every value computed into `results/exp5_stats.json`,
  emitted into FINDINGS mechanically, regeneration verified byte-identical before commit.
  Commit messages state verdicts qualitatively and point to the JSON.
- Pre-registered verdict rules above are fixed; sub-verdicts are machine-checked JSON fields.
- TDD, small commits, co-author trailer. `uv run`; prefix model commands with
  `set -a; [ -f .env ] && . ./.env; set +a; unset VIRTUAL_ENV;`.

## Deliverables

`results/exp5_*.parquet`, `results/exp5_stats.json`, `FINDINGS_exp5.md` (E2 + E1 verdicts,
machine-generated), all committed; fast tests passing. An honest final summary giving both
sub-verdicts and the damage/gap numbers as they appear in the JSON, plus the sanity check.
