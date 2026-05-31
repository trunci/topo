# Design: Experiment 3 — Residual-Information Test (is H1 topology redundant?)

**Date:** 2026-05-30
**Status:** Pre-registered (written before computing on real data)
**Parent proposal:** `WRITEUP.md` §7, follow-up direction #1 (residual-information test).
**Predecessors:** spike (`FINDINGS.md`, GREEN — topology tracks reasoning),
Experiment 1 (`FINDINGS_exp1.md`, NULL — DMT pruning), Experiment 2
(`FINDINGS_exp2.md`, RED — topology does not localize induction better than a
trivial attention-distance scalar). In Exp 2, mean attention distance predicted
induction score *better* than H1 persistence (Spearman 0.446 vs 0.114).

## Purpose

Answer one sharp question: **does H1 persistence add ANY predictive power for
induction score beyond first-order attention statistics?**

First-order statistics here = three cheap per-head scalars computed directly from
the attention matrix: mean attention distance, off-diagonal mass, and (NEW)
attention entropy. If H1 persistence has no residual predictive value once these
are controlled for, the honest conclusion is "topological features of attention
are redundant with simple first-order statistics." This is a legitimate negative
result and is to be reported as-is. We do NOT try to make topology look good.

## Method

Per-head data over all 144 GPT-2-small heads (12 layers x 12 heads), computed
from the same repeated-random-sequence forward passes used in Exp 2. Reuse the
Exp 2 run loop verbatim and add one column.

- **Target:** `induction_score` (independent induction-head detector).
- **First-order predictors (baseline set):** `attn_distance`, `offdiag_mass`,
  `attn_entropy` (NEW — mean over query rows of the Shannon entropy, in nats, of
  each causal row distribution).
- **Topological predictor:** `h1_persistence` (H1 total persistence).

Analysis (statsmodels OLS + scipy):

1. **Baseline OLS:** induction_score ~ attn_distance + offdiag_mass +
   attn_entropy (with intercept). Record R^2 (and adjusted R^2).
2. **Full OLS:** baseline predictors PLUS h1_persistence. Record R^2 (and adj R^2).
3. **delta-R^2** = full R^2 - baseline R^2 (the extra variance explained by H1).
4. **Nested-model F-test** comparing baseline vs full (statsmodels
   `anova_lm` / the F-test on the added term). Record F and p (= the p-value on
   the H1 term in the nested comparison; with one added regressor this equals the
   t-test p-value on the H1 coefficient). Record the H1 coefficient and its t-p.
5. **Partial Spearman:** Spearman(h1_persistence, induction_score | first-order
   controls), computed by Spearman-ranking, regressing rank(h1) and
   rank(induction) each on the rank-transformed controls, and correlating the
   residuals. Record rho and p. Also record the raw (uncontrolled)
   Spearman(h1, induction) for reference.

All predictors standardized is not required (R^2 / F-test / partial correlation
are scale-invariant in the relevant senses); we use raw scalars for the OLS and
rank-transformed for the partial Spearman.

## Pre-registered verdict rule (fixed before running)

Let `delta_r2` = full R^2 - baseline R^2, `f_pvalue` = nested-model F-test
p-value for adding h1_persistence, `partial_spearman_p` = p-value of the partial
Spearman.

- **GREEN:** H1 adds significant predictive power beyond first-order stats —
  `delta_r2 >= 0.02` AND `f_pvalue < 0.05`. (i.e. H1 explains a meaningful extra
  >=2 percentage points of induction-score variance, and the added term is
  statistically significant.)
- **RED:** H1 does NOT add predictive power — `delta_r2 < 0.02` OR
  `f_pvalue >= 0.05`. The honest interpretation: H1 persistence is redundant with
  first-order attention statistics for predicting induction.

The 0.02 delta-R^2 floor and the 0.05 alpha are fixed here, before any real-data
computation. The partial Spearman is reported as corroborating evidence but the
verdict is driven by (delta_r2, f_pvalue) as above.

## Components

- `src/residual.py` — NEW. `attention_entropy(A)`: mean over query rows of the
  Shannon entropy (nats) of each (causal, renormalized) row distribution. Pure
  numpy on an attention array; unit-testable.
- `src/run_exp3.py` — NEW. Reuse the Exp 2 GPT-2 CPU loop; produce
  `results/exp3.parquet` with columns layer, head, induction_score,
  h1_persistence, attn_distance, offdiag_mass, attn_entropy (one row per head).
- `src/compute_stats_exp3.py` — NEW. The analysis above ->
  `results/exp3_stats.json` (single source of truth).
- `src/write_findings_exp3.py` — NEW. Emit `FINDINGS_exp3.md` mechanically from
  the JSON. No hand-typed numbers.
- Tests: `tests/test_residual.py` (entropy on known matrices),
  `tests/test_run_exp3.py` (slow smoke: schema), `tests/test_compute_stats_exp3.py`
  (synthetic coupled -> GREEN, synthetic independent -> RED; on-disk JSON == dict).

## Safety & integrity

- GPT-2 small only, CPU, float32, eager attention, `torch.set_num_threads(2)`.
  One model in memory, no parallel heavy jobs. Modest batch (n_seqs<=8,
  seq_len<=30), mirroring Exp 2.
- No hand-typed numbers anywhere. `FINDINGS_exp3.md` is generated from
  `results/exp3_stats.json`; the verdict is a machine-checked field. This guard
  exists because result numbers were fabricated repeatedly earlier in this
  project.

## Scope guardrails (YAGNI)

- One model, one circuit (induction), three first-order baselines + H1. No new
  topological descriptors (that is WRITEUP direction #3, out of scope here).
- Reuse `topology.py`, `induction.py`, `attn_extract.py` as-is.
