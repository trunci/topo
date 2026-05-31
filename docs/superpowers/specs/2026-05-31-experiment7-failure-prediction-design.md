# Experiment 7 — Failure prediction from attention topology

**Date:** 2026-05-31
**Status:** pre-registered (design approved before any results)

## Motivation

The Spike (Result A) showed attention-graph H1 *tracks* reasoning hop-count — but only on
**attempted** reasoning (attention over the prompt), never gated on whether the model got
the answer right. This is the Spike's most direct payoff: if per-example topological
features predict *reasoning errors*, topology becomes a usable **diagnostic instrument**
("this attention graph looks like the model is about to fail"), not just a correlate.

The non-trivial bar (chosen): topology must predict failure **beyond the model's own
confidence**. A probe that merely rediscovers "low answer-logit-gap -> likely wrong" is
uninteresting — it would just be re-deriving calibration. The scientific claim is that the
*shape* of attention carries failure-relevant information that the model's output
confidence does not.

## Question

Do per-example H1 topological features predict whether Qwen2.5-0.5B-Instruct answers a
multi-hop reasoning item correctly, **over and above** the model's own confidence
(answer-token logit margin)?

## Design

- **Model:** Qwen2.5-0.5B-Instruct (same as the Spike). CPU-only, float32, eager,
  `torch.set_num_threads(2)`, one model in memory, gc per example. (The earlier crashes
  were MPS/parallel; the gentle CPU path has been stable since Exp 1.)
- **Items:** mixed-difficulty kinship + ordering, hops {1, 2, 3}, to get a natural spread
  of correct/incorrect (1-hop mostly right, 3-hop mostly wrong). Requires extending
  `data_gen.py` with 3-hop builders (4-name chains: great-grandfather / tallest-of-4).
  `n_per_family` chosen so total items >= ~150 after pooling hops.
- **Correctness label:** `is_correct` (already in `attn_extract.py`) — does the model's
  generated answer contain the gold token. Binary y per item.

### Features (per item)

For each item, run one forward pass with `output_attentions=True`, then:
- **Topology features (X_topo):** aggregate H1 features over (layer, head) — e.g. mean and
  max `total_persistence`, fraction of heads with non-trivial H1, and the same restricted
  to the Spike's top-discriminating heads (L8H2, L16H2, ...). Reuses `h1_features`.
- **Confidence feature (X_conf):** the model's answer-token **logit margin** (top-1 minus
  top-2 logit at the answer position) — a scalar proxy for the model's own confidence.
- **First-order controls (X_ctrl):** attn_distance, offdiag_mass, attn_entropy (so we also
  show topology adds beyond cheap attention scalars, mirroring Exp 3).

### Models compared (nested, cross-validated)

Logistic regression predicting `is_correct`, stratified k-fold CV (k=5), scored by ROC-AUC:
- **M0 baseline:** confidence only (X_conf).
- **M1 full:** confidence + topology (X_conf + X_topo).
- (reported) **M_topo:** topology only; **M_ctrl:** confidence + first-order controls.

### Statistics (`src/compute_stats_exp7.py` -> `results/exp7_stats.json`)

- Cross-validated AUC for M0, M1, M_topo, M_ctrl (mean +/- std over folds).
- **delta-AUC = AUC(M1) - AUC(M0)** with a paired test across folds (Wilcoxon) — does
  adding topology to confidence help?
- Likelihood-ratio test (M0 vs M1) on the full data as a secondary check.
- Base rate (majority-class accuracy) and overall model accuracy, reported for honesty.

### Verdict (pre-registered)

- **GREEN** — M_topo AUC > 0.5 (CV, beats chance) **AND** delta-AUC(M1 - M0) > 0 with
  paired p < 0.05 (topology adds beyond confidence).
- **PARTIAL** — M_topo beats chance but does not add beyond confidence (topology predicts
  failure, but only what confidence already knows).
- **RED** — M_topo does not beat chance.
- **UNDERPOWERED (reported, not a pass)** — if correctness has too little variance
  (base rate > 0.85 or < 0.15), flag it: the probe can't be evaluated. We then adjust the
  hop mix and re-run (mix is a design knob, not a result).

Also report the **2-hop-only slice** verdict separately (matches the Spike's original
regime), per request.

## Outputs

- `results/exp7_features.parquet` — one row per item: y (is_correct), all features, hop,
  family. Single source of truth.
- `results/exp7_stats.json` — computed stats + verdict.
- `FINDINGS_exp7.md` — generated mechanically by `src/write_findings_exp7.py` (no
  hand-typed numbers; the project rule).
- Fold into `WRITEUP.md` as Result H once results land.

## Honest caveats (pre-committed)

- Small synthetic probe (Qwen-0.5B, ~150 items, single seed). Direction, not magnitude.
- "Confidence" is a logit-margin proxy, not a calibrated probability; we report which
  confidence definition was used.
- Failure on synthetic kinship/ordering is narrow; positive result motivates, does not
  prove, general failure prediction.
- If the base rate is extreme the probe is uninformative — reported as UNDERPOWERED, mix
  retuned, never silently passed.
