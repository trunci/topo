# Experiment 18 — Grounding heads, Phase B (pre-registration)

**Date pre-registered: 2026-07-03. Written before any Exp 18 model forward
pass.** Confirmatory phase of the grounding-heads program, promoted from Exp 17
Phase A under the structure-agnostic framing (per-head attention statistics;
topology carried no signal beyond its entropy twin). Results are appended below
the marker by `compute_stats_exp18`; this section is immutable.

## New data to collect (one A100 session)

Same 200 HotpotQA bridge items (seed 0) as Exps 14–17. Model:
Mistral-7B-Instruct-v0.3, greedy, identical pipeline.

1. **NoContext condition** — question only, no paragraphs. Greedy answer,
   substring-match correctness, confidence margin. Defines the
   parametric-memory label: `knew_anyway` = NoContext answer correct.
2. **Span-resolved distractor condition** — the Exp 14/16 10-paragraph
   geometry, re-run capturing, per head, the generated answer's attention
   onto each context paragraph's token span: mean mass and max weight onto
   gold spans (2 paragraphs), distractor spans (8), and the question span.
   Per-head whole-prompt response entropy is retained for comparability.

## Labels

- `knew_anyway` (parametric memory): NoContext correct.
- `provenance` (primary, defined on the distractor-condition-CORRECT subset):
  `parametric` = correct AND knew_anyway; `context_derived` = correct AND NOT
  knew_anyway. This is the cleanest mechanical provenance contrast available
  without annotation.
- `stays_correct`: from Exps 16/17 — gold-condition-correct items that remain
  correct under distractors.
- `sway`: Exp 17's loose divergence label (reused for the secondary
  replication).

## Pre-registered hypotheses and verdicts

**H-B1 (provenance, primary).** On the distractor-correct subset, span-resolved
per-head features (fold-internal top-10 head selection, exp16 protocol)
classify parametric vs context-derived answers.
- GREEN iff bootstrap AUC CI95 lower bound > 0.5 AND beats the confidence
  baseline AND beats the pooled total-context-attention-mass baseline (both
  ΔAUC bootstrap one-sided p < 0.05).
- PARTIAL iff CI clears 0.5 but a baseline matches it.
- RED iff CI includes 0.5.
- UNDERPOWERED if subset base rate outside [0.15, 0.85] or subset n < 60.

**H-B2 (robustness).** Gold-vs-distractor attachment (per-head gold-mass
fraction, head-selected + pooled ratio) predicts `stays_correct`.
- Same verdict rules; baselines: confidence, whole-prompt response entropy.
- Known risk: base rate was 0.808 in Phase A — borderline; report regardless.

**Secondary (no verdicts).** S1: sway prediction replication with span
features (Phase A found 0.800 with whole-prompt features). S2: depth profile
of gold-attached heads on context-derived answers.

## Interpretation map (written in advance)

- H-B1 GREEN → attention span-attachment carries provenance information beyond
  trivial mass: "grounding heads" exist in the useful sense. Phase C (paper
  push) is justified: scale to more items, add annotated faithfulness labels.
- H-B1 PARTIAL with mass baseline matching → provenance is readable from
  attention but trivially (total mass suffices); still useful applied result,
  not a heads result.
- H-B1 RED → per-head attachment does not carry provenance; the Phase A sway
  signal was about answer instability, not grounding. Program pivots or stops.
- H-B2 any → sharpens or bounds the applied RAG claim from Phase A's A3.
