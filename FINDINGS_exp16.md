# Experiment 16 — Pre-registration (written before the run)

**Date pre-registered: 2026-07-02. No exp16 model forward pass has been run at
the time of writing.** Results will be appended below this line by
`compute_stats_exp16` after the run; this section is immutable.

## Question

Exps 14–15 (RED) showed our hand-specified pooled/residualized/head-selected H1
features are at chance on HotpotQA/Mistral-7B, while TOHA (Bazarova et al.,
ACL 2026) reports AUROC 0.71 ± 0.08 in that setting. The paper currently
*conjectures* the gap is feature construction (generation-time topological
divergence vs prompt-encoding persistence). Exp 16 tests that conjecture:
TOHA's MTop-Div features, our items, our probes.

## Design

- Model: `mistralai/Mistral-7B-Instruct-v0.3` (same as Exps 14–15, same as TOHA).
- Items: the same 200 HotpotQA bridge questions (seed 0) in both geometries:
  `distractor` (Exp 14) and `gold` (Exp 15).
- Feature: per-head MTop-Div — total MSF weight attaching answer tokens to the
  prompt in the (1 − attention) metric, normalized by answer length — computed
  from a teacher-forced forward over prompt + greedy answer
  (`src/toha_features.py`, unit-tested against scipy MST).
- Controls: confidence margin (as in Exps 13–15) and response-row attention
  entropy (the first-order twin of MTop-Div).
- Probes: 5-fold stratified CV logistic probes, identical machinery to
  Exps 13–15; ΔAUC assessed by one-sided fold Wilcoxon AND item-level paired
  bootstrap (B = 10,000) on out-of-fold predictions.

## Featurizations to be scored (per condition)

1. `pooled` — [mtd_mean, mtd_max, mtd_std] logistic probe.
2. `head_selected` — fold-internal TOHA protocol: within each training fold,
   rank heads by single-head |AUC − 0.5| on training items, take the top 10,
   z-score each on training stats, sign-align so higher = predicted failure,
   average into one scalar; test-fold AUC of that scalar. No test-fold
   information enters selection.
3. `head_selected_beyond_confidence` — confidence + the head-selected scalar
   vs confidence alone.
4. `head_selected_beyond_controls` — response-entropy controls + scalar vs
   controls alone (is MTop-Div more than an entropy meter here?).

## Pre-registered verdict rules (per condition)

- **GREEN** iff head_selected AUC beats chance (item-bootstrap 95% CI lower
  bound > 0.5) AND adds beyond confidence (bootstrap one-sided p < 0.05 and
  ΔAUC median > 0).
- **PARTIAL** iff head_selected beats chance by the same CI rule but does not
  add beyond confidence.
- **RED** iff head_selected does not beat chance.
- **UNDERPOWERED** if base rate outside [0.15, 0.85].

Secondary, non-verdict reproduction check: does any featurization reach
TOHA's reported band (AUROC ≥ 0.63, i.e. 0.71 − 0.08)?

## Interpretation map (written in advance)

- GREEN in either condition → the Exp 14/15 boundary is feature construction,
  as conjectured; the paper's §6 claim becomes a result.
- PARTIAL → engineered topology sees failure but is redundant with confidence
  on real QA; TOHA's advantage over confidence does not replicate in our
  bridge-only n=200 slice.
- RED in both → the boundary is NOT (this) feature construction; the §6
  conjecture is wrong and must be rewritten — naturalistic-task semantics
  defeat MTop-Div in our setting too, and the discrepancy with TOHA's 0.71
  needs diagnosis (item mix, answer sampling, head-selection supervision
  budget) rather than assertion.
