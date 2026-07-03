# Experiment 19 — Boundary interpolation (pre-registration)

**Date pre-registered: 2026-07-03, before any Exp 19 data generation or model
pass.** Research-ideas/04: the audit found attention-topology failure
prediction near-perfect on template reasoning at the capability ceiling
(Exp 13, GREEN) and dead on naturalistic QA (Exps 14–16, RED). That jump
confounds surface-form diversity, semantic openness, and task provenance.
Exp 19 interpolates the axis with a five-rung ladder, everything else held
fixed, and reports where the signal dies. Results are appended below the
marker by `compute_stats_exp19`; this section is immutable.

## The ladder (hops fixed at 4–5 where applicable; single seed 0)

| rung | contexts | questions | source |
|---|---|---|---|
| R1 | canonical templates | canonical | exp13 generator, verbatim |
| R2 | handcrafted paraphrase bank (~6 surface forms per relation sentence, sampled per sentence) | 3-variant bank | programmatic |
| R3 | LLM narrative rewrite of R1 facts (same names, same facts, flowing prose) | canonical | Mistral-7B rewriter |
| R4 | LLM narrative rewrite | LLM-rephrased question | Mistral-7B rewriter |
| R5 | HotpotQA gold-only contexts (n = 200, seed 0 — the Exp 15 items) | HotpotQA | natural |

R1–R4 use the same 120 items (2 families × 30 casts × hops 4,5), same entity
names, same underlying relation chains — only surface form changes. R2 keeps
sentence order fixed (order is a separate axis, not varied here). Rewrites
(R3/R4) are validated programmatically: every chain name present, the gold
answer's relation keyword absent from the context, gold not leaked into the
question; items failing validation after 3 attempts are dropped and attrition
reported. Rewriter = Mistral-7B-Instruct-v0.3 (different family from the
subject models; rewritten items persisted to results/ for reproducibility).

## Subject models, features, probes

Qwen2.5-0.5B-Instruct (primary — the Exp 13 GREEN model) and
Qwen2.5-1.5B-Instruct (replication) on all rungs. Features: the exp13/14/15
pooled set (H1 mean/max persistence, frac-nontrivial; attention distance,
off-diagonal mass, entropy; confidence margin), computed with the exp15 fast
path; equivalence with the exp13 slow path is verified on 5 R1 items before
the run (assert within 1e-6). Probes: stratified 5-fold logistic, exp13
machinery; item-level bootstrap (B = 10,000) per exp16 machinery.

## Pre-registered per-rung verdict ("signal alive")

For each (rung, model): ALIVE iff topology-only AUC bootstrap CI95 lower bound
> 0.5 AND topology adds beyond confidence (paired bootstrap one-sided
p < 0.05, ΔAUC > 0). DEAD iff neither; WEAK if it beats chance but adds
nothing beyond confidence. UNDERPOWERED if accuracy outside [0.15, 0.85].
Mechanism scalars recorded per rung: r(topo_mean_persist, ctrl_attn_entropy)
and ΔAUC of topology beyond first-order controls.

## Expectations (stated in advance)

R1 ALIVE (Exp 13 replication), R5 DEAD or UNDERPOWERED (Exp 15 analogue on a
smaller model). The finding is where R2–R4 land: R2 DEAD would mean mere
lexical variety kills the signal (fragile template artifact); R2 ALIVE but
R3 DEAD would locate the boundary at narrative/discourse form; R4 DEAD with
R3 ALIVE would locate it at question-form openness. A monotone decay with no
sharp step is also a legitimate outcome.

## Deliverable

Per-rung verdict table + a decay-curve figure (AUC of topology vs confidence
across rungs, both models) suitable for the paper's §5.3.
