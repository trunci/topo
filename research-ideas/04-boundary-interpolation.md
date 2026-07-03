# 4 — Boundary interpolation: which property of real context kills the signal?

## Question

The audit's sharpest unresolved fact: attention-topology failure prediction is
near-perfect on template-generated reasoning at the capability ceiling, and
dead on naturalistic QA at matched accuracy — surviving neither geometry
control (Exp 15) nor feature replacement (Exp 16). "Template → naturalistic"
is a confounded jump: surface-form diversity, semantic openness, answer-form
variety, and context heterogeneity all change at once. Interpolate the axis
and find where the signal dies.

## Design

A graded ladder holding hops (4–5) and accuracy band (~0.5) fixed:

1. our templates (known GREEN);
2. templates with lexical paraphrase (surface diversity only);
3. LLM-rewritten narratives of the same underlying relation graphs
   (naturalistic prose, controlled semantics);
4. LLM-generated multi-hop questions over synthetic entities (open form,
   closed world);
5. HotpotQA gold-only (known RED).

Same 0.5B/1.5B models, same features, same probes; one pre-registered verdict
per rung ("signal alive" = topology beyond confidence at bootstrap p < .05).
The deliverable is a *decay curve*: signal vs distance from template.

## Why it matters (attention/context SOTA)

This is a context-property experiment at heart: it asks which statistical
property of the *input distribution* (regularity? entropy? answer-form
predictability?) makes attention organization informative about model state.
That question generalizes past topology — any attention-based detector
(probes, entropy monitors, hallucination detectors) faces the same
template-to-wild gap, and nobody has mapped it with controls. Also directly
improves the paper's story if it precedes journal submission: "the boundary
is at rung k, and the property that changes there is X" beats "the boundary
is naturalism."

## Cost

Data generation is LLM-cheap; runs are 0.5B/1.5B scale — laptop/MPS viable,
or **~$5** of L4 time for all rungs × 2 models × 3 seeds.
