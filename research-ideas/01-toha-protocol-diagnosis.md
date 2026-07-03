# 1 — TOHA protocol diagnosis (Exp 17)

## Question

Exp 16: TOHA's MTop-Div metric is at chance on our items (0.560/0.504) while
TOHA reports 0.71 ± 0.08 on the same model + benchmark. Which protocol axis
carries the difference? The paper currently names four candidates and says
"adjudicating them is the natural next experiment" — this is that experiment.

## Axes to ablate (one at a time, pre-registered)

1. **Labels.** We use substring-match correctness; TOHA uses annotated
   hallucination labels. Wrong-but-grounded and right-by-luck answers land in
   different classes under the two schemes. Cheap test: LLM-judge (or hand)
   annotation of our existing 200×2 generations into
   grounded/hallucinated, re-score the *already computed* features. **$0 GPU.**
2. **Item mix.** We use bridge questions only; TOHA samples all HotpotQA types.
   Re-run on a type-stratified 200-item sample.
3. **Decoding.** We use greedy; TOHA scores sampled generations (hallucinations
   are rarer/different under greedy). Re-run with their sampling config.
4. **Selection budget.** Our head selection is fold-internal (≤160 items);
   TOHA optimizes head count on a labeled probe set. Give MTop-Div the same
   budget on a disjoint probe set and score on held-out items.

Run axis 1 first — it is free (features are persisted per head in
`results/exp16_*_features.parquet`) and the most likely culprit: label noise
attenuates AUC multiplicatively, and substring matching is noisy in both
directions.

## Verdicts

Pre-register per axis: "axis explains the gap" iff MTop-Div AUC under the
changed protocol reaches ≥ 0.63 (TOHA's band floor). If no single axis does,
report the interaction sweep as the finding.

## Why it matters (attention/context SOTA)

Attention-map hallucination detectors are an active line (TOHA, van Dijk's KL
probe, attention-sink work). A controlled account of *when their evaluation
protocols manufacture or destroy signal* is a service to that whole line — and
it is the one remaining hole in our paper's story. Publication-critical:
reviewers of the audit paper will ask exactly this question.

## Cost

Axis 1: $0 (offline). Axes 2–4: one A100 session each, ~15 min each at ~35
items/min → **~$5 total**. All reuse the exp16 harness unchanged.
