# Experiment 10 — Topological attribution on IOI (a fair fight for topology)

**Date:** 2026-05-31
**Status:** pre-registered (design approved before results)
**Lineage:** Exp 9 (Result K) found attention magnitude beats topology at attributing the
**induction** circuit — but induction is the *worst case* for topology: the magnitude
baseline trivially wins because induction heads place ~77% of row mass on a single copy
edge. Exp 10 moves to **IOI / name-mover heads**, where an exploratory probe (verified)
shows attention is meaningfully more diffuse: median max-edge fraction 0.54 (vs 0.77),
entropy 1.20 (vs 0.83), top1/top2 ratio 2.2 (vs 6.8). The magnitude baseline is handicapped
here — a fair fight, not a rigged one.

## Honest prior

This is a **genuine test, not a victory lap**. IOI is *more* diffuse, not *very* diffuse —
the top edge still holds ~54% of the mass, so magnitude remains a strong baseline. We expect
a coin-flip on whether topology clears it. Pre-registering so the outcome is credible either
way.

## Question

On the IOI task in GPT-2 small, does a per-edge **topological saliency** (cycle
participation / sheaf discord) localize the **ground-truth name-mover edge** (END query →
IO-name token) better than the **attention-magnitude** baseline, and are the flagged edges
**causally** necessary for the IO prediction?

## Substrate & ground truth

- **Model:** GPT-2 small (the model IOI was characterized on). CPU, float32, eager,
  threads=2, one model, gc per item.
- **Prompts:** the canonical IOI template, `"When {A} and {B} went to the {place}, {B} gave
  a {obj} to"` → correct next token is ` {A}` (the indirect object, IO). A and B are
  single-token first names; place/obj from fixed lists. N≈48 prompts, seeded.
- **Ground-truth edge:** for the END position (last token), the name-mover target is the
  **IO-name token position** (first occurrence of A). The directed edge (END → IO_pos) is
  the ground-truth circuit edge — analytically known, like the induction copy edge in Exp 9.
- **Name-mover heads:** top-K heads by attention(END → IO_pos), averaged over prompts (the
  probe recovered the canonical 9.9/10.7/9.6/10.0… — sanity check we re-assert).
- Everything is **within-example, per-edge** → no sequence-length confound (the Exp 8 trap).

## Part A — attribution

For each name-mover head and prompt, score every directed causal edge into END
(END → k, k<END) by each saliency; AUC + precision@k at recovering the ground-truth IO edge.
- Saliencies: `magnitude` = A[END,k] (baseline); `cycle_participation`; `sheaf_discord`
  (per-edge residual-stream sheaf, reuse Exp 9 `edge_discord`).
- Metric: ROC-AUC over (head × prompt), and p@k (k=1, single ground-truth edge).

**Verdict (Part A):** GREEN if a topological saliency beats chance (AUC>0.5, Wilcoxon
p<α) AND beats magnitude (paired Wilcoxon p<α); PARTIAL if beats chance only; RED otherwise.

## Part B — causal validation

Reuse Exp 6 / Exp 9 directed edge-exact ablation (`ablation_to_bias`) over the name-mover
heads. Readout: **IO logit margin** = logit(IO token) − logit(S token) at END (the standard
IOI behavioral metric; falls when the circuit is damaged). Damage = margin(unmasked) −
margin(condition) (positive = ablation hurt the IO behavior). Conditions, budget-matched per
head: top-salient (cycle, sheaf), magnitude, random.

**Verdict (Part B):** GREEN if a topological condition's damage > 0 AND beats random
(Wilcoxon p<α); PARTIAL if damage>0 but not beyond random; RED otherwise.

## Headline verdict

- **GREEN** — Part A GREEN and Part B GREEN: topology is a genuine attributor that beats the
  (weakened) magnitude baseline on a real, diffuse circuit and is causal. The project's first
  "topology is practically useful" result.
- **PARTIAL** — the expected honest middle (e.g. topology ties magnitude, or is causal but
  not better at localization). Reported plainly.
- **RED** — topology does not beat chance / is not causal even where magnitude is handicapped.

## Outputs & integrity

- `results/exp10_attribution.parquet`, `results/exp10_ablation.parquet` (sources of truth);
  `src/data_ioi.py` (prompt generator + ground-truth positions, unit-tested);
  `src/run_exp10.py`; `src/compute_stats_exp10.py` → `results/exp10_stats.json`;
  `src/write_findings_exp10.py` → `FINDINGS_exp10.md`. Fold into WRITEUP as Result L.
- **Provenance discipline (hard rule):** every number computed to JSON, mechanically
  rendered, verified via `json.load` — never hand-typed, never from terminal output.

## Honest pre-commitments

- Single model (GPT-2 small), one IOI template, ~48 prompts, single seed — directions, not
  magnitudes. A positive result motivates more templates/models; it does not generalize alone.
- The magnitude baseline is still strong (0.54 mass on top edge); "topology ties magnitude"
  is the most likely Part A outcome and is reported as PARTIAL, not spun as success.
- IOI ground truth is the *primary* name-mover edge; IOI also involves S-inhibition and
  duplicate-token heads we are NOT attributing — scope is name-mover only.
- Validate the prompt generator: model's top next-token must be the IO name at usable rate;
  if accuracy is too low the attribution target is unreliable (report and fix template).
