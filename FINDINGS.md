# Spike Findings: Topology of Attention Tracks Reasoning Hop-Count

**Date:** 2026-05-30
**Verdict:** 🟢 **GREEN** — topology is non-trivial AND discriminates reasoning hop-count. Proceed to Experiment 1.
**Spec:** `docs/superpowers/specs/2026-05-30-topology-attention-spike-design.md`
**Plan:** `docs/superpowers/plans/2026-05-30-topology-attention-spike.md`

## What was tested

The make-or-break question for the whole research program: does the **flag complex
of a transformer's attention graph have non-trivial H₁ (cycles) that tracks
reasoning**? We tested it on Qwen2.5-0.5B-Instruct using controlled minimal pairs —
matched prompts sharing identical context, differing only in whether the question
requires 1 hop or 2 hops of reasoning (e.g. "Who is Sue's father?" vs "Who is Sue's
grandfather?").

- **Model:** Qwen2.5-0.5B-Instruct, 24 layers × 14 heads, eager attention, float32, MPS.
- **Data:** 60 matched pairs (120 prompts) across 2 relation families (kinship, spatial ordering).
- **Pipeline:** per (layer, head) attention matrix → max-symmetrize → top-k=8 sparsify →
  flag complex with filtration f(edge)=1−weight → H₁ persistent homology (gudhi).
- **Dataset produced:** 40,320 rows (120 prompts × 336 heads). `results/spike.parquet`.

## Pre-registered criteria (decided before running)

| Criterion | Threshold | Result | Verdict |
|---|---|---|---|
| 1. Non-triviality | H₁ non-trivial in a substantial fraction of heads | **64.9%** of head-examples have a persistent cycle (max-persistence > 0.05) | ✅ PASS |
| 2. Discrimination | ≥1 (layer,head) separates 2-hop vs 1-hop, paired Wilcoxon, BH-corrected | **168 of 336 heads** significant after Benjamini–Hochberg | ✅ PASS |
| 3. Plausibility (bonus) | significant heads cluster mid/late, not random | layer median **13/24**; late-layer concentration (early 46 / mid 44 / late 78) | ✅ as predicted |

Primary metric: total H₁ persistence (Σ death−birth) per head per prompt.

## The result survives adversarial confound checks

A result this strong (168/336 heads) demands skepticism. The dominant risk is that H₁
just tracks **sequence length** (more tokens → bigger graph → more cycles). It does not:

- **Length is matched.** Mean tokens: 1-hop 44.30 vs 2-hop 44.40 (diff 0.08; 24/60 pairs
  *identical* length). The hop-count manipulation barely changes prompt length.
- **The effect is two-sided.** Of 168 significant heads, 78 increase persistence for
  2-hop and 90 *decrease* it. A length artifact would push uniformly in one direction;
  bidirectional structure indicates genuine reorganization of attention topology.
- **Length-normalized metric agrees.** Persistence ÷ tokens yields 169 significant heads
  (vs 168) — the signal is not a size effect.
- **Independent replication across families.** Kinship-only: 56 significant heads.
  Ordering-only: 76. Two unrelated relation types each show the effect; **22 heads are
  significant in both families independently**. The ordering family even has 2-hop prompts
  slightly *shorter* than 1-hop (−0.53 tokens) yet still shows the effect — the opposite of
  what a length artifact predicts.

Top discriminating heads (by adjusted p): L9H8 (d=+0.29), L17H2 (−0.49), L4H8 (+0.21),
L4H7 (+0.17), L8H3 (+0.23), L18H2 (−0.49). Mix of early/mid/late layers and both signs.

Plots: `results/heatmap.png` (per-head −log₁₀ p_adj over the 24×14 grid),
`results/best_head_dist.png` (L9H8 persistence distribution, 2-hop vs 1-hop).
Confound detail: `results/confound_report.txt`.

## Caveats (honest)

- **Model accuracy is mixed.** With a free-form prompt the model substring-matches the
  gold answer ~88% of the time; with a terse-answer prompt ~38%. The topology test does
  **not** depend on the model answering correctly — attention is extracted from the forward
  pass over the prompt regardless of the generated answer — so this does not threaten the
  result. But it means we cannot yet claim the topology tracks *successful* reasoning vs
  *attempted* reasoning. That separation is follow-up work.
- **Substring-match leak in the ordering family.** The "tallest" answer is always the
  first-named entity, which appears in any restatement of the premise, inflating naive
  correctness there. Irrelevant to the topology comparison (which is per-matched-pair on
  the prompts), but it is why kinship is the cleaner family for any future correctness-gated
  analysis.
- **Single small model.** 0.5B, one architecture. Scaling behavior (does the effect
  strengthen at 7B? does it localize to known circuit layers?) is exactly what Experiment 1
  and the proposal's larger plan should test.

## What this means for the project

The central premise is **validated**: attention flag-complexes carry non-trivial topology,
and that topology is not random with respect to reasoning — it discriminates hop-count
robustly, bidirectionally, across two relation types, after correction and confound checks.
This is the green light to build **Experiment 1 (DMT-based pruning)** and to pursue the
mechanistic-correspondence claim (criterion 2 of the full proposal). The discriminating
heads concentrate toward later layers, consistent with reasoning being assembled deeper in
the network — a concrete hypothesis to test against ablation-identified circuits next.

## Reproduce

```bash
uv run python -m src.run_spike       # writes results/spike.parquet (~few min on M-series)
uv run python -m src.analyze         # prints criteria + verdict, writes plots
uv run python -m src.confound_check  # writes results/confound_report.txt
uv run pytest -q                     # unit tests (topology, data_gen); add -m slow for model tests
```
