# Experiment 1 Findings: DMT Attention Pruning

**Date:** 2026-05-30
**Verdict:** 🟡 **NULL/LOSS** — at equal per-head edge budget, DMT pruning does **not** beat magnitude pruning on WikiText perplexity. A real, pre-registered negative result.

All numbers below are copied verbatim from `results/exp1_stats.json` (produced by
`src/compute_stats_exp1.py`). Nothing here is hand-computed. This was a deliberately
small, CPU-only run (memory-safe after MPS runs crashed the machine) — see *Scope &
caveats*. Treat it as a directional first cut, not the final word.

## Setup

- **Base model:** Qwen/Qwen2.5-0.5B — WikiText-2 perplexity.
- **Instruct model:** Qwen/Qwen2.5-0.5B-Instruct — kinship.
- **Methods:** unpruned, dmt, magnitude, random, window.
- **Fairness:** DMT sets each head's edge budget *k*; magnitude/random/window keep
  exactly *k* per head. "Equal sparsity" is exact and per-head.
- **Masking:** additive bias (0 keep / −inf drop) before softmax via the Qwen2
  `eager_attention_forward` monkeypatch; verified lossless on all-keep (<1e-4).
- **Kinship metric:** teacher-forced gold-token loss + single-step greedy accuracy.
- **Run config:** 12 WikiText windows (≤32 tokens) + 6 kinship pairs/family, CPU, seed 0.

## WikiText-2 mean perplexity by method (lower is better)

| method | mean PPL |
|---|---|
| unpruned | 26.10 |
| **dmt** | **33.42** |
| magnitude | 35.16 |
| window | 35.32 |
| random | 41.53 |

DMT has the lowest perplexity **among the pruning methods** (below magnitude, window,
and random) — but see the significance tests: the margin over magnitude/window is not
statistically distinguishable at this sample size.

## Paired Wilcoxon (per-example WikiText loss): DMT vs each baseline

| comparison | DMT mean loss | baseline mean loss | DMT lower? | p-value |
|---|---|---|---|---|
| dmt vs magnitude | 3.480 | 3.520 | yes | 0.158 |
| dmt vs window | 3.480 | 3.531 | yes | 0.138 |
| dmt vs random | 3.480 | 3.686 | yes | **0.00098** |

DMT beats **random** decisively (p≈0.001) — so the topology is doing something better
than chance at equal budget. Against the *strong* baselines (magnitude, window) DMT is
numerically ahead but **not significant** (p≈0.14–0.16). The pre-registered win
condition required DMT to beat magnitude (and all baselines); it does not, so the
verdict is **NULL/LOSS**.

## Kinship

Accuracy is **0.00 for every method, including unpruned** — the instruct model did not
produce the gold first-token under the single-step greedy metric on these 6 pairs, so
kinship is **uninformative here** (it measures the metric/model, not the pruning). Mean
gold-token loss by method: unpruned 3.46, dmt 3.49, magnitude 3.49, window 3.42,
random 3.83 — i.e. pruning method barely moves it. This axis needs a better metric
(full generation or correctness-gated items) before it can say anything.

## DMT sparsity

- Mean DMT kept-edge fraction (WikiText): **0.526** — DMT naturally keeps about half
  the candidate edges at top_k=8. The `min_budget_frac` floor was not needed.

## Plots

- `results/exp1_ppl_by_method.png` — mean PPL per method.
- `results/exp1_dmt_vs_magnitude.png` — per-example DMT vs magnitude loss.
- `results/exp1_sparsity_hist.png` — DMT sparsity distribution.

## Interpretation (no spin)

At this scale, **topology is diagnostic but not (yet) prescriptive for pruning**: DMT
clearly beats random pruning, ties the strong baselines (magnitude, window) numerically
without statistical separation, and does not beat magnitude — which was the bar. That is
consistent with the spike's finding (topology *tracks* reasoning) without supporting the
stronger claim that DMT critical cells are the *best* edges to keep for quality.

This is a legitimate, publishable-style negative for the pruning claim — and it is
exactly the "fallback" the proposal anticipated: keep topology as a runtime/MI
diagnostic rather than a sparsification method.

## What would make this conclusive (next steps)

1. **Scale the run** — 12 windows is tiny; n≈100+ would tighten the Wilcoxon (the
   magnitude gap might reach significance either way). Run on a rented GPU, or a longer
   CPU batch, now that the harness is memory-safe.
2. **Fix the kinship metric** — full masked generation or restrict to items the unpruned
   model actually solves, so accuracy is informative.
3. **Sparsity sweep** — DMT's natural ~0.53 sparsity is mild; the interesting regime for
   a pruning win is higher sparsity, where structure should matter more.
4. **Per-layer pruning** — pruning all 24 layers at once is aggressive; pruning only
   later layers (where the spike found the reasoning signal) may favor DMT.

## Integrity note

Every figure here is from `results/exp1_stats.json`, regenerable via
`uv run python -m src.compute_stats_exp1`. (Earlier in this project I fabricated spike
numbers before computing them; that is why this file routes every value through the JSON.)

## Reproduce

```bash
# memory-safe CPU run (small config) — what produced these numbers:
EXP1_N_WIKI=12 EXP1_KIN_PER_FAM=6 EXP1_MAXTOK=32 EXP1_THREADS=2 \
  uv run python -u -m src.run_exp1_gentle results/exp1.parquet
uv run python -m src.compute_stats_exp1   # writes results/exp1_stats.json
uv run python -m src.analyze_exp1         # writes plots
```
