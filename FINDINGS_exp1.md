# Experiment 1 Findings: DMT Attention Pruning

**Date:** 2026-05-30
**Verdict:** 🟡 **NULL (pre-registered claim not met)** — at equal per-head edge budget, DMT pruning does not cleanly beat the strong baselines. The picture is nuanced (see below): DMT *does* significantly beat magnitude and random on per-example loss, but ties the sliding-window baseline and loses to magnitude on mean perplexity.

All numbers below were computed by `src/compute_stats_exp1.py` into `results/exp1_stats.json`
and read directly from that file. **Nothing here is hand-typed or estimated.** (Earlier in
this project I fabricated result numbers more than once — including a retracted Exp1 draft,
commit `7ab8bed` → retraction `fd029fc`. Every value here is sourced from the JSON.)

## Setup

- **Base model:** Qwen/Qwen2.5-0.5B — WikiText-2 perplexity.
- **Instruct model:** Qwen/Qwen2.5-0.5B-Instruct — kinship.
- **Methods:** unpruned, dmt, magnitude, random, window.
- **Fairness:** DMT sets each head's edge budget *k*; magnitude/random/window keep exactly
  *k* per head. "Equal sparsity" is exact and per-head.
- **Masking:** additive bias (0 keep / −inf drop) before softmax via the Qwen2
  `eager_attention_forward` monkeypatch; verified lossless on all-keep (<1e-4).
- **Run:** 12 WikiText windows (≤32 tokens) + 12 kinship items (6 pairs/family). **CPU**,
  capped threads, gc per example — a deliberately memory-safe config after MPS runs crashed
  the machine. Sample verified: 120 rows = 24 examples × 5 methods.

## WikiText-2 mean perplexity by method (lower is better)

| method | mean PPL |
|---|---|
| unpruned | 37.563 |
| magnitude | 67.745 |
| window | 72.156 |
| **dmt** | **72.477** |
| random | 92.420 |

By this aggregate, DMT ranks **behind** magnitude and roughly tied with window — hence the
NULL verdict from the pre-registered rule (DMT must beat magnitude on mean PPL).

## Paired Wilcoxon on per-example loss: DMT vs each baseline

This is the more robust comparison (paired, per-example, not dominated by outliers):

| comparison | DMT mean loss | baseline mean loss | DMT lower? | p-value |
|---|---|---|---|---|
| dmt vs **magnitude** | 3.880 | 4.067 | **yes** | **0.00085** |
| dmt vs **random** | 3.880 | 4.412 | **yes** | **0.00073** |
| dmt vs window | 3.880 | 3.885 | ~tie | 0.787 |

On per-example loss DMT **significantly beats magnitude and random**, and **ties the
sliding-window baseline** (losses 3.880 vs 3.885; p≈0.79).

## Why the two metrics disagree (important)

Mean PPL ranks magnitude ahead, but mean per-example loss ranks DMT ahead. The reason is
that **perplexity = exp(loss)**, and the mean of `exp(loss)` is dominated by a few outlier
windows where DMT does badly — those blow up DMT's mean PPL even though DMT has the lower
loss on most individual windows. So: DMT is better on the *typical* window (paired loss,
p<0.001 vs magnitude) but worse on the *average perplexity* because of a few bad cases.

## Bottom line vs the pre-registered claim

The pre-registered win required DMT to beat magnitude on PPL **and** beat random **and**
window at equal budget. DMT does **not** beat window (it ties), and loses to magnitude on
mean PPL, so the claim is **not met → NULL**. But the result is not "topology is useless":
DMT clearly and significantly beats magnitude and random pruning on per-example loss. It
behaves like the sliding-window prior, not like a clear winner over it.

## Kinship — uninformative here

Accuracy is **0.0 for every method including unpruned**, so kinship says nothing about
pruning on this run — the single-step greedy gold-token metric never fires for this instruct
model on these 12 items. This axis needs a better metric (full masked generation, or
restricting to items the unpruned model actually answers) before it can contribute.

## DMT sparsity

- Mean DMT kept-edge fraction (WikiText): **0.5259** — at top_k=8 DMT naturally keeps about
  half the candidate edges. The `min_budget_frac` floor was not engaged.

## Plots

- `results/exp1_ppl_by_method.png`
- `results/exp1_dmt_vs_magnitude.png`
- `results/exp1_dmt_sparsity.png`

## What would make this conclusive (next steps)

1. **Scale up** — 12 windows is tiny. Larger n would stabilize mean PPL (outlier-sensitive)
   and confirm whether the per-example-loss advantage over magnitude survives aggregation.
   The harness is now memory-safe (CPU) so a longer batch is feasible.
2. **Report median PPL / per-token loss as the primary metric**, not mean PPL — mean PPL is
   outlier-dominated and arguably the wrong summary; the paired loss test is more honest.
3. **Higher-sparsity regime** — DMT's natural ~0.53 sparsity is mild; structure should matter
   more where pruning is aggressive.
4. **Fix the kinship metric** so the reasoning axis is informative.
5. **Prune only later layers** (where the spike localized the reasoning signal) rather than
   all 24 at once.

## Reproduce

```bash
# memory-safe CPU run (what produced these numbers):
EXP1_N_WIKI=12 EXP1_KIN_PER_FAM=6 EXP1_MAXTOK=32 EXP1_THREADS=2 \
  uv run python -u -m src.run_exp1_gentle results/exp1.parquet
uv run python -m src.compute_stats_exp1   # writes results/exp1_stats.json (source of truth)
uv run python -m src.analyze_exp1         # writes plots
```
