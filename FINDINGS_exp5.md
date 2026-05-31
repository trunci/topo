# Experiment 5 Findings

Causal test (E1) and cycle inspection (E2) for the induction circuit. Generated mechanically from `results/exp5_stats.json`; do not edit by hand.

- Model: `gpt2`
- Induction copy distance S = 25
- Gap tolerance for E2: |gap - S| <= 2
- Significance level alpha = 0.05

## E2 - Cycle inspection (descriptive)

**Verdict: GREEN**

Metric: fraction of cycle edges with |gap - S| <= 2 (S=25).

- Induction heads (n = 9): mean fraction at gap ~ S = 0.2332
- Non-induction heads (n = 10): mean fraction at gap ~ S = 0.0134
- Mann-Whitney (induction > non-induction), p = 0.0004345

Per-head fractions:

| group | fractions |
|---|---|
| induction | 0.333, 0.333, 0.000, 0.312, 0.176, 0.333, 0.167, 0.300, 0.143 |
| non-induction | 0.105, 0.000, 0.029, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 |

Gap histograms (gap -> count):

| group | histogram |
|---|---|
| induction | 1:3, 2:1, 3:5, 4:2, 5:1, 7:2, 8:2, 10:2, 11:1, 12:2, 13:1, 14:1, 17:1, 18:2, 19:2, 21:5, 22:3, 23:6, 24:4, 25:5, 26:4, 27:1, 28:1, 29:3, 30:4, 31:2, 32:3, 33:1, 34:1, 37:1, 38:3, 39:2, 40:4, 41:1, 42:1, 43:1, 45:2, 46:1, 47:1, 54:1 |
| non-induction | 1:37, 2:70, 3:34, 4:52, 5:11, 6:10, 7:12, 8:15, 9:1, 10:3, 12:2, 13:1, 14:1, 15:1, 17:3, 22:1, 25:4, 27:1, 30:3, 32:1, 35:2, 36:1, 37:1, 39:1, 41:1, 45:1, 46:2, 50:1, 53:1 |

## E1 - Causal ablation (functional)

**Verdict: RED**

Readout: model second-copy next-token cross-entropy. Damage(condition) = loss(condition) - loss(unmasked), paired across 8 sequences.

Median damage by condition:

| condition | median damage |
|---|---|
| cycle | -0.0270 |
| magnitude | -0.0300 |
| random | -0.0227 |

Paired one-sided Wilcoxon signed-rank (cycle damage greater):

| comparison | median diff | p-value |
|---|---|---|
| cycle vs magnitude | 0.0070 | 0.2305 |
| cycle vs random | -0.0031 | 0.9453 |

Sanity check (not a verdict):

- cycle damage > 0: False
- magnitude damage > 0: False
- ablation/readout too weak to interpret if cycle and magnitude damages are ~0

## Interpretation

E1 RED: cycle ablation does not significantly exceed random - no causal signal that topology marks special edges for induction behavior. CAUTION: at least one of cycle/magnitude damage is not positive, so the ablation/readout may be too weak to interpret the E1 contrast confidently. E2 GREEN: induction-head cycle edges concentrate at the induction gap S more than non-induction heads.
