# Experiment 6 Findings

Stronger directed causal test of induction cycle edges (resolving Exp 5's underpowered E1). Generated mechanically from `results/exp6_stats.json`; do not edit by hand.

- Induction copy distance S = 25
- Gap tolerance (treatment = directed cycle edges at |gap - S| <= 2)
- Significance level alpha = 0.05
- Intervention: directed, edge-exact, cumulative across all induction heads. Readout: second-copy next-token cross-entropy.
- Damage(condition) = loss(condition) - loss(unmasked), paired across sequences.

## Model: `distilgpt2`

**Verdict: GREEN**

- Sequences: 12; mean edges ablated per sequence K = 6.08

Median damage by condition (loss increase vs unmasked; positive = ablation hurt induction):

| condition | median damage | mean damage |
|---|---|---|
| cycle | 0.0043 | 0.0269 |
| magnitude | -0.0016 | -0.0032 |
| random | 0.0000 | -0.0021 |

Paired one-sided Wilcoxon (cycle damage greater):

| comparison | median diff | p-value |
|---|---|---|
| cycle vs magnitude | 0.0097 | 0.0261230 |
| cycle vs random | 0.0069 | 0.0017090 |

- cycle damage positive: True
- cycle beats random (p < 0.05): True

Interpretation (distilgpt2): GREEN -- ablating the directed gap~S critical-cycle edges damages induction significantly more than random edges of equal budget. The topology-selected edges are causally functional for induction behavior.

## Model: `gpt2`

**Verdict: INCONCLUSIVE**

- Sequences: 12; mean edges ablated per sequence K = 2.50

Median damage by condition (loss increase vs unmasked; positive = ablation hurt induction):

| condition | median damage | mean damage |
|---|---|---|
| cycle | 0.0000 | -0.0050 |
| magnitude | 0.0000 | 0.0007 |
| random | 0.0000 | -0.0002 |

Paired one-sided Wilcoxon (cycle damage greater):

| comparison | median diff | p-value |
|---|---|---|
| cycle vs magnitude | 0.0000 | 0.8203125 |
| cycle vs random | 0.0000 | 0.5000000 |

- cycle damage positive: False
- cycle beats random (p < 0.05): False

Interpretation (gpt2): INCONCLUSIVE -- cycle damage is not positive, so the ablation/readout remains too weak to interpret the contrast (the Exp 5 E1 failure mode).
