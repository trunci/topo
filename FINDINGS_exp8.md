# Experiment 8 Findings

Cellular sheaves over the residual stream: does *what flows* (sheaf consistency) beat *graph shape* (H1)? Generated mechanically from `results/exp8_stats.json`; do not edit by hand.

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Items: 180; overall accuracy 0.772
- alpha = 0.05; 5-fold CV; delta-R2 floor 0.02

## Frame 1 - Discrimination (does sheaf add beyond H1 at predicting hop?)

**Verdict: GREEN**

Nested OLS predicting reasoning hop-count:

| model | R2 |
|---|---|
| H1 + first-order controls (baseline) | 0.758 |
| + sheaf features (full) | 0.809 |

- delta-R2 (sheaf block) = 0.051
- nested F = 11.31, p = 0.00000

## Frame 2 - Failure prediction (does sheaf add beyond confidence?)

**Verdict: PARTIAL**

- Base rate correct = 0.772; underpowered = False

Cross-validated ROC-AUC (predict is_correct):

| model | mean AUC | std | folds |
|---|---|---|---|
| confidence only (M0) | 0.805 | 0.048 | 5 |
| confidence + H1 (M1) | 0.789 | 0.042 | 5 |
| confidence + sheaf (M2) | 0.814 | 0.026 | 5 |
| confidence + H1 + sheaf (M3) | 0.850 | 0.028 | 5 |
| sheaf only | 0.765 | 0.079 | 5 |

- sheaf vs confidence (M2 - M0): median delta 0.004, paired Wilcoxon p = 0.4375
- sheaf beats chance: True; adds beyond confidence: False

## Interpretation

Frame 1 GREEN: sheaf consistency adds predictive power for reasoning hop-count beyond H1 persistence -- 'what flows' carries structure that graph shape alone misses. Frame 2 PARTIAL: sheaf predicts failure above chance but does not beat confidence -- same ceiling H1 hit in Exp 7.
