# Experiment 13 Findings

Deep-hop failure prediction (hop=4, hop=5). Generated mechanically from `results/exp13_stats.json`; do not edit by hand.

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Hops tested: [4, 5]
- Overall accuracy: 0.525
- Pre-condition met (at least one hop in [0.35, 0.70]): True
- Significance level alpha = 0.05; 5-fold stratified CV

Accuracy by hop:

| hop | accuracy |
|---|---|
| 4 | 0.550 |
| 5 | 0.500 |

### All items (hop=4 + hop=5 combined)

**Verdict: GREEN**

- Items: 120; base rate correct = 0.525; majority-class acc = 0.525

Cross-validated ROC-AUC (predicting is_correct):

| model | mean AUC | std | folds |
|---|---|---|---|
| confidence only (M0) | 0.681 | 0.070 | 5 |
| confidence + topology (M1) | 0.932 | 0.070 | 5 |
| topology only | 0.930 | 0.067 | 5 |
| confidence + first-order controls | 0.953 | 0.055 | 5 |

- delta-AUC (M1 - M0): median 0.245, paired Wilcoxon p = 0.0312
- topology beats chance: True
- topology adds beyond confidence: True

### Hop=4 only

**Verdict: GREEN**

- Items: 60; base rate correct = 0.550; majority-class acc = 0.550

Cross-validated ROC-AUC (predicting is_correct):

| model | mean AUC | std | folds |
|---|---|---|---|
| confidence only (M0) | 0.780 | 0.100 | 5 |
| confidence + topology (M1) | 0.960 | 0.043 | 5 |
| topology only | 0.977 | 0.046 | 5 |
| confidence + first-order controls | 0.954 | 0.046 | 5 |

- delta-AUC (M1 - M0): median 0.171, paired Wilcoxon p = 0.0312
- topology beats chance: True
- topology adds beyond confidence: True

### Hop=5 only

**Verdict: GREEN**

- Items: 60; base rate correct = 0.500; majority-class acc = 0.500

Cross-validated ROC-AUC (predicting is_correct):

| model | mean AUC | std | folds |
|---|---|---|---|
| confidence only (M0) | 0.617 | 0.145 | 5 |
| confidence + topology (M1) | 1.000 | 0.000 | 5 |
| topology only | 1.000 | 0.000 | 5 |
| confidence + first-order controls | 1.000 | 0.000 | 5 |

- delta-AUC (M1 - M0): median 0.361, paired Wilcoxon p = 0.0312
- topology beats chance: True
- topology adds beyond confidence: True

## Interpretation

GREEN on at least one hop: topology adds predictive power for failure beyond the model's own confidence in the deeper-hop regime, confirming the miscalibration hypothesis.
