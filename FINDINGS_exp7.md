# Experiment 7 Findings

Failure prediction from attention topology. Generated mechanically from `results/exp7_stats.json`; do not edit by hand.

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Overall accuracy: 0.772
- Significance level alpha = 0.05; 5-fold stratified CV

Accuracy by hop:

| hop | accuracy |
|---|---|
| 1 | 0.717 |
| 2 | 0.850 |
| 3 | 0.750 |

### All items (mixed 1/2/3-hop)

**Verdict: PARTIAL**

- Items: 180; base rate correct = 0.772; majority-class acc = 0.772

Cross-validated ROC-AUC (predicting is_correct):

| model | mean AUC | std | folds |
|---|---|---|---|
| confidence only (M0) | 0.805 | 0.048 | 5 |
| confidence + topology (M1) | 0.789 | 0.042 | 5 |
| topology only | 0.690 | 0.114 | 5 |
| confidence + first-order controls | 0.787 | 0.018 | 5 |

- delta-AUC (M1 - M0): median 0.009, paired Wilcoxon p = 0.6875
- topology beats chance: True
- topology adds beyond confidence: False

### 2-hop only (Spike regime)

**Verdict: PARTIAL**

- Items: 60; base rate correct = 0.850; majority-class acc = 0.850

Cross-validated ROC-AUC (predicting is_correct):

| model | mean AUC | std | folds |
|---|---|---|---|
| confidence only (M0) | 0.787 | 0.281 | 5 |
| confidence + topology (M1) | 0.785 | 0.212 | 5 |
| topology only | 0.774 | 0.062 | 5 |
| confidence + first-order controls | 0.755 | 0.213 | 5 |

- delta-AUC (M1 - M0): median 0.000, paired Wilcoxon p = 0.6875
- topology beats chance: True
- topology adds beyond confidence: False

## Interpretation

PARTIAL: topology predicts failure above chance, but does not add beyond the model's confidence -- it may be re-deriving what confidence already encodes.
