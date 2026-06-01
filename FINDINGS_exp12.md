# Experiment 12 Findings

Distractor-augmented failure prediction. Generated mechanically from `results/exp12_stats.json`; do not edit by hand.

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Overall accuracy: 0.823
- Significance level alpha = 0.05; 5-fold CV

**Distractor effectiveness check:**

- Accuracy clean: 0.796 | distracted: 0.850
- Accuracy drop (clean - distracted): -0.054
- Effective (drop > 0.05): False

Accuracy by hop and type:

| hop | clean | distracted |
|---|---|---|
| 1 | 0.725 | 0.875 |
| 2 | 0.887 | 0.912 |
| 3 | 0.775 | 0.762 |

### All items

**Verdict: PARTIAL**

- Items: 480; base rate correct = 0.823; majority-class acc = 0.823

Cross-validated ROC-AUC (predicting is_correct):

| model | mean AUC | std | folds |
|---|---|---|---|
| confidence only (M0) | 0.777 | 0.044 | 5 |
| confidence + topology (M1) | 0.776 | 0.046 | 5 |
| topology only | 0.658 | 0.041 | 5 |
| confidence + first-order controls | 0.780 | 0.047 | 5 |

- delta-AUC (M1 - M0): median 0.002, paired Wilcoxon p = 0.5938
- topology beats chance: True
- topology adds beyond confidence: False

### Clean items (pre-registered: PARTIAL)

**Verdict: PARTIAL** *(pre-registered prediction: PARTIAL)*

- Items: 240; base rate correct = 0.796; majority-class acc = 0.796

Cross-validated ROC-AUC (predicting is_correct):

| model | mean AUC | std | folds |
|---|---|---|---|
| confidence only (M0) | 0.768 | 0.031 | 5 |
| confidence + topology (M1) | 0.774 | 0.035 | 5 |
| topology only | 0.705 | 0.095 | 5 |
| confidence + first-order controls | 0.771 | 0.016 | 5 |

- delta-AUC (M1 - M0): median -0.003, paired Wilcoxon p = 0.5938
- topology beats chance: True
- topology adds beyond confidence: False

### Distracted items (pre-registered: GREEN)

**Verdict: PARTIAL** *(pre-registered prediction: GREEN)*

- Items: 240; base rate correct = 0.850; majority-class acc = 0.850

Cross-validated ROC-AUC (predicting is_correct):

| model | mean AUC | std | folds |
|---|---|---|---|
| confidence only (M0) | 0.762 | 0.058 | 5 |
| confidence + topology (M1) | 0.762 | 0.067 | 5 |
| topology only | 0.628 | 0.080 | 5 |
| confidence + first-order controls | 0.782 | 0.076 | 5 |

- delta-AUC (M1 - M0): median -0.007, paired Wilcoxon p = 0.4062
- topology beats chance: True
- topology adds beyond confidence: False

## Prediction check

Pre-registered: clean=PARTIAL, distracted=GREEN. Prediction correct: **False**
