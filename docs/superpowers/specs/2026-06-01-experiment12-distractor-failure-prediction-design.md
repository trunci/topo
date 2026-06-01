# Experiment 12 — Distractor-augmented failure prediction

**Date:** 2026-06-01
**Status:** pre-registered (design approved before any results)

## Motivation

Experiment 7 (Result H) found that attention-graph H1 topology predicts reasoning errors above
chance (AUC 0.690) but adds nothing beyond the model's own confidence (delta-AUC median 0.009,
p = 0.6875). The WRITEUP identified the likely reason: on simple kinship/ordering chains, the
model's logit margin is already well-calibrated — its "I'm unsure" feeling correlates with
actual correctness, leaving no room for topology to add signal.

The follow-up hypothesis: topology can add beyond confidence in regimes where confidence is
**poorly calibrated** — specifically, where the model is *confidently wrong* because a surface
pattern strongly triggers a wrong answer. This experiment manufactures that regime via
**answer-blocker distractors**: sentences inserted into the context that associate a wrong name
directly with the query entity, giving the model a plausible but incorrect pattern to latch onto.

## Question

Do H1 topological features predict reasoning errors beyond model confidence on items
where a distractor sentence makes the model likely to produce confident-wrong responses?

**Pre-registered prediction (committed before any run):**
- Clean items (no distractor): PARTIAL — replicates Exp 7 (topology beats chance, adds nothing
  beyond confidence).
- Distracted items (answer-blocker present): GREEN — topology adds beyond confidence
  (delta-AUC > 0, p < 0.05), because the model's logit margin tracks the distractor-induced
  pattern while topology tracks whether attention cycles span the actual reasoning chain.

## Design

### Model

**Qwen2.5-1.5B-Instruct** (cached locally). Larger than the 0.5B used in Exp 7; stronger
pattern-matcher and more likely to produce high-confidence wrong answers when a distractor
is present. CPU-only, float32, eager, `torch.set_num_threads(2)`, gc per item.

### Data generation

Two item pools, extending `data_gen.py` with distractor variants:

**Clean items** (identical format to Exp 7):
- Families: kinship + ordering
- Hops: {1, 2, 3}
- No modifications to prompt structure

**Distracted items** (new):
- Same chain as clean, but one **answer-blocker** sentence is inserted immediately before the
  question. The blocker associates a wrong name (`distractor_name`) with the query entity in a
  plausible kinship/ordering role.
- Kinship example (2-hop, gold = Tom):
  ```
  Tom is Mary's father. Mary is Sue's father. Bob is Sue's uncle. Who is Sue's grandfather?
  ```
  Here "Bob is Sue's uncle" makes Bob appear adjacent to Sue — the model may confidently output
  Bob despite Tom being the correct answer.
- Ordering example (2-hop, gold = Tom):
  ```
  Tom is taller than Mary. Mary is taller than Sue. Bob is taller than Sue. Who is the tallest?
  ```
  "Bob is taller than Sue" makes Bob appear as a candidate for tallest related to Sue.
- `distractor_name` is drawn from the NAMES pool but excluded from the chain names; it must be
  distinct from gold and all chain intermediaries.

**Scale:** n_per_family = 40 → 40 × 2 families × 3 hops × 2 types = **480 items total**
(240 clean, 240 distracted). Seed = 0.

### Features per item (identical to Exp 7)

For each item, one forward pass with `output_attentions=True`:

- **Topology (X_topo):** aggregate H1 features over all (layer, head):
  `topo_mean_persist`, `topo_max_persist`, `topo_frac_nontrivial`
- **Confidence (X_conf):** answer-token logit margin (top-1 minus top-2 logit at the first
  generated position) — scalar proxy for the model's own confidence
- **First-order controls (X_ctrl):** `ctrl_attn_distance`, `ctrl_offdiag_mass`,
  `ctrl_attn_entropy`

### Analysis (identical to Exp 7)

Nested logistic regression predicting `is_correct`, stratified 5-fold CV, scored by ROC-AUC:

| model | features |
|---|---|
| M0 (baseline) | confidence only |
| M1 (full) | confidence + topology |
| M_topo | topology only |
| M_ctrl | confidence + controls |

Primary statistic: **delta-AUC = AUC(M1) − AUC(M0)** with paired Wilcoxon test across folds
(one-sided, greater).

Run separately on three slices:
1. **Clean items only** — should replicate Exp 7 PARTIAL
2. **Distracted items only** — primary test; hypothesis GREEN
3. **All items combined** — secondary, for overall picture

### Verdict (pre-registered, applied per slice)

- **GREEN** — M_topo AUC > 0.5 AND delta-AUC > 0 with paired Wilcoxon p < 0.05
- **PARTIAL** — M_topo beats chance but delta-AUC not significant
- **RED** — M_topo does not beat chance
- **UNDERPOWERED** — base rate outside [0.15, 0.85]; reported, not a pass

### Effectiveness check (secondary)

Report separately: accuracy on clean vs. distracted items per hop. If the distractor does not
lower accuracy (model ignores it), the experiment is informative but the manipulation was
ineffective — reported honestly, not silently passed.

## Outputs

- `results/exp12_features.parquet` — one row per item: is_correct, confidence_margin, all
  topo/ctrl features, hop, family, item_type (clean/distracted), distractor_name. Single
  source of truth.
- `results/exp12_stats.json` — computed stats + verdicts for all three slices.
- `FINDINGS_exp12.md` — generated mechanically by `src/write_findings_exp12.py`.
- Folded into `WRITEUP.md` as **Result O** once results land.

## Files to create / modify

| file | action |
|---|---|
| `src/data_gen.py` | add `build_items_distracted()` and distractor-variant builders |
| `src/run_exp12.py` | forward pass + feature extraction (mirrors run_exp7.py) |
| `src/compute_stats_exp12.py` | analysis (mirrors compute_stats_exp7.py, adds slice loop) |
| `src/write_findings_exp12.py` | findings renderer |

## Honest caveats (pre-committed)

- n = 480 items, single seed, one model (Qwen2.5-1.5B-Instruct). Direction, not magnitude.
- Distractor effectiveness is empirically checked but not guaranteed: if the 1.5B model
  correctly ignores all distractors, the regime of interest will be under-populated.
- "Confidence" remains a logit-margin proxy, not a calibrated probability.
- Topology features are still aggregate over all heads; head-level or edge-level features
  (as in Exp 9/10) could be more informative but are out of scope here.
- If the distracted-items base rate falls outside [0.15, 0.85], the slice is UNDERPOWERED —
  we adjust n_per_family or hop mix and re-run (design knob, not a result).
- Positive result would motivate but not prove general applicability beyond synthetic tasks.
