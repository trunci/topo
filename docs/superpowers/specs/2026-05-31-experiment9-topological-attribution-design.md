# Experiment 9 — Topological attribution + causal validation

**Date:** 2026-05-31
**Status:** pre-registered (design approved before results)
**Lineage:** supersedes the scalar-prediction framing (Exp 8 / Result J, withdrawn).
Built on the explicit finding that **global graph scalars are confounded by sequence
length** (exploratory probe: token-length↔hop ρ=0.76, Fiedler↔token-length ρ=−0.87, and
Fiedler's edge over H1 does not survive adding length: ΔR² 0.044 → 0.005, p=0.069).

## Motivation & reframing

Every confounded result in this project came from **cross-example scalar regression**
(predict a per-example label from a per-example graph statistic). Every result that
*survived* confound checks lived on **fixed-length substrates** and was **within-example**
(suppression Exp 3/4, causal ablation Exp 6, descriptive cycle-gap E2). The lesson: stop
predicting example-level labels from global scalars; instead use topology as a **per-edge
attributor** over a single attention graph and validate it causally. This is
explainability-native and confound-resistant by construction — all comparisons are between
edges *within one fixed-length graph*, so sequence length cannot confound.

**The contribution we are testing:** topology as an *unsupervised, causally-validated
attributor* of circuit edges — does the topological structure of an attention graph
**identify the edges that mechanistically implement a known circuit**, better than the
trivial attention-magnitude baseline, and confirmed by ablation? If yes, that is a genuine
mechanistic-interpretability method, not a correlation.

## Substrate

Induction on repeated-random sequences (fixed S=25, prefix=5), gpt2 + distilgpt2 — the
substrate where we have ground-truth circuit edges and validated machinery (Exp 5/6). The
**ground-truth circuit edge** for a second-copy query position i is the directed induction
copy edge (i → i−S+1): the token after the first occurrence. Everything is fixed-length, so
no length confound.

## Part A — Attribution quality (the new result)

For each induction head and sequence, assign every directed causal edge (q>k) a
**topological saliency** and ask whether saliency ranks the ground-truth copy edges above
other edges — and crucially, **better than the attention-magnitude baseline**.

- **Saliency definitions (each tested separately):**
  - `cycle_participation`: 1 if the (symmetrized) edge is a critical 1-cell of the head's
    flag complex (Morse), else 0 — reuse `morse.critical_cycle_edges`.
  - `sheaf_discord`: per-edge transport disagreement `‖F_v x_v − F_u x_u‖²` from the
    residual-stream sheaf — reuse `sheaf.mean_discord`'s per-edge term (refactor to expose
    per-edge values).
- **Baseline saliency:** `attention_magnitude` = A[q,k] (the trivial attributor).
- **Metric:** treat "is this the ground-truth copy edge" as a binary label over candidate
  edges; compute **ROC-AUC of each saliency** at recovering copy edges, per (head, seq),
  averaged. Also precision@k where k = #copy edges.

**Pre-registered verdict (Part A):**
- **GREEN** — a topological saliency achieves AUC significantly > 0.5 (one-sided t/Wilcoxon
  over head×seq) **AND** significantly exceeds the attention-magnitude baseline AUC
  (paired). Topology adds attribution power beyond raw attention weight.
- **PARTIAL** — topological saliency beats chance but **not** the magnitude baseline (it
  localizes the circuit, but no better than "look at the biggest weight"). This is the
  likely outcome given Exp 6 found cycle≈magnitude; we pre-commit to reporting it plainly.
- **RED** — topological saliency does not beat chance.

## Part B — Causal validation (does flagged = functional)

Reuse the Exp 6 directed, edge-exact, cumulative ablation (`ablation_to_bias`) and
second-copy-loss readout. Ablate, budget-matched per head:
- top-k **topology-salient** edges, top-k **magnitude** edges, k **random** edges.

**Pre-registered verdict (Part B):**
- **GREEN** — ablating topology-salient edges damages induction significantly more than
  random (paired Wilcoxon p<0.05) and ≥ magnitude (not significantly worse).
- **PARTIAL** — beats random but not magnitude. **RED** — does not beat random.
(This re-tests Exp 6's finding with the attribution framing; budget set adequately, top_k≥16,
per the Exp 6b sweep lesson.)

## Headline verdict

- **GREEN (the respectable result):** Part A GREEN **and** Part B GREEN — topology is an
  unsupervised attributor that localizes the induction circuit beyond attention magnitude
  **and** the edges it flags are causally necessary. A real interpretability method.
- **PARTIAL:** the common honest case — e.g. A PARTIAL (topology ≈ magnitude at
  attribution) but B GREEN: "topology recovers the circuit and it's causal, but a magnitude
  baseline does about as well" — still a clean, publishable characterization.
- **RED:** topology neither localizes beyond chance nor validates causally.

## Outputs & integrity

- `results/exp9_attribution.parquet` (Part A: per head×seq×saliency AUC / p@k),
  `results/exp9_ablation.parquet` (Part B), both single-source-of-truth.
- `src/compute_stats_exp9.py` → `results/exp9_stats.json` (pre-registered verdicts);
  `src/write_findings_exp9.py` → `FINDINGS_exp9.md` (mechanical).
- **Provenance discipline (hard rule):** every number computed to JSON, rendered
  mechanically, verified via `json.load` — never hand-typed, never from terminal output.
- Fold into WRITEUP as Result K.
- CPU-only, float32, eager, `torch.set_num_threads(2)`, one model at a time, gc per item.

## Honest pre-commitments

- **Most likely outcome is A-PARTIAL** (Exp 6 already hinted cycle≈magnitude). We are
  explicitly testing whether topology beats the trivial baseline; if it ties, we say so —
  that is the whole point of including the magnitude baseline.
- Sheaf per-edge discord is the more novel attributor than cycle-participation; if discord
  beats magnitude where cycle-participation doesn't, that *partially rehabilitates* the
  flow idea Result J withdrew — reported carefully, not oversold.
- Induction only, two small models, fixed length. A positive result motivates extension to
  IOI / real circuits; it does not by itself generalize.
