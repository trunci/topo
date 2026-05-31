# Experiment 8 — Cellular sheaves over the residual stream

**Date:** 2026-05-31
**Status:** pre-registered (design approved before any results)

## Motivation

Every experiment so far measures attention-graph **shape** (does an H1 cycle exist?) and
is blind to **what flows** along edges. A cellular sheaf attaches a vector space (stalk) to
each node/edge with restriction maps between them; the **sheaf Laplacian** `L_F = δᵀδ`
measures *consistency of information flow*: `xᵀ L_F x = Σ_edges ‖F_v x_v − F_u x_u‖²` is the
total disagreement after transporting each endpoint's vector along its edge map.

This is the proposal's fallback #2 ("sheaves over the residual stream"). Literature survey
(Hansen-Ghrist 1808.01513; Neural Sheaf Diffusion, Bodnar et al. NeurIPS 2022 2202.04579;
Connection-Laplacian SNNs, Barbero et al. 2206.08702; Singer-Wu VDM 1102.0075; GAT-as-sheaf
2601.21207; Bosca-Ghrist 2603.14831) confirms: **no prior work puts a sheaf on a
transformer residual stream** — this construction is novel, but every building block is
established and non-learned.

## Question

Does a sheaf-Laplacian **consistency** measure (a) discriminate reasoning hop-count and add
beyond H1 persistence (Spike frame), and (b) predict reasoning failure beyond model
confidence where H1 did not (Exp 7 frame)? I.e. does *what flows* beat *graph shape*?

## Constructions (both computed; Tier 1 is the honest baseline)

Per (layer, head) attention matrix A (reuse the existing symmetrize + top-k sparsify graph,
top_k=8). Let `w_uv = (A_uv + A_vu)/2` be the symmetrized edge weight.

### Tier 1 — scalar weighted-graph Laplacian (shape-only baseline)
- d = 1; restriction = √w. Then `L = D − W` (the weighted graph Laplacian); use the
  normalized `L_norm = I − D^{-1/2} W D^{-1/2}`.
- Features: **Fiedler value λ₁** (smallest nonzero eigenvalue = algebraic connectivity) and
  `dim ker` (≈ #connected components). Pure graph shape, like H1 — the control.

### Tier 2 — Connection-Laplacian sheaf over the residual stream (the real test)
- Node stalk `R^d`, d = 4: token's residual-stream hidden vector reduced by PCA over token
  positions (per item).
- Per node v: local PCA over its top-k neighbours' reduced vectors → orthonormal frame
  `O_v ∈ O(d)`.
- Per edge (u,v): orthogonal transport `O_uv = U Vᵀ` from `SVD(O_uᵀ O_v)` (Procrustes);
  restriction maps weighted by √w_uv. Handles attention's asymmetry (symmetrize the scalar
  weight, orthogonalize the transport map).
- Assemble block `L_F` (nd × nd) by the Hansen-Ghrist formulas; normalize
  `Δ_F = D^{-1/2} L_F D^{-1/2}`.
- Features: **harmonic-space dim** `dim ker(Δ_F)` (global consistency capacity; theory:
  ≤ d, equality iff transport is path-independent), **spectral gap λ₁(Δ_F)**, and **mean
  per-edge discord** `mean_e ‖O_uv x_v − x_u‖²` (a direct "inconsistency of flow" scalar).

Aggregate each feature over (layer, head) per item (mean / max), same as Exp 7's topology
aggregation.

## Two evaluation frames (both run)

**Frame 1 — Spike discrimination (reuse build_items, hops 1/2/3 paired by difficulty):**
- Does each sheaf feature discriminate hop-count? Paired/rank test across hops.
- Nested test: does the sheaf feature add beyond H1 persistence (partial Spearman /
  nested OLS, controlling H1 + first-order scalars), mirroring Exp 3/4.
- Per-feature verdict: **GREEN** if discriminates AND adds beyond H1 (delta-R2 >= 0.02,
  nested-F p < 0.05); **RED** otherwise.

**Frame 2 — failure prediction (reuse the Exp 7 design exactly):**
- Add sheaf features to the nested logistic model. Compare:
  M0 = confidence; M1 = confidence + H1 (Exp 7's full); M2 = confidence + sheaf;
  M3 = confidence + H1 + sheaf. 5-fold CV ROC-AUC.
- Per-frame verdict: **GREEN** if sheaf beats chance AND M2 > M0 (sheaf adds beyond
  confidence, where H1 did not); **PARTIAL** if beats chance but no add; **RED** if not.

## Statistics & outputs (single source of truth = parquet/JSON)

- `results/exp8_features.parquet` — one row per item: y (is_correct), hop, family, all sheaf
  features (T1 + T2), reused H1 + confidence + first-order controls.
- `src/compute_stats_exp8.py` → `results/exp8_stats.json` (both frames, pre-registered
  verdicts). No hand-typed numbers.
- `src/write_findings_exp8.py` → `FINDINGS_exp8.md` (mechanical).
- Fold into `WRITEUP.md` as Result I.

## Model & safety

- Qwen2.5-0.5B-Instruct (continuity with Spike + Exp 7). CPU-only, float32, eager,
  `torch.set_num_threads(2)`, one model, gc per item. Same gentle path as Exp 7.
- Reuse Exp 7's 180 mixed-hop items so Frames 1 and 2 share one forward-pass sweep
  (capture attentions AND hidden states once per item).

## Honest caveats (pre-committed)

- Non-learned sheaf; restriction maps from local PCA + Procrustes, not from the model's true
  computation — Tier 3 (OV-circuit maps) is the more faithful follow-up, not in scope here.
- d = 4 and top_k = 8 are fixed design knobs; a small sensitivity check (d ∈ {2,4,8}) is a
  follow-up, not the headline.
- Small single model, ~180 items, single seed. Direction, not magnitude.
- If a sheaf feature is constant/degenerate (e.g. harmonic dim always = d), report it as
  uninformative rather than forcing a verdict.
