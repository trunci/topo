# Follow-up Study Proposal — Does "what flows" really beat "what shape"?
## Validating and strengthening the sheaf-flow signal (Exp 8, Frame 1)

**Date:** 2026-05-31
**Status:** proposal (pre-registration draft) — for review before implementation
**Anchors:** Direction 1 (validity & de-confounding) → Direction 2 (faithful OV-circuit
sheaf), with Directions 3–4 (scale, mechanism) as conditional extensions.

---

## 0. One-line thesis

Exp 8 Frame 1 found that a cellular-sheaf "flow-consistency" feature block adds predictive
power for reasoning depth **beyond** H1 topology (ΔR² = 0.051, nested-F p = 3.7e-8). This
study asks the only two questions that decide whether that is a real finding: **(1) is the
added signal actually about *flow*, or is it a shape statistic in disguise; and (2) does the
model's *own* information transport (the OV circuit) carry it, rather than generic
geometry?** Both questions carry pre-registered kill criteria that can falsify the headline.

---

## 1. Background — what Frame 1 found, and why it is fragile

Exp 8 built a **cellular sheaf** over the residual stream of Qwen2.5-0.5B-Instruct: node
stalks are PCA-reduced hidden vectors (d = 4), edge restriction maps are orthogonal frames
from local PCA + Procrustes, and the **sheaf Laplacian** measures *consistency of
information flow*, `xᵀ L_F x = Σ_edges ‖F_v x_v − F_u x_u‖²`. Three scalar features per
(layer, head) were aggregated per item: Fiedler value (Tier-1 scalar Laplacian), sheaf
spectral gap, harmonic dimension, and **discord** (`mean_e ‖F_v x_v − F_u x_u‖²`).

**Frame 1 (discrimination):** a nested OLS predicting reasoning hop-count from
`[H1 persistence + first-order controls]` (baseline, R² = 0.758) vs `[+ sheaf block]` (full,
R² = 0.809) gave ΔR² = 0.051, nested-F p = 3.7e-8 → **GREEN**. (All numbers are the
committed `results/exp8_stats.json`.)

**Why this is fragile — three threats, in priority order:**

1. **The carrier may be shape, not flow.** The sheaf "block" mixes features of two very
   different kinds. **Fiedler value is a scalar graph-Laplacian quantity — pure graph
   *shape*** (algebraic connectivity), no different in spirit from H1. **Harmonic dimension
   was degenerate** in Exp 8 (always = d, by construction on connected graphs with
   orthogonal maps), so it contributed nothing. The *only* genuinely vector-valued,
   flow-dependent feature is **discord**. If the ΔR² is carried by Fiedler, the "flow beats
   shape" headline is simply false — we would have shown "a different shape statistic beat
   H1," which is far weaker. **Exp 8 never decomposed the block. This is the make-or-break
   question and it has not been answered.**

2. **Hop-count is confounded with surface complexity.** More hops ⇒ longer context ⇒ more
   tokens and more named entities. The sheaf may be reading sequence length / entity count,
   not reasoning depth. The Spike controlled this with length-matched kinship; Exp 8 did
   not. ΔR² could be partly (or wholly) a surface-length effect.

3. **The construction is not faithful to the model.** Restriction maps come from *local PCA
   of hidden states* — a generic geometric choice, **not the linear transport the model
   actually applies**. The phrase "what flows along edges" is therefore aspirational: we are
   measuring consistency of an *imposed* geometry, not of the model's computation. The
   literal claim requires restriction maps derived from the head's **OV circuit**.

This study confronts all three. Threats 1–2 are Phase A (cheap, could kill the result);
threat 3 is Phase B (the positive scientific contribution, gated on surviving Phase A).

---

## 2. Central question & sub-questions

**Central:** Is there a genuine, model-faithful *information-flow* signal in attention that
predicts reasoning depth beyond everything explainable by graph shape and surface features?

- **Q1 (carrier):** Does **discord** add predictive power beyond a baseline containing *all*
  shape features (H1 persistence, Fiedler, classical graph statistics) and surface controls?
- **Q2 (confound):** Does the signal survive when sequence length and entity count are held
  fixed (length-matched items) or regressed out?
- **Q3 (faithfulness):** Does an **OV-circuit sheaf** (restriction maps from the head's true
  transport) carry the signal at least as well as — and ideally better than — the generic
  PCA sheaf?
- **Q4 (robustness):** Is the effect stable across stalk dimension d, sparsity top_k, and
  random seeds, with confidence intervals that exclude zero?

---

## 3. Phase A — Validity & de-confounding (Direction 1)

Cheap; reuses the Exp 8 pipeline and mostly the existing forward-pass sweep. **This phase
can falsify the headline.**

### A1 — Carrier decomposition (the decisive test)

Build a *strict shape baseline* and ask whether discord adds beyond it.

- **B_shape** = H1 persistence features + first-order controls + **Fiedler** + classical
  graph statistics (see A3). Everything that is a function of graph *shape* only.
- **Full** = B_shape + **discord** (and sheaf spectral gap, the other flow-dependent term).
- Report ΔR²(discord | B_shape), nested-F p, and partial Spearman(discord, hop | B_shape).
- Symmetric companion: ΔR²(Fiedler | H1 + controls) to quantify how much of Exp 8's original
  ΔR² was *just* Fiedler.

**Pre-registered decision (kill criterion):**
- **CONFIRMED-FLOW** — discord adds beyond B_shape (ΔR² ≥ 0.02 AND nested-F p < 0.05). The
  "flow beats shape" claim stands; proceed to Phase B.
- **SHAPE-IN-DISGUISE (kill)** — discord does *not* add beyond B_shape, but Fiedler did add
  beyond H1. We report honestly that Exp 8 Frame 1 was a *shape* effect (a graph statistic
  H1 missed), retract the "flow" framing in the WRITEUP, and the study pivots to
  characterizing *which* shape statistic and why. Phase B is not run.
- **NULL (kill)** — neither discord nor Fiedler survives the expanded baseline + larger n;
  Frame 1 does not replicate. Reported as a negative.

### A2 — Surface-complexity confound

- **Regress-out approach (cheap):** add `sequence_length` and `n_entities` to *every*
  baseline. The flow signal must add beyond these too.
- **Matched-design approach (strong):** construct **length- and entity-matched items** where
  reasoning hop varies but token count and number of named entities are held fixed (extend
  `build_items` with padding / filler clauses, analogous to the Spike's length-matched
  kinship). Re-run A1 on the matched set.
- Report A1's verdict separately on (i) raw items, (ii) length-controlled, (iii) matched
  subset. The headline only holds if it survives (ii) and (iii).

### A3 — Honest graph competitors

H1 is a weak strawman; a cheap classical statistic might match the sheaf. Add to B_shape and
test whether discord still adds beyond all of them:
- modularity (Louvain community structure), normalized-Laplacian spectral gap, global
  clustering coefficient, average shortest-path length, degree assortativity.
- Also run each competitor *in place of* the sheaf in the Exp 8 frame, to report a fair
  leaderboard: "what is the best single graph descriptor for hop-count, and does flow beat
  all of them."

### A4 — Robustness & power

- **Sensitivity:** d ∈ {2, 4, 8, 16}; top_k ∈ {4, 8, 16}; report ΔR²(discord | B_shape)
  surface over the grid (is the effect a knob artifact?).
- **Seeds & n:** ≥ 5 data seeds; raise n to ≈ 360–600 items; **bootstrap 95% CIs** on every
  reported ΔR² and AUC. Headline claims require CIs excluding zero.
- **Aggregation check:** mean vs max vs per-layer-block aggregation of head-level discord
  (Exp 8 used mean/max only).

---

## 4. Phase B — The faithful OV-circuit sheaf (Direction 2)

Run **only if Phase A returns CONFIRMED-FLOW.** This is the positive contribution: make the
restriction maps the model's *actual* transport.

### B1 — Construction

For head h, the **OV circuit** `W_OV = W_V W_O` is the linear map the head applies to the
residual stream of an attended token before writing it back. The information transported
along edge (j → i) is (up to the attention scalar) `x_j W_OV`. Construct the sheaf:
- Project `W_OV` into the per-item PCA subspace: `M = Pᵀ W_OV P` (P = PCA basis, d-dim).
- **Orthogonalize** `M` via polar decomposition `M = Q S` (or SVD `M = UΣVᵀ → Q = UVᵀ`) to
  get a valid orthogonal restriction map (handles attention's asymmetry the standard way:
  symmetrize the scalar weight, orthogonalize the transport map).
- Assemble the sheaf Laplacian and recompute **discord_OV** per (layer, head), aggregated
  per item like Exp 8.

### B2 — Comparison

In the same nested OLS / partial-correlation framework on hop-count (and the length-matched
set from A2):
- Does **discord_OV add beyond B_shape** (replicating A1 with the faithful feature)?
- Does **discord_OV add beyond discord_PCA** (the faithful map beats generic geometry)?
- Leaderboard: H1 vs discord_PCA vs discord_OV vs best classical competitor.

**Pre-registered decision:**
- **GREEN (faithful flow)** — discord_OV adds beyond B_shape AND beyond discord_PCA. The
  model's own transport carries reasoning-depth information beyond shape and beyond generic
  geometry. This is the study's strongest possible positive result.
- **PARTIAL** — discord_OV ≈ discord_PCA (both add beyond shape, but the OV map is not
  better than PCA): flow is real but not specifically the OV circuit; report as such.
- **RED** — discord_OV does not add beyond B_shape although discord_PCA did: the PCA signal
  was geometric, not computational; report the tension honestly.

### B3 — Per-edge interpretability (descriptive)

Does the OV-sheaf's per-edge consistency localize onto the *actual reasoning structure*? For
kinship/ordering items, test whether the chain-link token edges (e.g. A→B→C) have
systematically lower discord (more consistent flow) than length-matched random edges
(Mann-Whitney), and visualize a few discord maps. Descriptive, not a gating verdict.

---

## 5. Phase C — Conditional extensions (Directions 3–4)

Run **only if Phase B returns GREEN/PARTIAL.** Scoped here, pre-registered later.
- **Scale (Direction 3):** repeat the A1 + B2 leaderboard on a larger model (Qwen2.5-1.5B;
  larger only if a memory-safe path exists — Exp 6 showed scale changes conclusions) and on
  a **real** multi-hop task (IOI, or a small GSM8K-style slice) instead of synthetic kinship.
- **Mechanism (Direction 4):** causal test — does intervening on high-discord_OV edges
  (reuse the Exp 6 directed/edge-exact ablation) damage reasoning more than budget-matched
  controls? Closes the loop from *correlate* to *cause* for the flow signal.

---

## 6. Shared methods & integrity

- **Model/data:** Qwen2.5-0.5B-Instruct (continuity); mixed-hop kinship + ordering via
  `build_items`, extended for length/entity matching (A2). CPU-only, float32, eager,
  `torch.set_num_threads(2)`, one model, gc per item — the gentle path that has been stable
  since Exp 1.
- **One sweep:** a single forward pass per item captures attentions, hidden states, and the
  static `W_OV` per head (weights, not activations) — so PCA-sheaf, OV-sheaf, H1, classical
  graph stats, confidence, and surface controls all come from one pass.
- **Statistics:** nested OLS (ΔR², nested-F), partial Spearman, bootstrap 95% CIs, ≥5 seeds.
  Pre-registered thresholds: ΔR² floor 0.02, α = 0.05.
- **Provenance discipline (non-negotiable, project rule):** every number is computed into a
  JSON source-of-truth (`results/exp9_*.json`), rendered into FINDINGS mechanically, and
  verified by `json.load` — never hand-typed, never read from terminal output. Verdicts are
  machine-checked fields. (This rule exists because terminal output has twice fed me
  fabricated numbers in this project; both were caught against the JSON.)

---

## 7. Pre-registered verdicts (summary table)

| Phase | Test | GREEN | KILL / negative |
|---|---|---|---|
| A1 | discord beyond shape baseline | ΔR² ≥ 0.02, F p < 0.05 | SHAPE-IN-DISGUISE / NULL |
| A2 | survives length/entity control | holds on matched set | confound explains it |
| A3 | beats classical graph competitors | discord still adds | a cheap stat matches it |
| A4 | stable across d/top_k/seeds | CI excludes 0 | knob/seed artifact |
| B2 | OV-sheaf beyond shape & PCA | both hold | PARTIAL / RED |
| B3 | discord localizes to chain edges | MW p < 0.05 | descriptive only |

**The study succeeds in the strong sense iff A1–A4 confirm AND B2 is GREEN.** It produces a
valuable *negative* (and an honest WRITEUP correction) if A1 returns SHAPE-IN-DISGUISE. Both
outcomes are publishable within the project's "say what's true" frame.

---

## 8. Risks, and why the design absorbs them

- **Risk: the headline dies in Phase A.** Mitigation: that *is* the point — A1 is designed
  to kill a false claim cheaply before we invest in Phase B. A retraction-with-explanation
  is a good outcome, not a failure.
- **Risk: OV-circuit extraction is fiddly** (GQA/attention-variant bookkeeping, projecting
  W_OV correctly). Mitigation: unit-test the OV map on a toy head (known transport →
  recovered restriction map); validate that identity transport recovers the PCA-sheaf limit.
- **Risk: degeneracy** (harmonic dim was constant in Exp 8). Mitigation: pre-commit to
  dropping zero-variance features and report them as uninformative rather than forcing them
  into models.
- **Risk: small-scale only.** Mitigation: Phase C scales *if* the signal is real; we do not
  over-claim generality from 0.5B.

---

## 9. Deliverables & sequencing

1. **Phase A** — `src/run_exp9a.py`, `src/compute_stats_exp9a.py` (decomposition + confound
   + competitors + sensitivity), `results/exp9a_stats.json`, `FINDINGS_exp9a.md`. Decision
   gate. *(Days; mostly reuses Exp 8 data.)*
2. **Phase B** (if CONFIRMED-FLOW) — OV-circuit sheaf in `src/sheaf.py` (new
   `ov_restriction_maps`), `src/run_exp9b.py`, stats, findings. Decision gate.
3. **Phase C** (if GREEN/PARTIAL) — separate pre-registered specs for scale and causal test.
4. **WRITEUP** — new Result J (or a correction to Result I if Phase A kills the flow claim),
   rendered mechanically.

Each phase is independently committable and produces a working, verifiable artifact. We stop
at any gate that returns a kill verdict.

---

## 10. Why this is the right next bet

Across nine experiments, the sheaf-flow result (Exp 8 Frame 1) is the **only place the signal
is growing rather than hitting a ceiling** — every other "is topology *uniquely* useful" test
plateaued (pruning NULL, failure-prediction stuck at the confidence ceiling, causal signal
budget-sensitive). Frame 1 is also the **least scrutinized** of our positives. This study
either converts it into the project's first robust, model-faithful, causal-capable
"information-flow" finding — or it honestly retires the flow framing. Either way it resolves
the most consequential open question we have, cheaply and falsifiably.
