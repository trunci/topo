# Experiment 6 — Stronger causal test of induction cycle edges (resolving Exp 5 E1)

**Date:** 2026-05-31
**Status:** pre-registered (design approved before any results)

## Motivation

Exp 5's E1 causal test returned RED but **underpowered**: median "damage" (loss
increase) was *negative* for all conditions (cycle -0.0270, magnitude -0.0300,
random -0.0227), i.e. the ablation barely perturbed — sometimes *helped* — induction
behavior. Three design flaws made it weak:

1. **Undirected ablation.** `keepsets_to_bias` symmetrizes (ablates both `(i,j)` and
   `(j,i)`). The functional induction edge is *directed*: query at the second-copy
   position attends *back* to the token after the first occurrence (`q > k`, gap ≈ S).
2. **Common-mode noise.** Each condition kept only a per-head keep-set and ablated
   *everything else* — including all non-candidate (near-zero) edges — in every
   condition. The cycle-vs-control signal was buried under that shared perturbation.
3. **Too few edges, per-head budget.** Small per-head cycle sets; induction is
   distributed across ~10 heads, so leaving most heads' induction edge intact left
   plenty of redundancy.

Exp 5's E2 (descriptive) is GREEN: induction-head critical-cycle edges concentrate at
gap ≈ S (the copy distance). So the cycles *are* the copy edges. Exp 6 asks the
**causal** question with an instrument strong enough to answer it.

## Question

When we ablate the **directed, gap-≈S** critical-cycle edges of the induction heads —
**cumulatively, across all induction heads at once** — does induction behavior (model
loss on second-copy tokens) degrade **more than** budget-matched magnitude- or
random-selected directed edge sets?

## Design

- **Models:** `gpt2`, `distilgpt2` (the two induction-GREEN models from Exp 4).
  CPU-only, float32, eager, `torch.set_num_threads(2)`, one model in memory at a time.
- **Substrate:** repeated-random sequences `[prefix(5); block(S); block(S)]`, S = 25,
  same as Exp 2–5. `n_seqs = 12` (up from 8 for paired-test power).
- **Induction heads:** top `K_HEADS = 10` by mean `induction_score` over the batch
  (same selector as Exp 5).
- **Readout:** `second_copy_loss` — mean next-token cross-entropy on second-copy
  positions (identical to Exp 5, so results are directly comparable).
- **Damage(condition) = loss(condition) − loss(unmasked)**, paired across sequences.

### The new intervention — directed, edge-exact ablation

New helper `ablation_to_bias(ablate_sets, n, H, L, device)` in `pruned_forward.py`:
- Starts from **all-zero** bias (keep everything).
- For each `(layer, head)` and each *directed* `(q, k)` in its ablate-set with
  `q != k`, sets bias `[h, q, k] = -inf`.
- Diagonal never ablated.

This removes **exactly** the named directed edges and nothing else — the clean
contrast E1 lacked.

### Conditions (budget-matched, per induction head)

For each induction head `(l, h)` and sequence:
- Compute undirected critical cycle edges `cyc = critical_cycle_edges(A[l,h], top_k=8)`.
- Keep only gap-≈S edges: `{e in cyc : |i-j| within tol=2 of S}`. Direct them as
  `(q, k) = (max(i,j), min(i,j))` (later attends earlier). Call this set `D_cyc`;
  let `m = |D_cyc|` (the per-head budget).
- **cycle:** ablate `D_cyc`.
- **magnitude:** ablate the `m` highest-weight *directed causal* edges `(q>k)` by raw
  `A[q,k]` (any gap).
- **random:** ablate `m` random directed causal edges `(q>k)` (seeded, deterministic).

Cumulative: the per-head ablate-sets are combined into ONE bias and applied to all
induction heads simultaneously.

### Outputs (single source of truth = parquet)

- `results/exp6_ablation.parquet`: one row per `(model, seq, condition)` with
  `second_copy_loss` and the total ablated-edge count `K`.

### Statistics (`src/compute_stats_exp6.py` → `results/exp6_stats.json`)

Per model, paired one-sided Wilcoxon signed-rank on per-sequence damage:
- `cycle_vs_random` (cycle damage greater) and `cycle_vs_magnitude`.
- Report median damage per condition and `cycle_damage_positive = median > 0`.

**Per-model verdict:**
- **GREEN** — `cycle_damage_positive` AND `cycle_vs_random` p < 0.05.
  Topology-selected edges are causally functional for induction, beyond a null.
- **RED** — `cycle_damage_positive` but cycle not > random (edges matter, but no more
  than random selection — topology not special).
- **INCONCLUSIVE** — NOT `cycle_damage_positive` (still underpowered; should not happen
  with the directed edge-exact instrument, but reported honestly if it does).

`cycle_vs_magnitude` is reported as the specificity sub-question (does topology beat
the raw-weight heuristic), not part of the GREEN/RED gate.

## Findings + writeup

- `src/write_findings_exp6.py` → `FINDINGS_exp6.md`, generated mechanically from the
  JSON (no hand-typed numbers — the project's hard rule).
- Fold into `WRITEUP.md` (`src/write_writeup.py`) as Result G once results land.

## Honest caveats (pre-committed)

- gap-≈S edges *are* the induction offset by construction; the scientific content is
  the **contrast vs magnitude/random at equal budget**, not "ablating the induction
  edge breaks induction" (which is expected). cycle≈magnitude is the likely outcome
  (both grab the strong gap-S edge); cycle≫random is the real causal claim.
- Still small CPU runs (≤144 heads, n_seqs=12). Directions, not exact magnitudes.
