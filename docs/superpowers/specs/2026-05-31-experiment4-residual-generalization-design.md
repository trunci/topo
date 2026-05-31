# Design: Experiment 4 — Does the residual-information result generalize?

**Date:** 2026-05-31
**Status:** Pre-registered (criteria fixed before any real-data compute).
**Parent:** WRITEUP.md "Follow-up direction 1". Builds on Experiment 3, which found that
H1 persistence adds significant predictive power for **induction** score beyond first-order
attention statistics in **GPT-2 small** (delta-R2=0.074, nested-F p=0.00044, partial
Spearman rho=0.372). Exp 3 was one circuit, one model, 144 heads. This experiment tests
whether that residual (non-redundant) signal **generalizes** across more circuits and a
second model — or whether it was an induction-in-GPT2 fluke.

## Question

Across multiple (model, circuit) combinations, does H1 attention-graph persistence carry
predictive signal for the circuit's score **beyond** first-order attention statistics
(attention distance, off-diagonal mass, entropy)?

## Design — a grid of residual-information tests

**Models (CPU, eager, float32):**
- `gpt2` (124M, 12x12 = 144 heads) — same as Exp 2/3, the reference.
- `distilgpt2` (82M, 6x12 = 72 heads) — a distinct, smaller architecture for model diversity.

(Both are tiny and CPU-safe. A larger model is explicitly out of scope — earlier MPS/large
runs crashed the machine; do NOT use MPS or models bigger than gpt2.)

**Circuits (each detected independently of topology, from attention on repeated-random
sequences `[prefix; block S; block S]`):**
- **induction** — query i in the 2nd copy attends to key `i - S + 1` (token after the
  first-copy match). Reuse `induction.induction_score`.
- **previous-token** — query i attends to key `i - 1`. New: `previous_token_score(A)`.
- **duplicate-token** — query i in the 2nd copy attends to key `i - S` (the identical token
  in the first copy). New: `duplicate_token_score(A, seq_len, prefix_len)`.

That is **2 models x 3 circuits = 6 residual-information tests**, each over that model's heads.

## Per-cell method (reuse Exp 3 exactly)

For each (model, circuit): nested OLS predicting the circuit score from the heads' features.
- Baseline: score ~ [attn_distance, offdiag_mass, attn_entropy].
- Full: + h1_persistence.
- Report baseline R2, full R2, delta-R2, nested-model F-test (F, p), H1 coefficient + p,
  partial Spearman(H1, score | controls) and raw Spearman, n_heads.
Reuse the statistics from `src/compute_stats_exp3.py` (factor its core into a shared helper
that takes a target column + controls, so Exp 3 behavior is preserved).

## Pre-registered verdict rule (fixed here, before running)

Identical floor to Exp 3, applied per cell:
- A cell is **GREEN** iff `delta_r2 >= 0.02` AND nested-F `p < 0.05` (H1 adds signal).
- Else **RED** (H1 redundant with first-order stats for that model+circuit).

**Overall verdict (pre-registered):**
- **STRONG-GENERALIZE** iff >= 5 of the 6 cells are GREEN.
- **PARTIAL-GENERALIZE** iff 2-4 cells GREEN (note which circuits/models drive it).
- **DOES-NOT-GENERALIZE** iff <= 1 cell GREEN (Exp 3 was likely induction-in-GPT2 specific).

Report whatever the grid says, including the induction-GPT2 cell as a replication check of
Exp 3 (it should reproduce delta-R2 ~ 0.074; if it does not, flag a pipeline bug).

## Components (small, heavy reuse)

- `src/circuits.py` — NEW. `previous_token_score(A)`, `duplicate_token_score(A, seq_len,
  prefix_len)`. Pure numpy over an [n,n] attention matrix. TDD against synthetic matrices
  with the exact pattern (score ~1) vs uniform (low).
- Reuse `src/induction.py` (`make_repeat_batch`, `induction_score`, `attention_distance`,
  `offdiag_mass`), `src/residual.py` (`attention_entropy`), `src/topology.py`
  (`h1_features`), `src/attn_extract.py` (`load_named`).
- `src/run_exp4.py` — NEW. For each model: load (CPU), run one repeat batch, per head record
  model, layer, head, induction_score, prev_token_score, dup_token_score, h1_persistence,
  attn_distance, offdiag_mass, attn_entropy -> `results/exp4.parquet`. One model in memory
  at a time (del + gc between models). Flushed per-(model) progress prints.
- `src/compute_stats_exp4.py` — NEW. Run the 6 per-cell residual tests -> the grid +
  overall verdict -> `results/exp4_stats.json` (single source of truth). Factor/reuse the
  Exp 3 nested-OLS helper. TDD on a synthetic parquet (coupled cell -> GREEN, independent
  cell -> RED, on-disk JSON == returned dict).
- `src/write_findings_exp4.py` — NEW. Emit `FINDINGS_exp4.md` mechanically from the JSON
  (a grid table + overall verdict). No hand-typed numbers.

## Testing strategy (TDD)

- `circuits.py`: previous-token pattern (mass on `i-1`) -> prev_token_score ~ 1, uniform ->
  low; duplicate pattern (mass on `i-S` in 2nd copy) -> dup_token_score ~ 1, uniform -> low.
- `compute_stats_exp4.py`: synthetic parquet where one (model,circuit) cell is constructed so
  H1 tracks the score beyond controls (-> GREEN) and another is independent (-> RED); assert
  the per-cell verdicts and the overall verdict; assert on-disk JSON equals the returned dict.
- `run_exp4.py`: a slow smoke test (tiny batch) asserting the parquet has the expected columns
  and both model names appear.

## Safety & integrity (HARD constraints)

- **CPU only, one model in memory at a time, `torch.set_num_threads(2)`, modest batch
  (n_seqs<=8, seq_len<=25).** No MPS. No parallel heavy jobs. (Two machine crashes earlier
  this session came from MPS/parallel/large runs.)
- **No hand-typed numbers anywhere.** Every reported value is computed into
  `results/exp4_stats.json` and emitted into `FINDINGS_exp4.md` mechanically; verify the
  findings regenerate byte-identically before committing. Commit messages must not assert
  numeric results from memory — state the verdict qualitatively and point to the JSON.
- Pre-registered rule above is fixed; the overall verdict is a machine-checked JSON field.
- TDD, frequent small commits, co-author trailer on each commit. `uv run` for everything;
  prefix model-loading commands with `set -a; [ -f .env ] && . ./.env; set +a; unset VIRTUAL_ENV;`.

## Deliverables

`results/exp4.parquet`, `results/exp4_stats.json`, `FINDINGS_exp4.md` (grid + overall
verdict, machine-generated), all committed; fast tests passing. A final honest summary
stating the per-cell grid and the overall generalization verdict as they appear in the JSON.
