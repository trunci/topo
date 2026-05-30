# Spike progress (working notes — not the deliverable)

Branch: `spike/topology-attention`. Plan: `docs/superpowers/plans/2026-05-30-topology-attention-spike.md`.
Executing via subagent-driven-development (implementer → independent verify per task).

## Status by task
- Task 0 (scaffold + deps): DONE, committed. uv project, all deps, MPS available.
- Task 1 (`src/topology.py`, TDD): DONE, committed `1a1c06f`. 7 tests pass independently.
- Task 2 (`src/data_gen.py`, TDD): DONE, committed `aaba248`. 5 tests pass independently.
- Task 3 (`src/attn_extract.py`): DONE, committed `e823251`. 2 slow tests pass.
  Attention shape confirmed: **24 layers × 14 heads × n×n**. Qwen2.5-0.5B-Instruct, eager, float32, MPS.
  NOTE: implementer added `[tool.pytest.ini_options]` slow marker to pyproject — verify it's committed
  (there were uncommitted `pyproject.toml`/`uv.lock` mods; reconcile before Task 4).
- Task 4 (`src/run_spike.py`): NOT STARTED — next.
- Task 5 (`src/analyze.py`): not started.
- Task 6 (full run + FINDINGS.md): not started.

## KEY RESEARCH FINDING from probe (scratch_probe.py, deleted)
Ran Qwen2.5-0.5B on the minimal pairs:
- **Free-form/plain chat prompt: ~88% "correct"** by substring match.
- Constrained ("answer with only the name"): ~38% — terse answers hurt.
- **CAVEAT (substring leak):** ordering-family gold = tallest = first-named `a`, which always
  appears in any restatement of the premise → ordering "correctness" is inflated/unreliable.
  Kinship is the genuinely clean minimal pair (1-hop vs 2-hop differ by ONE word: father→grandfather).
- The committed `is_correct` uses the plain-chat prompt (the good 88% regime). Good.

## IMPLICATION for the spike (decided)
- PRIMARY test (paired Wilcoxon of H1 topology features, 2hop vs 1hop, per head) does NOT depend on
  model correctness — attention is extracted over the prompt forward pass. Proceed regardless.
- Treat "restrict to solved items" as a weak secondary only; flag the substring-leak caveat in FINDINGS.
- In FINDINGS, also stratify by family (kinship = clean minimal pair) even though plan's analyze.py pools.

## Gotchas learned
- Prefix shell with `unset VIRTUAL_ENV;` to silence uv venv warning.
- No `timeout` cmd on this mac. Don't chain `sleep`. Avoid huge parallel tool batches (caused cascades).
- gudhi: MUST use `compute_persistence(persistence_dim_max=True)` or top-dim H1 dropped.
