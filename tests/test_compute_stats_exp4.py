import json
import numpy as np
import pandas as pd
from src.compute_stats_exp4 import compute_stats


def _make_cell(rng, n, coupled, score_col):
    """Build one model's worth of head rows for a single target circuit.

    coupled=True : the score depends on h1 on top of the controls -> GREEN.
    coupled=False: the score is explained by controls + noise; h1 independent -> RED.
    The two non-target score columns are filled with controls-only noise so they
    are not GREEN (keeps the synthetic overall verdict deterministic).
    """
    dist = rng.random(n)
    off = rng.random(n)
    ent = rng.random(n)
    h1 = rng.random(n)
    base = 0.5 * dist + 0.3 * off + 0.4 * ent
    if coupled:
        target = base + 0.9 * h1 + rng.normal(0, 0.02, n)
    else:
        target = base + rng.normal(0, 0.02, n)
    cols = {
        "induction_score": base + rng.normal(0, 0.02, n),
        "prev_token_score": base + rng.normal(0, 0.02, n),
        "dup_token_score": base + rng.normal(0, 0.02, n),
        "h1_persistence": h1, "attn_distance": dist,
        "offdiag_mass": off, "attn_entropy": ent,
    }
    cols[score_col] = target
    return cols


def _synthetic(path):
    """gpt2 has a coupled induction cell (GREEN); distilgpt2 is independent (RED).

    Other cells are controls-only -> RED, so exactly 1 cell is GREEN overall.
    """
    rng = np.random.default_rng(0)
    rows = []
    g = _make_cell(rng, 144, coupled=True, score_col="induction_score")
    for i in range(144):
        rows.append({"model": "gpt2", "layer": i // 12, "head": i % 12,
                     **{k: v[i] for k, v in g.items()}})
    d = _make_cell(rng, 72, coupled=False, score_col="induction_score")
    for i in range(72):
        rows.append({"model": "distilgpt2", "layer": i // 12, "head": i % 12,
                     **{k: v[i] for k, v in d.items()}})
    pd.DataFrame(rows).to_parquet(path)


def test_keys_and_disk_match(tmp_path):
    pq = tmp_path / "exp4.parquet"; out = tmp_path / "exp4_stats.json"
    _synthetic(pq)
    stats = compute_stats(str(pq), str(out))
    assert stats["n_cells"] == 6
    assert len(stats["cells"]) == 6
    for c in stats["cells"]:
        for k in ("model", "circuit", "n_heads", "baseline_r2", "full_r2",
                  "delta_r2", "f_stat", "f_pvalue", "partial_spearman_rho",
                  "partial_spearman_p", "raw_spearman_rho", "cell_verdict"):
            assert k in c
    assert json.loads(out.read_text()) == stats


def test_coupled_cell_green_independent_cell_red(tmp_path):
    pq = tmp_path / "exp4.parquet"; out = tmp_path / "exp4_stats.json"
    _synthetic(pq)
    stats = compute_stats(str(pq), str(out))
    by = {(c["model"], c["circuit"]): c for c in stats["cells"]}
    assert by[("gpt2", "induction")]["cell_verdict"] == "GREEN"
    assert by[("distilgpt2", "induction")]["cell_verdict"] == "RED"


def test_overall_verdict_one_green_is_does_not_generalize(tmp_path):
    pq = tmp_path / "exp4.parquet"; out = tmp_path / "exp4_stats.json"
    _synthetic(pq)
    stats = compute_stats(str(pq), str(out))
    assert stats["n_green"] == 1
    assert stats["overall_verdict"].startswith("DOES-NOT-GENERALIZE")
