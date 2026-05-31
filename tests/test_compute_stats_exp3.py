import json
import numpy as np
import pandas as pd
from src.compute_stats_exp3 import compute_stats


def _synthetic(path, coupled):
    """144-head synthetic parquet.

    coupled=True : induction depends on h1 ON TOP of the first-order controls,
                   so H1 should add real predictive power (-> GREEN).
    coupled=False: induction is fully explained by the first-order controls plus
                   noise; h1 is independent noise (-> RED).
    """
    rng = np.random.default_rng(0)
    dist = rng.random(144)
    off = rng.random(144)
    ent = rng.random(144)
    if coupled:
        h1 = rng.random(144)
        ind = 0.3 * dist + 0.2 * off + 0.8 * h1 + rng.normal(0, 0.02, 144)
    else:
        h1 = rng.random(144)  # independent of induction
        ind = 0.5 * dist + 0.3 * off + 0.4 * ent + rng.normal(0, 0.02, 144)
    df = pd.DataFrame({
        "layer": np.repeat(np.arange(12), 12),
        "head": np.tile(np.arange(12), 12),
        "induction_score": ind, "h1_persistence": h1,
        "attn_distance": dist, "offdiag_mass": off, "attn_entropy": ent,
    })
    df.to_parquet(path)


def test_keys_and_disk_match(tmp_path):
    pq = tmp_path / "exp3.parquet"; out = tmp_path / "exp3_stats.json"
    _synthetic(pq, coupled=True)
    stats = compute_stats(str(pq), str(out))
    for k in ("baseline_r2", "full_r2", "delta_r2", "f_stat", "f_pvalue",
              "h1_coef", "h1_coef_p", "partial_spearman_rho",
              "partial_spearman_p", "raw_spearman_rho", "raw_spearman_p",
              "verdict"):
        assert k in stats
    assert json.loads(out.read_text()) == stats


def test_coupled_data_gives_green(tmp_path):
    pq = tmp_path / "exp3.parquet"; out = tmp_path / "exp3_stats.json"
    _synthetic(pq, coupled=True)
    stats = compute_stats(str(pq), str(out))
    assert stats["delta_r2"] >= 0.02
    assert stats["f_pvalue"] < 0.05
    assert stats["verdict"].startswith("GREEN")


def test_independent_data_gives_red(tmp_path):
    pq = tmp_path / "exp3.parquet"; out = tmp_path / "exp3_stats.json"
    _synthetic(pq, coupled=False)
    stats = compute_stats(str(pq), str(out))
    assert stats["verdict"].startswith("RED")
