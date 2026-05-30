import json
import numpy as np
import pandas as pd
from src.compute_stats_exp2 import compute_stats


def _synthetic(path, coupled=True):
    rng = np.random.default_rng(0)
    ind = rng.random(144)
    if coupled:
        h1 = ind * 2.0 + rng.normal(0, 0.05, 144)
    else:
        h1 = rng.random(144)
    dist = rng.random(144)
    df = pd.DataFrame({
        "layer": np.repeat(np.arange(12), 12),
        "head": np.tile(np.arange(12), 12),
        "induction_score": ind, "h1_persistence": h1,
        "attn_distance": dist, "offdiag_mass": rng.random(144),
    })
    df.to_parquet(path)


def test_coupled_data_gives_green(tmp_path):
    pq = tmp_path / "exp2.parquet"; out = tmp_path / "exp2_stats.json"
    _synthetic(pq, coupled=True)
    stats = compute_stats(str(pq), str(out))
    assert stats["spearman_rho"] > 0
    assert stats["spearman_p"] < 0.05
    assert stats["mannwhitney_p"] < 0.05
    assert stats["verdict"].startswith("GREEN")
    assert json.loads(out.read_text()) == stats


def test_uncoupled_data_gives_red(tmp_path):
    pq = tmp_path / "exp2.parquet"; out = tmp_path / "exp2_stats.json"
    _synthetic(pq, coupled=False)
    stats = compute_stats(str(pq), str(out))
    assert stats["verdict"].startswith("RED")
