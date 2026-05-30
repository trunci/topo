import json
import numpy as np
import pandas as pd
from src.compute_stats_exp1 import compute_stats


def _toy_df():
    rows = []
    # wikitext: dmt clearly better (lower ppl) than magnitude
    for i in range(8):
        rows.append({"example_id": f"wiki-{i}", "domain": "wikitext", "method": "dmt",
                     "loss": 2.0, "ppl": 7.389, "correct": float("nan"),
                     "mean_sparsity": 0.5, "seq_len": 64})
        rows.append({"example_id": f"wiki-{i}", "domain": "wikitext", "method": "magnitude",
                     "loss": 2.5, "ppl": 12.18, "correct": float("nan"),
                     "mean_sparsity": 0.5, "seq_len": 64})
        rows.append({"example_id": f"wiki-{i}", "domain": "wikitext", "method": "random",
                     "loss": 3.0, "ppl": 20.09, "correct": float("nan"),
                     "mean_sparsity": 0.5, "seq_len": 64})
        rows.append({"example_id": f"wiki-{i}", "domain": "wikitext", "method": "window",
                     "loss": 2.8, "ppl": 16.44, "correct": float("nan"),
                     "mean_sparsity": 0.5, "seq_len": 64})
        rows.append({"example_id": f"wiki-{i}", "domain": "wikitext", "method": "unpruned",
                     "loss": 1.5, "ppl": 4.48, "correct": float("nan"),
                     "mean_sparsity": 1.0, "seq_len": 64})
    return pd.DataFrame(rows)


def test_compute_stats_reports_per_method_ppl_and_verdict(tmp_path):
    df = _toy_df()
    pq = tmp_path / "exp1.parquet"
    df.to_parquet(pq)
    out = tmp_path / "exp1_stats.json"
    stats = compute_stats(str(pq), str(out))
    # per-method mean ppl present
    assert "wikitext_ppl_by_method" in stats
    assert abs(stats["wikitext_ppl_by_method"]["dmt"] - 7.389) < 1e-2
    # dmt beats magnitude on ppl -> verdict notes a dmt win on ppl
    assert stats["dmt_beats_magnitude_ppl"] is True
    # json written and matches returned dict
    on_disk = json.loads(out.read_text())
    assert on_disk == stats


def test_compute_stats_handles_missing_domains(tmp_path):
    # only kinship rows
    rows = [{"example_id": "kin-1", "domain": "kinship", "method": m, "loss": 0.5,
             "ppl": float("nan"), "correct": 1.0, "mean_sparsity": 0.5, "seq_len": 40}
            for m in ["unpruned", "dmt", "magnitude", "random", "window"]]
    df = pd.DataFrame(rows)
    pq = tmp_path / "exp1.parquet"
    df.to_parquet(pq)
    stats = compute_stats(str(pq), str(tmp_path / "out.json"))
    assert "kinship_accuracy_by_method" in stats
    assert stats["kinship_accuracy_by_method"]["dmt"] == 1.0
