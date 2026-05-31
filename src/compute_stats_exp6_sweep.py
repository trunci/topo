"""Per-top_k statistics for the Exp 6b gpt2 budget sweep.

Reuses the pre-registered Exp 6 per-model verdict logic (_model_stats) on each
top_k slice of results/exp6_sweep.parquet. Writes results/exp6_sweep_stats.json:
{ "model", "seq_len", "gap_tolerance", "alpha", "by_top_k": { "<tk>": {...} } }.
No numbers are hand-typed; findings render from this JSON.
"""
from __future__ import annotations

import json

import pandas as pd

from src.compute_stats_exp6 import _model_stats, ALPHA


def compute_stats(sweep_path="results/exp6_sweep.parquet",
                  out_path="results/exp6_sweep_stats.json", seq_len=25, gap_tol=2):
    df = pd.read_parquet(sweep_path)
    by_top_k = {}
    for tk, sub in df.groupby("top_k"):
        by_top_k[str(int(tk))] = _model_stats(sub)
    stats = {
        "model": str(df["model"].iloc[0]) if len(df) else "gpt2",
        "seq_len": seq_len,
        "gap_tolerance": gap_tol,
        "alpha": ALPHA,
        "by_top_k": by_top_k,
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    compute_stats()
