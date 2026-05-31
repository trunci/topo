"""Experiment 6c: causal test on a medium model (gpt2-medium, 355M, 24L x 16H).

More induction heads -> more statistical power and a stronger generalization claim.
Uses the SAME directed/edge-exact/cumulative instrument as Exp 6, at the adequate
budget top_k=24 established by the Exp 6b sweep (top_k=8 starved gpt2). CPU-only,
float32, eager, torch.set_num_threads(2), one model -- memory-safe.

Output: results/exp6_medium_ablation.parquet (same schema as exp6_ablation), stats
via compute_stats_exp6 -> results/exp6_medium_stats.json.
"""
from __future__ import annotations

from src.run_exp6 import run

if __name__ == "__main__":
    run(out="results/exp6_medium_ablation.parquet",
        models=["gpt2-medium"], top_k=24)
