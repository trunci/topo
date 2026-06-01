"""Run Exp 13 for additional seeds (1 and 2) to replicate the deep-hop GREEN result.

Uses the same parameters as run_exp13.py (hop=4,5; n_per_family=30; Qwen2.5-0.5B)
but different random seeds for item generation. Outputs to per-seed parquets;
compute_stats_exp13_replication pools all three.
"""
from src.run_exp13 import run as _run

for seed in [1, 2]:
    print(f"\n{'='*60}\n[replication] seed={seed}\n{'='*60}", flush=True)
    _run(out=f"results/exp13_s{seed}_features.parquet", seed=seed)
