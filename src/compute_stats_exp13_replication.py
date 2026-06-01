"""Replication statistics for Experiment 13 -> results/exp13_replication_stats.json.

Pools results/exp13_features.parquet (seed=0) with results/exp13_s1_features.parquet
and results/exp13_s2_features.parquet to get 3× the original item count.

Reports per-seed and combined verdicts so we can check whether the strong result
(hop=5 AUC=1.000) replicates or was a lucky split.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.compute_stats_exp13 import _verdict

def _f(x, nd=3):
    return "n/a" if x is None else f"{float(x):.{nd}f}"


def compute_stats(
    seed0="results/exp13_features.parquet",
    seed1="results/exp13_s1_features.parquet",
    seed2="results/exp13_s2_features.parquet",
    out_path="results/exp13_replication_stats.json",
):
    dfs = {}
    for label, path in [("seed0", seed0), ("seed1", seed1), ("seed2", seed2)]:
        dfs[label] = pd.read_parquet(path)

    combined = pd.concat(list(dfs.values()), ignore_index=True)

    def _seed_result(df, label):
        h4 = df[df["hop"] == 4]
        h5 = df[df["hop"] == 5]
        return {
            "n_items": int(len(df)),
            "overall_accuracy": float(df["is_correct"].mean()),
            "acc_by_hop": {
                "4": float(h4["is_correct"].mean()) if len(h4) else None,
                "5": float(h5["is_correct"].mean()) if len(h5) else None,
            },
            "hop4_verdict": _verdict(h4)["verdict"] if len(h4) > 0 else "n/a",
            "hop5_verdict": _verdict(h5)["verdict"] if len(h5) > 0 else "n/a",
            "all_verdict": _verdict(df)["verdict"],
            "hop4_topo_auc": (
                _verdict(h4)["auc"]["topology_only"]["mean_auc"]
                if len(h4) > 0 else None
            ),
            "hop5_topo_auc": (
                _verdict(h5)["auc"]["topology_only"]["mean_auc"]
                if len(h5) > 0 else None
            ),
            "hop4_conf_auc": (
                _verdict(h4)["auc"]["confidence_only"]["mean_auc"]
                if len(h4) > 0 else None
            ),
            "hop5_conf_auc": (
                _verdict(h5)["auc"]["confidence_only"]["mean_auc"]
                if len(h5) > 0 else None
            ),
        }

    combined_h4 = combined[combined["hop"] == 4]
    combined_h5 = combined[combined["hop"] == 5]

    stats = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "seeds": [0, 1, 2],
        "n_per_seed": 120,
        "per_seed": {label: _seed_result(df, label) for label, df in dfs.items()},
        "combined": {
            "n_items": int(len(combined)),
            "overall_accuracy": float(combined["is_correct"].mean()),
            "acc_by_hop": {
                "4": float(combined_h4["is_correct"].mean()),
                "5": float(combined_h5["is_correct"].mean()),
            },
            "all_items": _verdict(combined),
            "hop4_only": _verdict(combined_h4),
            "hop5_only": _verdict(combined_h5),
        },
    }

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[replication] combined n={len(combined)}, "
          f"acc={combined['is_correct'].mean():.3f}")
    print("[replication] per-seed hop=4/5 verdicts:")
    for label, r in stats["per_seed"].items():
        print(f"  {label}: hop4={r['hop4_verdict']} "
              f"(topo={r['hop4_topo_auc']:.3f}/conf={r['hop4_conf_auc']:.3f})  "
              f"hop5={r['hop5_verdict']} "
              f"(topo={r['hop5_topo_auc']:.3f}/conf={r['hop5_conf_auc']:.3f})")
    print(f"[replication] combined hop4={stats['combined']['hop4_only']['verdict']} "
          f"hop5={stats['combined']['hop5_only']['verdict']}")
    return stats


if __name__ == "__main__":
    compute_stats()
