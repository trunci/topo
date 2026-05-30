"""Experiment 1 analysis: plots and a readable summary, numbers from compute_stats_exp1."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.compute_stats_exp1 import compute_stats, METHODS


def main(in_path="results/exp1.parquet"):
    df = pd.read_parquet(in_path)
    stats = compute_stats(in_path)

    # Plot 1: mean PPL by method (wikitext)
    wiki = df[df.domain == "wikitext"]
    if len(wiki):
        means = wiki.groupby("method")["ppl"].mean().reindex(METHODS)
        plt.figure(figsize=(7, 4))
        plt.bar(range(len(means)), means.values)
        plt.xticks(range(len(means)), means.index, rotation=20)
        plt.ylabel("mean perplexity"); plt.title("WikiText-2 PPL by pruning method")
        plt.tight_layout(); plt.savefig("results/exp1_ppl_by_method.png", dpi=120)
        plt.close()

        # Plot 2: per-example DMT vs magnitude loss scatter
        d = df[df.domain == "wikitext"]
        pdmt = d[d.method == "dmt"].set_index("example_id")["loss"]
        pmag = d[d.method == "magnitude"].set_index("example_id")["loss"]
        common = pdmt.index.intersection(pmag.index)
        plt.figure(figsize=(5, 5))
        plt.scatter(pmag.loc[common], pdmt.loc[common], alpha=0.6)
        lo = float(min(pmag.min(), pdmt.min())); hi = float(max(pmag.max(), pdmt.max()))
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        plt.xlabel("magnitude loss"); plt.ylabel("dmt loss")
        plt.title("Per-example loss: DMT vs magnitude (below line = DMT better)")
        plt.tight_layout(); plt.savefig("results/exp1_dmt_vs_magnitude.png", dpi=120)
        plt.close()

        # Plot 3: DMT sparsity histogram
        plt.figure(figsize=(6, 4))
        plt.hist(wiki[wiki.method == "dmt"]["mean_sparsity"], bins=20)
        plt.xlabel("DMT mean kept-edge fraction"); plt.ylabel("count")
        plt.title("DMT natural sparsity (WikiText)")
        plt.tight_layout(); plt.savefig("results/exp1_dmt_sparsity.png", dpi=120)
        plt.close()

    # Readable summary
    print("=== Experiment 1 summary ===")
    if "wikitext_ppl_by_method" in stats:
        print("\nWikiText mean PPL by method:")
        for m in METHODS:
            v = stats["wikitext_ppl_by_method"].get(m)
            if v is not None:
                print(f"  {m:10s} {v:.3f}")
        print("\nDMT vs baseline (per-example loss, paired Wilcoxon):")
        for b, r in stats["wikitext_dmt_vs_baseline_loss"].items():
            print(f"  dmt vs {b:10s} n={r['n']} median_diff={r['median_diff']:+.4f} p={r['p']:.2e}")
        print(f"\nDMT mean sparsity: {stats.get('dmt_mean_sparsity', float('nan')):.3f}")
    if "kinship_accuracy_by_method" in stats:
        print("\nKinship accuracy by method:")
        for m in METHODS:
            v = stats["kinship_accuracy_by_method"].get(m)
            if v is not None:
                print(f"  {m:10s} {v:.3f}")
    print(f"\n=== VERDICT === {stats['verdict']}")
    return stats


if __name__ == "__main__":
    main()
