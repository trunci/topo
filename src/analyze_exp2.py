"""Experiment 2 plot: induction score vs H1 persistence, top induction heads marked."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze(parquet_path="results/exp2.parquet", out_png="results/exp2_scatter.png",
            top_k=10):
    df = pd.read_parquet(parquet_path)
    ind = df["induction_score"].to_numpy()
    h1 = df["h1_persistence"].to_numpy()
    top = np.argsort(ind)[::-1][:top_k]
    mask = np.zeros(len(df), dtype=bool)
    mask[top] = True

    plt.figure(figsize=(6, 5))
    plt.scatter(ind[~mask], h1[~mask], alpha=0.5, label="other heads")
    plt.scatter(ind[mask], h1[mask], color="crimson", label=f"top-{top_k} induction")
    plt.xlabel("induction score")
    plt.ylabel("H1 total persistence")
    plt.title("GPT-2 small: induction vs attention-graph topology")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"wrote {out_png}")


if __name__ == "__main__":
    analyze()
