"""Evaluate the pre-registered criteria from results/spike.parquet."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

METRIC = "total_persistence"   # primary H1 summary for the paired test
ALPHA = 0.05


def paired_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (layer, head): paired Wilcoxon of metric, 2-hop vs 1-hop."""
    out = []
    for (layer, head), g in df.groupby(["layer", "head"]):
        piv = g.pivot_table(index="pair_id", columns="hop", values=metric)
        piv = piv.dropna()
        if 1 not in piv.columns or 2 not in piv.columns or len(piv) < 5:
            continue
        diff = piv[2].values - piv[1].values
        if np.allclose(diff, 0):
            p, stat = 1.0, 0.0
        else:
            stat, p = wilcoxon(piv[2].values, piv[1].values)
        # matched-pairs effect size: mean signed diff normalized by its SD
        eff = diff.mean() / (diff.std() + 1e-9)
        out.append({"layer": layer, "head": head, "n_pairs": len(piv),
                    "mean_diff": diff.mean(), "effect": eff, "p": p})
    res = pd.DataFrame(out)
    if len(res):
        res["p_adj"] = multipletests(res["p"], alpha=ALPHA, method="fdr_bh")[1]
        res["sig"] = res["p_adj"] < ALPHA
    return res


def nontriviality(df: pd.DataFrame) -> float:
    """Fraction of (head, example) observations with a non-trivial H1 cycle."""
    has_cycle = (df["n_cycles"] > 0) & (df["max_persistence"] > 0.05)
    return float(has_cycle.mean())


def main(in_path: str = "results/spike.parquet"):
    df = pd.read_parquet(in_path)

    # --- Criterion 1: non-triviality ---
    frac = nontriviality(df)
    crit1 = frac > 0.10
    print(f"[Criterion 1] non-trivial-H1 fraction = {frac:.2%} -> {'PASS' if crit1 else 'FAIL'}")

    # --- Criterion 2: discrimination ---
    res = paired_table(df, METRIC)
    n_sig = int(res["sig"].sum()) if len(res) else 0
    crit2 = n_sig > 0
    print(f"[Criterion 2] significant (layer,head) after BH = {n_sig} -> {'PASS' if crit2 else 'FAIL'}")
    if len(res):
        print("\nTop discriminating heads:")
        print(res.sort_values("p_adj").head(10).to_string(index=False))

    # --- Criterion 3: layer clustering (plausibility) ---
    if n_sig > 0:
        sig_layers = res.loc[res["sig"], "layer"]
        n_layers = df["layer"].max() + 1
        print(f"\n[Criterion 3] sig heads layer median = {sig_layers.median():.1f} "
              f"of {n_layers} (mid/late => plausible)")

    # --- Plots ---
    if len(res):
        n_layers = int(df["layer"].max() + 1)
        n_heads = int(df["head"].max() + 1)
        grid = np.full((n_layers, n_heads), np.nan)
        for _, r in res.iterrows():
            grid[int(r["layer"]), int(r["head"])] = -np.log10(max(r["p_adj"], 1e-12))
        plt.figure(figsize=(8, 6))
        plt.imshow(grid, aspect="auto", cmap="viridis")
        plt.colorbar(label="-log10(p_adj)")
        plt.xlabel("head"); plt.ylabel("layer")
        plt.title(f"2-hop vs 1-hop discrimination ({METRIC})")
        plt.tight_layout(); plt.savefig("results/heatmap.png", dpi=120)
        print("\nSaved results/heatmap.png")

        # distribution plot for the single most discriminating head
        best = res.sort_values("p_adj").iloc[0]
        g = df[(df["layer"] == best["layer"]) & (df["head"] == best["head"])]
        plt.figure(figsize=(6, 4))
        for hop, sub in g.groupby("hop"):
            plt.hist(sub[METRIC], bins=20, alpha=0.5, label=f"{hop}-hop")
        plt.legend(); plt.xlabel(METRIC)
        plt.title(f"L{int(best['layer'])}H{int(best['head'])} (p_adj={best['p_adj']:.1e})")
        plt.tight_layout(); plt.savefig("results/best_head_dist.png", dpi=120)
        print("Saved results/best_head_dist.png")

    # --- Verdict ---
    if crit1 and crit2:
        verdict = "GREEN: topology is non-trivial AND tracks hop-count. Build Experiment 1."
    elif crit1:
        verdict = "YELLOW: topology is real but not reasoning-linked. Pivot to runtime diagnostic."
    else:
        verdict = "RED: topology is trivial. Stop or rethink the premise."
    print(f"\n=== VERDICT === {verdict}")
    return res


if __name__ == "__main__":
    main()
