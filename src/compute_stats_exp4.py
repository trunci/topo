"""Experiment 4: residual-information generalization grid -> results/exp4_stats.json.

For each (model, circuit) in the 2x3 grid {gpt2, distilgpt2} x {induction,
previous-token, duplicate-token}, run the SAME nested-OLS residual test as Exp 3
(reused via `compute_stats_exp3.residual_test`): does h1_persistence add predictive
power for that circuit's score beyond the three first-order controls?

Pre-registered (see docs/.../experiment4-residual-generalization-design.md):
  per cell: GREEN iff delta_r2 >= 0.02 AND nested-F p < 0.05; else RED.
  overall:  STRONG-GENERALIZE iff >=5/6 GREEN; PARTIAL-GENERALIZE iff 2-4 GREEN;
            DOES-NOT-GENERALIZE iff <=1 GREEN.
These rules are FIXED before computing on real data.
"""
from __future__ import annotations

import json
import pandas as pd

from src.compute_stats_exp3 import residual_test, CONTROLS, DELTA_R2_FLOOR, ALPHA

MODELS = ["gpt2", "distilgpt2"]
# (circuit name, score column) pairs, evaluated in this fixed order.
CIRCUITS = [
    ("induction", "induction_score"),
    ("previous-token", "prev_token_score"),
    ("duplicate-token", "dup_token_score"),
]
FEATURE = "h1_persistence"


def _overall_verdict(n_green: int) -> str:
    if n_green >= 5:
        return ("STRONG-GENERALIZE: H1 persistence adds residual predictive power "
                "in >=5 of 6 (model,circuit) cells.")
    if n_green >= 2:
        return ("PARTIAL-GENERALIZE: H1 persistence adds residual predictive power "
                "in 2-4 of 6 (model,circuit) cells.")
    return ("DOES-NOT-GENERALIZE: H1 persistence adds residual predictive power in "
            "<=1 of 6 (model,circuit) cells (Exp 3 likely induction-in-GPT2 specific).")


def compute_stats(parquet_path: str, out_path: str) -> dict:
    df = pd.read_parquet(parquet_path)

    cells = []
    n_green = 0
    for model in MODELS:
        sub = df[df["model"] == model]
        controls = [sub[c].to_numpy() for c in CONTROLS]
        for circuit, score_col in CIRCUITS:
            r = residual_test(sub[score_col].to_numpy(), sub[FEATURE].to_numpy(),
                              controls)
            green = r["adds_power"]
            if green:
                n_green += 1
            cells.append({
                "model": model,
                "circuit": circuit,
                "score_column": score_col,
                "n_heads": r["n"],
                "baseline_r2": r["baseline_r2"],
                "full_r2": r["full_r2"],
                "delta_r2": r["delta_r2"],
                "f_stat": r["f_stat"],
                "f_pvalue": r["f_pvalue"],
                "h1_coef": r["coef"],
                "h1_coef_p": r["coef_p"],
                "partial_spearman_rho": r["partial_spearman_rho"],
                "partial_spearman_p": r["partial_spearman_p"],
                "raw_spearman_rho": r["raw_spearman_rho"],
                "raw_spearman_p": r["raw_spearman_p"],
                "cell_verdict": "GREEN" if green else "RED",
            })

    stats = {
        "models": MODELS,
        "circuits": [c for c, _ in CIRCUITS],
        "controls": CONTROLS,
        "feature": FEATURE,
        "delta_r2_floor": DELTA_R2_FLOOR,
        "alpha": ALPHA,
        "n_cells": len(cells),
        "n_green": n_green,
        "overall_verdict": _overall_verdict(n_green),
        "cells": cells,
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    compute_stats("results/exp4.parquet", "results/exp4_stats.json")
    print("WROTE results/exp4_stats.json")
