"""Generate the paper's figures from on-disk stats JSONs and parquets.

Same provenance discipline as write_paper.py: every plotted value is read from
a results/ file; no number is hand-typed.

* paper/figures/fig_regime_map.png   -- topology vs confidence AUC across the
                                        eight failure-prediction regimes
* paper/figures/fig_suppression.png  -- marginal vs partial view of per-head
                                        H1 persistence against induction strength

Regenerate with: uv run python -m src.make_figures
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# palette (dataviz reference instance, validated on white surface)
BLUE = "#2a78d6"      # series 1: topology
AQUA = "#1baf7a"      # series 2: confidence (relief: direct labels + shape)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "svg.fonttype": "path",
})


def _j(name):
    return json.load(open(f"results/{name}.json"))


def fig_regime_map(out="paper/figures/fig_regime_map.png"):
    e7 = _j("exp7_stats")["all_items"]
    e13 = _j("exp13_stats")["all_items"]
    e13r = _j("exp13_replication_stats")["combined"]["hop5_only"]
    e13b = _j("exp13_1p5b_stats")["all_items"]
    e14 = _j("exp14_stats")["all_items"]
    e15 = _j("exp15_stats")["all_items"]
    e16 = _j("exp16_stats")

    def row(label, s):
        acc = s.get("base_rate_correct", s.get("acc"))
        return (label.format(acc=acc),
                s["auc"]["topology_only"]["mean_auc"],
                s["auc"]["confidence_only"]["mean_auc"])

    def row16(label, cond):
        c = e16[cond]
        return (label.format(acc=c["acc"]),
                c["bootstrap"]["head_selected"]["auc"],
                c["auc"]["confidence_only"]["mean_auc"])

    # top -> bottom narrative order
    rows = [
        row("easy, 1–3 hops (acc {acc:.2f})", e7),
        row("ceiling, 4–5 hops (acc {acc:.2f})", e13),
        row("hop 5, 3 seeds pooled (acc {acc:.2f})", e13r),
        row("1.5B, 3–5 hops (acc {acc:.2f})", e13b),
        row("distractor contexts (acc {acc:.2f})", e14),
        row("gold-only contexts (acc {acc:.2f})", e15),
        row16("MTop-Div, distractor (acc {acc:.2f})", "distractor"),
        row16("MTop-Div, gold-only (acc {acc:.2f})", "gold"),
    ]
    groups = [("SYNTHETIC · QWEN2.5", 0, 4), ("HOTPOTQA · MISTRAL-7B", 4, 8)]

    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ys = np.arange(len(rows))[::-1].astype(float)
    ys[4:] -= 0.55  # breathing room between the two groups

    ax.axvline(0.5, color=MUTED, lw=1, zorder=1)
    ax.text(0.5, ys[0] + 0.85, "chance", color=MUTED, fontsize=7.5,
            ha="center", va="bottom")

    for (label, topo, conf), y in zip(rows, ys):
        ax.plot([conf, topo], [y, y], color=GRID, lw=2, zorder=2,
                solid_capstyle="round")
        ax.plot(conf, y, "s", color=AQUA, ms=7, mec="white", mew=1.4, zorder=3)
        ax.plot(topo, y, "o", color=BLUE, ms=8.5, mec="white", mew=1.4, zorder=4)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    for name, a, b in groups:
        ax.text(-0.02, ys[a] + 0.62, name, transform=ax.get_yaxis_transform(),
                fontsize=6.8, color=MUTED, ha="right", va="bottom")

    # direct labels on the first row (relief for the aqua contrast warn)
    ax.annotate("topology", (rows[0][1], ys[0]), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=7.5,
                color=INK2)
    ax.annotate("confidence", (rows[0][2], ys[0]), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=7.5,
                color=INK2)

    ax.set_xlim(0.38, 1.02)
    ax.set_xticks(np.arange(0.4, 1.01, 0.1))
    ax.set_xlabel("failure-prediction ROC-AUC (5-fold CV)", fontsize=8)
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 1.25)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def fig_suppression(out="paper/figures/fig_suppression.png"):
    s = _j("exp3_stats")
    df = pd.read_parquet("results/exp3.parquet")
    controls = s["controls"]
    x = df["h1_persistence"].to_numpy(dtype=float)
    y = df["induction_score"].to_numpy(dtype=float)

    # residualize both on the same first-order controls as compute_stats_exp3
    C = np.column_stack([df[c].to_numpy(dtype=float) for c in controls])
    C = np.column_stack([np.ones(len(C)), C])
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]

    def _p(v):
        return f"p = {v:.2f}" if v >= 0.001 else f"p = {v:.0e}".replace("e-0", "e-")

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    panels = [
        (axes[0], x, y, "marginal",
         f"Spearman ρ = {s['raw_spearman_rho']:.3f} ({_p(s['raw_spearman_p'])})",
         "H1 total persistence", "induction strength"),
        (axes[1], rx, ry, "partial (first-order controls removed)",
         f"partial ρ = {s['partial_spearman_rho']:.3f} "
         f"({_p(s['partial_spearman_p'])})",
         "H1 persistence, residual", "induction strength, residual"),
    ]
    for ax, px, py, title, note, xl, yl in panels:
        ax.plot(px, py, "o", color=BLUE, ms=5, mec="white", mew=0.8,
                alpha=0.9, zorder=3)
        b = np.polyfit(px, py, 1)
        xs = np.array([px.min(), px.max()])
        ax.plot(xs, np.polyval(b, xs), color=INK2, lw=1.2, zorder=2)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.text(0.97, 0.97, note, transform=ax.transAxes, fontsize=7.5,
                color=INK2, va="top", ha="right",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.grid(color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.6, w_pad=1.6)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_regime_map()
    fig_suppression()
