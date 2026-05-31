"""Generate FINDINGS_exp9.md mechanically from results/exp9_stats.json. No hand-typed numbers."""
from __future__ import annotations

import json


def _f(x, nd=3):
    return "n/a" if x is None else f"{float(x):.{nd}f}"


def generate(stats_path="results/exp9_stats.json", out_path="FINDINGS_exp9.md"):
    s = json.load(open(stats_path))
    a, b = s["part_a_attribution"], s["part_b_causal"]
    L = ["# Experiment 9 Findings", "",
         "Topological attribution + causal validation: does per-edge topology localize the "
         "induction copy edge beyond attention magnitude, and is it causal? Generated "
         "mechanically from `results/exp9_stats.json`; do not edit by hand.", "",
         f"- alpha = {s['alpha']}", f"- **Headline verdict: {s['headline_verdict']}**", "",
         "## Part A - Attribution (ROC-AUC at recovering ground-truth copy edges)",
         "", f"**Verdict: {a['verdict']}** (n = {a['n_observations']} head x seq)", "",
         "| saliency | mean AUC | mean p@k | p vs chance | p vs magnitude |",
         "|---|---|---|---|---|"]
    for sal, m in a["per_saliency"].items():
        L.append(f"| {sal} | {_f(m['mean_auc'])} | {_f(m['mean_p_at_k'])} | "
                 f"{_f(m.get('p_vs_chance'), 4)} | {_f(m.get('p_vs_magnitude'), 4)} |")
    L += ["", "## Part B - Causal validation (damage = loss(cond) - loss(unmasked))",
          "", f"**Verdict: {b['verdict']}** (n = {b['n']} seq)", "",
          "| condition | median damage |", "|---|---|"]
    for c in ["cycle", "sheaf", "magnitude", "random"]:
        L.append(f"| {c} | {_f(b['median_damage'][c])} |")
    L += [""]
    for sal in ["cycle", "sheaf"]:
        L.append(f"- {sal}: damage>0 = {b[f'{sal}_damage_positive']}, "
                 f"vs random p = {_f(b[f'{sal}_vs_random_p'], 4)}, "
                 f"vs magnitude p = {_f(b[f'{sal}_vs_magnitude_p'], 4)}")
    L += ["", "## Interpretation", "", _interpret(s, a, b), ""]
    text = "\n".join(L).rstrip() + "\n"
    open(out_path, "w").write(text)
    return text


def _interpret(s, a, b):
    parts = []
    if a["verdict"] == "GREEN":
        parts.append("Part A GREEN: a topological saliency localizes the induction copy edge "
                     "beyond the attention-magnitude baseline -- topology adds attribution power.")
    elif a["verdict"] == "PARTIAL":
        parts.append("Part A PARTIAL: topology localizes the copy edge above chance but not "
                     "beyond attention magnitude -- it recovers the circuit, but the trivial "
                     "magnitude attributor does about as well.")
    else:
        parts.append("Part A RED: topological saliency does not localize the copy edge above chance.")
    if b["verdict"] == "GREEN":
        parts.append("Part B GREEN: ablating topology-flagged edges breaks induction more than "
                     "random -- the flagged edges are causally necessary.")
    elif b["verdict"] == "PARTIAL":
        parts.append("Part B PARTIAL: flagged-edge ablation hurts (damage>0) but not reliably "
                     "more than random.")
    else:
        parts.append("Part B RED: ablating flagged edges does not exceed random damage.")
    return " ".join(parts)


if __name__ == "__main__":
    print(generate()[:200])
