"""Generate FINDINGS_exp6.md mechanically from results/exp6_stats.json.

No numbers are hand-typed here; every value is read from the JSON produced by
src/compute_stats_exp6.py. This keeps the writeup faithful to the computed stats.
"""
from __future__ import annotations

import json


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def generate(stats_path="results/exp6_stats.json", out_path="FINDINGS_exp6.md"):
    with open(stats_path) as f:
        s = json.load(f)
    lines = []
    lines.append("# Experiment 6 Findings")
    lines.append("")
    lines.append("Stronger directed causal test of induction cycle edges "
                 "(resolving Exp 5's underpowered E1). Generated mechanically "
                 "from `results/exp6_stats.json`; do not edit by hand.")
    lines.append("")
    lines.append(f"- Induction copy distance S = {s['seq_len']}")
    lines.append(f"- Gap tolerance (treatment = directed cycle edges at "
                 f"|gap - S| <= {s['gap_tolerance']})")
    lines.append(f"- Significance level alpha = {s['alpha']}")
    lines.append("- Intervention: directed, edge-exact, cumulative across all "
                 "induction heads. Readout: second-copy next-token cross-entropy.")
    lines.append("- Damage(condition) = loss(condition) - loss(unmasked), paired "
                 "across sequences.")
    lines.append("")
    for model_name, m in s["models"].items():
        lines.append(f"## Model: `{model_name}`")
        lines.append("")
        lines.append(f"**Verdict: {m['verdict']}**")
        lines.append("")
        lines.append(f"- Sequences: {m['n_sequences']}; mean edges ablated per "
                     f"sequence K = {_fmt(m['k_ablated_mean'], 2)}")
        lines.append("")
        lines.append("Median damage by condition (loss increase vs unmasked; "
                     "positive = ablation hurt induction):")
        lines.append("")
        lines.append("| condition | median damage | mean damage |")
        lines.append("|---|---|---|")
        for c in ["cycle", "magnitude", "random"]:
            lines.append(f"| {c} | {_fmt(m['median_damage'][c])} | "
                         f"{_fmt(m['mean_damage'][c])} |")
        lines.append("")
        lines.append("Paired one-sided Wilcoxon (cycle damage greater):")
        lines.append("")
        lines.append("| comparison | median diff | p-value |")
        lines.append("|---|---|---|")
        for ctrl in ["magnitude", "random"]:
            cv = m[f"cycle_vs_{ctrl}"]
            lines.append(f"| cycle vs {ctrl} | {_fmt(cv['median_diff'])} | "
                         f"{_fmt(cv['p_value'], 7)} |")
        lines.append("")
        lines.append(f"- cycle damage positive: {m['cycle_damage_positive']}")
        lines.append(f"- cycle beats random (p < {s['alpha']}): {m['beats_random']}")
        lines.append("")
        lines.append(_interpret(model_name, m, s["alpha"]))
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


def _interpret(model_name, m, alpha):
    v = m["verdict"]
    if v == "GREEN":
        return (f"Interpretation ({model_name}): GREEN -- ablating the directed "
                "gap~S critical-cycle edges damages induction significantly more "
                "than random edges of equal budget. The topology-selected edges "
                "are causally functional for induction behavior.")
    if v == "RED":
        return (f"Interpretation ({model_name}): RED -- cycle ablation hurts "
                "induction, but no more than random edges of equal budget. The "
                "edges matter, yet topology selection is not special.")
    return (f"Interpretation ({model_name}): INCONCLUSIVE -- cycle damage is not "
            "positive, so the ablation/readout remains too weak to interpret the "
            "contrast (the Exp 5 E1 failure mode).")


if __name__ == "__main__":
    print(generate()[:400])
