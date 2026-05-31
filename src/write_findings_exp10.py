"""Generate FINDINGS_exp10.md mechanically from results/exp10_stats.json. No hand-typed numbers."""
from __future__ import annotations

import json


def _f(x, nd=3):
    return "n/a" if x is None else f"{float(x):.{nd}f}"


def generate(stats_path="results/exp10_stats.json", out_path="FINDINGS_exp10.md"):
    s = json.load(open(stats_path))
    a, b = s["part_a_attribution"], s["part_b_causal"]
    L = ["# Experiment 10 Findings", "",
         "Topological attribution + causal validation on **IOI** (name-mover heads, GPT-2), "
         "where attention is more diffuse so the magnitude baseline is handicapped. Generated "
         "mechanically from `results/exp10_stats.json`; do not edit by hand.", "",
         f"- task = {s['task']}, model = `{s['model']}`, alpha = {s['alpha']}",
         f"- **Headline verdict: {s['headline_verdict']}**", "",
         "## Part A - attribution (recover the END->IO name-mover edge)",
         "", f"**Verdict: {a['verdict']}** (n = {a['n_observations']} head x prompt)", "",
         "| saliency | mean AUC | mean p@1 | p vs chance | p vs magnitude |",
         "|---|---|---|---|---|"]
    for sal, m in a["per_saliency"].items():
        L.append(f"| {sal} | {_f(m['mean_auc'])} | {_f(m['mean_p_at_1'])} | "
                 f"{_f(m.get('p_vs_chance'),4)} | {_f(m.get('p_vs_magnitude'),4)} |")
    L += ["", "## Part B - causal (damage = IO-S margin drop from ablation)",
          "", f"**Verdict: {b['verdict']}** (n = {b['n']} prompts)", "",
          "| condition | median damage |", "|---|---|"]
    for c in ["cycle", "sheaf", "magnitude", "random"]:
        L.append(f"| {c} | {_f(b['median_damage'][c])} |")
    L += [""]
    for sal in ["cycle", "sheaf"]:
        L.append(f"- {sal}: damage>0 = {b[f'{sal}_damage_positive']}, "
                 f"vs random p = {_f(b[f'{sal}_vs_random_p'],4)}, "
                 f"vs magnitude p = {_f(b[f'{sal}_vs_magnitude_p'],4)}")
    L += ["", "## Interpretation", "", _interpret(s, a, b), ""]
    open(out_path, "w").write("\n".join(L).rstrip() + "\n")
    return "\n".join(L)


def _interpret(s, a, b):
    ps = a["per_saliency"]; mag = ps["magnitude"]["mean_auc"]; al = s["alpha"]
    beat = [t for t in ("cycle_participation", "sheaf_discord")
            if ps[t].get("p_vs_chance") is not None and ps[t]["p_vs_chance"] < al
            and ps[t]["mean_auc"] > 0.5]
    parts = []
    if a["verdict"] == "GREEN":
        parts.append(f"Part A GREEN: a topological saliency beats the attention-magnitude "
                     f"baseline (AUC {mag:.3f}) at localizing the name-mover edge -- topology "
                     f"earns its keep where attention is diffuse.")
    elif a["verdict"] == "PARTIAL":
        parts.append(f"Part A PARTIAL: magnitude (AUC {mag:.3f}) is not beaten; "
                     + ("topological " + ", ".join(beat) + " beat chance but not magnitude."
                        if beat else "no topological saliency beat chance."))
    else:
        parts.append(f"Part A RED: no topological saliency beats chance (magnitude AUC {mag:.3f}).")
    md = b["median_damage"]
    all_neg = all(md[c] <= 0 for c in ("cycle", "sheaf", "magnitude"))
    if b["verdict"] == "GREEN":
        parts.append("Part B GREEN: ablating topology-flagged edges reduces the IO-S margin more "
                     "than random -- the flagged edges are causally part of the name-mover circuit.")
    elif b["verdict"] == "PARTIAL":
        parts.append("Part B PARTIAL: flagged-edge ablation hurts the IO behavior but not reliably "
                     "more than random.")
    else:
        parts.append("Part B RED: ablating flagged edges does not reduce the IO margin beyond random.")
    if all_neg:
        parts.append("CAUTION: every condition's median damage is <=0 (ablation tends to RAISE the "
                     "IO-S margin, magnitude most of all). The name-mover edge feeds both IO and S "
                     "logits, so removing it within these heads can net-help the margin -- this "
                     "readout cannot cleanly score 'necessity' here; the Part B RED reflects a "
                     "compromised instrument, not strong evidence against topology.")
    return " ".join(parts)


if __name__ == "__main__":
    print(generate()[:200])
