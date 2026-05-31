"""Generate FINDINGS_exp5.md mechanically from results/exp5_stats.json.

No hand-typed numbers: every value is read from the JSON produced by
compute_stats_exp5. Re-running this script reproduces the file byte-for-byte.
"""
from __future__ import annotations

import json


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def _p(x):
    if x is None:
        return "n/a (test undefined)"
    return f"{x:.4g}"


def render(stats: dict) -> str:
    L = []
    A = L.append
    A("# Experiment 5 Findings")
    A("")
    A("Causal test (E1) and cycle inspection (E2) for the induction circuit. "
      "Generated mechanically from `results/exp5_stats.json`; do not edit by hand.")
    A("")
    A(f"- Model: `{stats['model']}`")
    A(f"- Induction copy distance S = {stats['seq_len']}")
    A(f"- Gap tolerance for E2: |gap - S| <= {stats['gap_tolerance']}")
    A(f"- Significance level alpha = {stats['alpha']}")
    A("")

    # ---- E2 -----------------------------------------------------------------
    e2 = stats["E2"]
    A("## E2 - Cycle inspection (descriptive)")
    A("")
    A(f"**Verdict: {e2['verdict']}**")
    A("")
    A(f"Metric: {e2['metric']}.")
    A("")
    A(f"- Induction heads (n = {e2['n_induction_heads']}): "
      f"mean fraction at gap ~ S = {_fmt(e2['induction_mean_fraction'])}")
    A(f"- Non-induction heads (n = {e2['n_noninduction_heads']}): "
      f"mean fraction at gap ~ S = {_fmt(e2['noninduction_mean_fraction'])}")
    A(f"- Mann-Whitney (induction > non-induction), p = {_p(e2['mann_whitney_p'])}")
    A("")
    A("Per-head fractions:")
    A("")
    A("| group | fractions |")
    A("|---|---|")
    A(f"| induction | {', '.join(_fmt(x, 3) for x in e2['induction_fractions']) or '(none)'} |")
    A(f"| non-induction | {', '.join(_fmt(x, 3) for x in e2['noninduction_fractions']) or '(none)'} |")
    A("")
    A("Gap histograms (gap -> count):")
    A("")
    A("| group | histogram |")
    A("|---|---|")
    A(f"| induction | {_hist(e2['gap_histogram_induction'])} |")
    A(f"| non-induction | {_hist(e2['gap_histogram_noninduction'])} |")
    A("")

    # ---- E1 -----------------------------------------------------------------
    e1 = stats["E1"]
    A("## E1 - Causal ablation (functional)")
    A("")
    A(f"**Verdict: {e1['verdict']}**")
    A("")
    A(f"Readout: model second-copy next-token cross-entropy. Damage(condition) = "
      f"loss(condition) - loss(unmasked), paired across {e1['n_sequences']} sequences.")
    A("")
    A("Median damage by condition:")
    A("")
    A("| condition | median damage |")
    A("|---|---|")
    for cond in ("cycle", "magnitude", "random"):
        A(f"| {cond} | {_fmt(e1['median_damage'][cond])} |")
    A("")
    A("Paired one-sided Wilcoxon signed-rank (cycle damage greater):")
    A("")
    A("| comparison | median diff | p-value |")
    A("|---|---|---|")
    cm = e1["cycle_vs_magnitude"]
    cr = e1["cycle_vs_random"]
    A(f"| cycle vs magnitude | {_fmt(cm['median_diff'])} | {_p(cm['p_value'])} |")
    A(f"| cycle vs random | {_fmt(cr['median_diff'])} | {_p(cr['p_value'])} |")
    A("")
    sa = e1["sanity"]
    A("Sanity check (not a verdict):")
    A("")
    A(f"- cycle damage > 0: {sa['cycle_damage_positive']}")
    A(f"- magnitude damage > 0: {sa['magnitude_damage_positive']}")
    A(f"- {sa['note']}")
    A("")

    # ---- interpretation -----------------------------------------------------
    A("## Interpretation")
    A("")
    A(_interpret(e1, e2))
    A("")
    return "\n".join(L)


def _hist(h):
    if not h:
        return "(empty)"
    return ", ".join(f"{k}:{v}" for k, v in h.items())


def _interpret(e1, e2):
    parts = []
    if e1["verdict"] == "GREEN":
        parts.append("E1 GREEN: ablating the topological cycle edges hurts "
                     "induction behavior more than removing the same number of "
                     "highest-magnitude or random edges - topology selects "
                     "functionally special edges beyond magnitude.")
    elif e1["verdict"] == "PARTIAL":
        parts.append("E1 PARTIAL: cycle ablation hurts more than random but not "
                     "reliably more than magnitude - cycle edges matter, but not "
                     "beyond what magnitude already captures.")
    else:
        parts.append("E1 RED: cycle ablation does not significantly exceed "
                     "random - no causal signal that topology marks special "
                     "edges for induction behavior.")
    if not (e1["sanity"]["cycle_damage_positive"]
            and e1["sanity"]["magnitude_damage_positive"]):
        parts.append("CAUTION: at least one of cycle/magnitude damage is not "
                     "positive, so the ablation/readout may be too weak to "
                     "interpret the E1 contrast confidently.")
    if e2["verdict"] == "GREEN":
        parts.append("E2 GREEN: induction-head cycle edges concentrate at the "
                     "induction gap S more than non-induction heads.")
    else:
        parts.append("E2 RED: induction-head cycle edges do not concentrate at "
                     "the induction gap more than non-induction heads; cycles "
                     "are not obviously the copy edges (still informative).")
    return " ".join(parts)


def main(stats_path="results/exp5_stats.json", out_path="FINDINGS_exp5.md"):
    with open(stats_path) as f:
        stats = json.load(f)
    md = render(stats)
    with open(out_path, "w") as f:
        f.write(md)
    return md


if __name__ == "__main__":
    main()
