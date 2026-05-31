"""Generate FINDINGS_exp8.md mechanically from results/exp8_stats.json.

No hand-typed numbers: every value is read from the JSON produced by
compute_stats_exp8. Re-running reproduces the file byte-for-byte.
"""
from __future__ import annotations

import json


def _f(x, nd=3):
    return "n/a" if x is None else f"{float(x):.{nd}f}"


def _auc_row(name, d):
    return f"| {name} | {_f(d['mean_auc'])} | {_f(d['std_auc'])} | {d['n_folds']} |"


def generate(stats_path="results/exp8_stats.json", out_path="FINDINGS_exp8.md"):
    with open(stats_path) as f:
        s = json.load(f)
    f1 = s["frame1_discrimination"]
    f2 = s["frame2_failure_prediction"]
    L = ["# Experiment 8 Findings", "",
         "Cellular sheaves over the residual stream: does *what flows* (sheaf "
         "consistency) beat *graph shape* (H1)? Generated mechanically from "
         "`results/exp8_stats.json`; do not edit by hand.", "",
         f"- Model: `{s['model']}`",
         f"- Items: {s['n_items']}; overall accuracy {_f(s['overall_accuracy'])}",
         f"- alpha = {s['alpha']}; {s['n_splits']}-fold CV; delta-R2 floor "
         f"{s['delta_r2_floor']}", ""]

    # Frame 1
    L += ["## Frame 1 - Discrimination (does sheaf add beyond H1 at predicting hop?)",
          "", f"**Verdict: {f1['verdict']}**", "",
          "Nested OLS predicting reasoning hop-count:", "",
          "| model | R2 |", "|---|---|",
          f"| H1 + first-order controls (baseline) | {_f(f1['baseline_r2'])} |",
          f"| + sheaf features (full) | {_f(f1['full_r2'])} |", "",
          f"- delta-R2 (sheaf block) = {_f(f1['delta_r2'])}",
          f"- nested F = {_f(f1['f_stat'], 2)}, p = {_f(f1['f_pvalue'], 5)}", ""]

    # Frame 2
    a = f2["auc"]
    L += ["## Frame 2 - Failure prediction (does sheaf add beyond confidence?)",
          "", f"**Verdict: {f2['verdict']}**", "",
          f"- Base rate correct = {_f(f2['base_rate_correct'])}; underpowered = "
          f"{f2['underpowered']}", "",
          "Cross-validated ROC-AUC (predict is_correct):", "",
          "| model | mean AUC | std | folds |", "|---|---|---|---|",
          _auc_row("confidence only (M0)", a["confidence_only"]),
          _auc_row("confidence + H1 (M1)", a["confidence_plus_h1"]),
          _auc_row("confidence + sheaf (M2)", a["confidence_plus_sheaf"]),
          _auc_row("confidence + H1 + sheaf (M3)", a["confidence_plus_h1_plus_sheaf"]),
          _auc_row("sheaf only", a["sheaf_only"]), ""]
    sc = f2["sheaf_vs_confidence"]
    L += [f"- sheaf vs confidence (M2 - M0): median delta {_f(sc['median_delta'])}, "
          f"paired Wilcoxon p = {_f(sc['wilcoxon_p'], 4)}",
          f"- sheaf beats chance: {f2['sheaf_beats_chance']}; "
          f"adds beyond confidence: {f2['sheaf_adds_beyond_confidence']}", ""]

    L += ["## Interpretation", "", _interpret(f1, f2), ""]
    text = "\n".join(L).rstrip() + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


def _interpret(f1, f2):
    parts = []
    if f1["verdict"] == "GREEN":
        parts.append("Frame 1 GREEN: sheaf consistency adds predictive power for "
                     "reasoning hop-count beyond H1 persistence -- 'what flows' "
                     "carries structure that graph shape alone misses.")
    else:
        parts.append("Frame 1 RED: sheaf features do not add beyond H1 for "
                     "discriminating hop-count; on this task the flow signal is "
                     "redundant with topological shape.")
    if f2["verdict"] == "GREEN":
        parts.append("Frame 2 GREEN: sheaf consistency predicts reasoning failure "
                     "beyond the model's own confidence -- succeeding where H1 "
                     "(Exp 7) was redundant.")
    elif f2["verdict"] == "PARTIAL":
        parts.append("Frame 2 PARTIAL: sheaf predicts failure above chance but does "
                     "not beat confidence -- same ceiling H1 hit in Exp 7.")
    elif f2["verdict"] == "UNDERPOWERED":
        parts.append("Frame 2 UNDERPOWERED: correctness variance too low to judge.")
    else:
        parts.append("Frame 2 RED: sheaf does not predict failure above chance.")
    return " ".join(parts)


if __name__ == "__main__":
    print(generate()[:300])
