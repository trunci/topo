"""Generate FINDINGS_exp13.md mechanically from results/exp13_stats.json.

No hand-typed numbers: every value is read from the JSON produced by
compute_stats_exp13. Re-running reproduces the file byte-for-byte.
"""
from __future__ import annotations

import json


def _f(x, nd=3):
    return "n/a" if x is None else f"{float(x):.{nd}f}"


def _auc_row(name, d):
    return (f"| {name} | {_f(d['mean_auc'])} | {_f(d['std_auc'])} | "
            f"{d['n_folds']} |")


def _slice_md(title, v):
    L = [f"### {title}", ""]
    if v is None:
        L += ["(no items in this slice)", ""]
        return L
    L += [f"**Verdict: {v['verdict']}**", ""]
    L += [f"- Items: {v['n_items']}; base rate correct = {_f(v['base_rate_correct'])}; "
          f"majority-class acc = {_f(v['majority_class_acc'])}", ""]
    L += ["Cross-validated ROC-AUC (predicting is_correct):", "",
          "| model | mean AUC | std | folds |", "|---|---|---|---|"]
    a = v["auc"]
    L += [_auc_row("confidence only (M0)", a["confidence_only"]),
          _auc_row("confidence + topology (M1)", a["confidence_plus_topology"]),
          _auc_row("topology only", a["topology_only"]),
          _auc_row("confidence + first-order controls", a["confidence_plus_controls"])]
    d = v["delta_auc_full_minus_baseline"]
    L += ["",
          f"- delta-AUC (M1 - M0): median {_f(d['median'])}, "
          f"paired Wilcoxon p = {_f(d['wilcoxon_p'], 4)}",
          f"- topology beats chance: {v['topology_beats_chance']}",
          f"- topology adds beyond confidence: {v['topology_adds_beyond_confidence']}",
          ""]
    return L


def _interpret(s):
    h4 = s.get("hop4_only") or {}
    h5 = s.get("hop5_only") or {}
    verdicts = {h: v.get("verdict", "n/a") for h, v in [("hop4", h4), ("hop5", h5)]}
    any_green = any(v == "GREEN" for v in verdicts.values())
    any_partial = any(v == "PARTIAL" for v in verdicts.values())
    both_underpowered = all(v == "UNDERPOWERED" for v in verdicts.values())

    if both_underpowered:
        return ("UNDERPOWERED on both hop levels: the model is still too accurate at "
                "hop=4 and hop=5 to probe miscalibration; the target accuracy range "
                "[0.35, 0.70] was not reached. Deeper hops or a harder task needed.")
    if any_green:
        return ("GREEN on at least one hop: topology adds predictive power for failure "
                "beyond the model's own confidence in the deeper-hop regime, confirming "
                "the miscalibration hypothesis.")
    if any_partial:
        return ("PARTIAL: topology beats chance at predicting failure in the harder "
                "regime but still does not add beyond confidence. The miscalibration "
                "hypothesis is not confirmed.")
    return ("RED: topology does not predict failure above chance even in the harder "
            "deep-hop regime.")


def generate(stats_path="results/exp13_stats.json", out_path="FINDINGS_exp13.md"):
    with open(stats_path) as f:
        s = json.load(f)
    L = ["# Experiment 13 Findings", "",
         "Deep-hop failure prediction (hop=4, hop=5). Generated mechanically from "
         "`results/exp13_stats.json`; do not edit by hand.", "",
         f"- Model: `{s['model']}`",
         f"- Hops tested: {s['hops_tested']}",
         f"- Overall accuracy: {_f(s['overall_accuracy'])}",
         f"- Pre-condition met (at least one hop in [0.35, 0.70]): "
         f"{s['precondition_met']}",
         f"- Significance level alpha = {s['alpha']}; {s['n_splits']}-fold "
         f"stratified CV", ""]
    L += ["Accuracy by hop:", "", "| hop | accuracy |", "|---|---|"]
    for hop, acc in s["acc_by_hop"].items():
        L.append(f"| {hop} | {_f(acc)} |")
    L.append("")
    L += _slice_md("All items (hop=4 + hop=5 combined)", s["all_items"])
    L += _slice_md("Hop=4 only", s.get("hop4_only"))
    L += _slice_md("Hop=5 only", s.get("hop5_only"))
    L += ["## Interpretation", "", _interpret(s), ""]
    text = "\n".join(L).rstrip() + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    print(generate()[:400])
