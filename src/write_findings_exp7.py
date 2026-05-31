"""Generate FINDINGS_exp7.md mechanically from results/exp7_stats.json.

No hand-typed numbers: every value is read from the JSON produced by
compute_stats_exp7. Re-running reproduces the file byte-for-byte.
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


def generate(stats_path="results/exp7_stats.json", out_path="FINDINGS_exp7.md"):
    with open(stats_path) as f:
        s = json.load(f)
    L = ["# Experiment 7 Findings", "",
         "Failure prediction from attention topology. Generated mechanically from "
         "`results/exp7_stats.json`; do not edit by hand.", "",
         f"- Model: `{s['model']}`",
         f"- Overall accuracy: {_f(s['overall_accuracy'])}",
         f"- Significance level alpha = {s['alpha']}; {s['n_splits']}-fold "
         f"stratified CV", ""]
    L += ["Accuracy by hop:", "", "| hop | accuracy |", "|---|---|"]
    for hop, acc in s["acc_by_hop"].items():
        L.append(f"| {hop} | {_f(acc)} |")
    L.append("")
    L += _slice_md("All items (mixed 1/2/3-hop)", s["all_items"])
    L += _slice_md("2-hop only (Spike regime)", s["two_hop_only"])
    L += ["## Interpretation", "", _interpret(s["all_items"]), ""]
    text = "\n".join(L).rstrip() + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


def _interpret(v):
    if v is None:
        return "No items to interpret."
    if v["verdict"] == "GREEN":
        return ("GREEN: topology predicts reasoning failure above chance AND adds "
                "predictive power beyond the model's own confidence -- attention "
                "shape carries failure-relevant information that output confidence "
                "does not.")
    if v["verdict"] == "PARTIAL":
        return ("PARTIAL: topology predicts failure above chance, but does not add "
                "beyond the model's confidence -- it may be re-deriving what "
                "confidence already encodes.")
    if v["verdict"] == "UNDERPOWERED":
        return ("UNDERPOWERED: correctness variance is too low (base rate extreme) "
                "to evaluate the probe; the hop mix should be retuned and the run "
                "repeated. Not a pass or fail.")
    return ("RED: topology does not predict reasoning failure above chance in this "
            "setup.")


if __name__ == "__main__":
    print(generate()[:300])
