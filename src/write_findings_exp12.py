"""Generate FINDINGS_exp12.md mechanically from results/exp12_stats.json."""
from __future__ import annotations

import json


def _f(x, nd=3):
    return "n/a" if x is None else f"{float(x):.{nd}f}"


def _auc_row(name, d):
    return (f"| {name} | {_f(d['mean_auc'])} | {_f(d['std_auc'])} | "
            f"{d['n_folds']} |")


def _slice_md(title, v, prediction=None):
    L = [f"### {title}", ""]
    if v is None:
        L += ["(no items in this slice)", ""]
        return L
    pred_note = f" *(pre-registered prediction: {prediction})*" if prediction else ""
    L += [f"**Verdict: {v['verdict']}**{pred_note}", ""]
    L += [f"- Items: {v['n_items']}; base rate correct = {_f(v['base_rate_correct'])}; "
          f"majority-class acc = {_f(v['majority_class_acc'])}", ""]
    if v["underpowered"]:
        L += ["UNDERPOWERED: base rate outside [0.15, 0.85] — probe uninformative.", ""]
        return L
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


def generate(stats_path="results/exp12_stats.json", out_path="FINDINGS_exp12.md"):
    with open(stats_path) as f:
        s = json.load(f)

    de = s["distractor_effectiveness"]
    L = ["# Experiment 12 Findings", "",
         "Distractor-augmented failure prediction. Generated mechanically from "
         "`results/exp12_stats.json`; do not edit by hand.", "",
         f"- Model: `{s['model']}`",
         f"- Overall accuracy: {_f(s['overall_accuracy'])}",
         f"- Significance level alpha = {s['alpha']}; {s['n_splits']}-fold CV", ""]

    L += ["**Distractor effectiveness check:**", "",
          f"- Accuracy clean: {_f(s['acc_by_type'].get('clean'))} | "
          f"distracted: {_f(s['acc_by_type'].get('distracted'))}",
          f"- Accuracy drop (clean - distracted): {_f(de['acc_drop_clean_minus_distracted'])}",
          f"- Effective (drop > 0.05): {de['effective']}", ""]

    L += ["Accuracy by hop and type:", "", "| hop | clean | distracted |", "|---|---|---|"]
    for hop, by_type in sorted(s["acc_by_hop_and_type"].items()):
        c = _f(by_type.get("clean"))
        d = _f(by_type.get("distracted"))
        L.append(f"| {hop} | {c} | {d} |")
    L.append("")

    L += _slice_md("All items", s["all_items"])
    L += _slice_md("Clean items (pre-registered: PARTIAL)", s["clean_items"],
                   prediction="PARTIAL")
    L += _slice_md("Distracted items (pre-registered: GREEN)", s["distracted_items"],
                   prediction="GREEN")

    prediction_correct = (
        s["clean_items"]["verdict"] in ("PARTIAL", "RED")
        and s["distracted_items"]["verdict"] == "GREEN"
    )
    L += ["## Prediction check", "",
          f"Pre-registered: clean=PARTIAL, distracted=GREEN. "
          f"Prediction correct: **{prediction_correct}**", ""]

    text = "\n".join(L).rstrip() + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    print(generate()[:500])
