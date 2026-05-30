"""Authoritative Experiment 1 statistics: compute every reported number from the
parquet and dump to JSON. No statistic is hand-typed elsewhere.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METHODS = ["unpruned", "dmt", "magnitude", "random", "window"]
BASELINES = ["magnitude", "random", "window"]


def _paired_wilcoxon(df, domain, metric, a, b):
    """Paired test of metric between method a and b across shared example_ids."""
    d = df[df.domain == domain]
    pa = d[d.method == a].set_index("example_id")[metric]
    pb = d[d.method == b].set_index("example_id")[metric]
    common = pa.index.intersection(pb.index)
    if len(common) < 5:
        return {"n": int(len(common)), "p": float("nan"),
                "median_diff": float("nan")}
    va, vb = pa.loc[common].values, pb.loc[common].values
    diff = va - vb
    if np.allclose(diff, 0):
        p = 1.0
    else:
        p = float(wilcoxon(va, vb).pvalue)
    return {"n": int(len(common)), "p": p, "median_diff": float(np.median(diff))}


def compute_stats(in_path="results/exp1.parquet", out_path="results/exp1_stats.json"):
    df = pd.read_parquet(in_path)
    stats = {}

    wiki = df[df.domain == "wikitext"]
    if len(wiki):
        ppl_by = wiki.groupby("method")["ppl"].mean().to_dict()
        stats["wikitext_ppl_by_method"] = {m: float(ppl_by[m]) for m in ppl_by}
        stats["wikitext_n_examples"] = int(wiki["example_id"].nunique())
        # paired tests: dmt vs each baseline on per-example loss (lower is better)
        stats["wikitext_dmt_vs_baseline_loss"] = {
            b: _paired_wilcoxon(df, "wikitext", "loss", "dmt", b) for b in BASELINES
        }
        dmt_ppl = stats["wikitext_ppl_by_method"].get("dmt", float("nan"))
        stats["dmt_beats_magnitude_ppl"] = bool(
            dmt_ppl < stats["wikitext_ppl_by_method"].get("magnitude", float("inf")))
        stats["dmt_beats_all_baselines_ppl"] = bool(
            all(dmt_ppl < stats["wikitext_ppl_by_method"].get(b, float("inf"))
                for b in BASELINES))
        stats["dmt_mean_sparsity"] = float(
            wiki[wiki.method == "dmt"]["mean_sparsity"].mean())

    kin = df[df.domain == "kinship"]
    if len(kin):
        acc_by = kin.groupby("method")["correct"].mean().to_dict()
        stats["kinship_accuracy_by_method"] = {m: float(acc_by[m]) for m in acc_by}
        stats["kinship_n_examples"] = int(kin["example_id"].nunique())
        dmt_acc = stats["kinship_accuracy_by_method"].get("dmt", float("nan"))
        stats["dmt_ge_magnitude_accuracy"] = bool(
            dmt_acc >= stats["kinship_accuracy_by_method"].get("magnitude", float("inf")))

    # overall verdict
    ppl_win = stats.get("dmt_beats_all_baselines_ppl", False)
    acc_ok = stats.get("dmt_ge_magnitude_accuracy", True)
    if ppl_win and acc_ok:
        stats["verdict"] = "WIN: DMT beats all baselines on WikiText PPL and >= magnitude on kinship."
    elif stats.get("dmt_beats_magnitude_ppl", False):
        stats["verdict"] = "PARTIAL: DMT beats magnitude on PPL but not all baselines / accuracy."
    else:
        stats["verdict"] = "NULL: DMT does not beat magnitude on WikiText PPL."

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    s = compute_stats()
    print(json.dumps(s, indent=2))
