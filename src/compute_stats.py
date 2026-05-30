"""Compute every reported statistic from results/spike.parquet and dump to JSON.

Single source of truth for FINDINGS.md numbers. Hardcodes NO statistics — every
value is computed here so the JSON cannot drift from the data.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def paired(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for (L, H), g in d.groupby(["layer", "head"]):
        pv = g.pivot_table(index="pair_id", columns="hop", values=metric).dropna()
        if 1 not in pv.columns or 2 not in pv.columns or len(pv) < 5:
            continue
        diff = pv[2].values - pv[1].values
        p = 1.0 if np.allclose(diff, 0) else wilcoxon(pv[2].values, pv[1].values)[1]
        rows.append({"layer": int(L), "head": int(H),
                     "mean_diff": float(diff.mean()), "p": float(p)})
    r = pd.DataFrame(rows)
    r["p_adj"] = multipletests(r["p"], method="fdr_bh")[1]
    r["sig"] = r["p_adj"] < 0.05
    return r


def main():
    df = pd.read_parquet("results/spike.parquet")
    res = {"rows": int(len(df)), "pairs": int(df.pair_id.nunique()),
           "items": int(df.item_id.nunique()),
           "layers": int(df["layer"].max() + 1), "heads": int(df["head"].max() + 1)}

    has_cycle = (df["n_cycles"] > 0) & (df["max_persistence"] > 0.05)
    res["c1_nontriv_frac"] = round(float(has_cycle.mean()), 4)

    r = paired(df, "total_persistence")
    sig = r[r.sig]
    n = df["layer"].max() + 1
    res["c2_heads_tested"] = int(len(r))
    res["c2_n_sig"] = int(len(sig))
    res["c2_sig_up"] = int((sig.mean_diff > 0).sum())
    res["c2_sig_down"] = int((sig.mean_diff < 0).sum())
    res["c3_sig_layer_median"] = float(sig.layer.median())
    res["top_heads"] = [
        f"L{int(x.layer)}H{int(x.head)} d={x.mean_diff:+.3f} p_adj={x.p_adj:.1e}"
        for x in r.sort_values("p_adj").head(6).itertuples()
    ]

    sl = df.drop_duplicates("item_id")[["pair_id", "hop", "seq_len", "family"]]
    pv = sl.pivot_table(index="pair_id", columns="hop", values="seq_len")
    dlen = (pv[2] - pv[1]).dropna()
    res["len_1hop_mean"] = round(float(pv[1].mean()), 2)
    res["len_2hop_mean"] = round(float(pv[2].mean()), 2)
    res["len_diff_mean"] = round(float(dlen.mean()), 3)
    res["len_identical_pairs"] = int((dlen == 0).sum())
    res["len_total_pairs"] = int(len(dlen))

    df = df.copy()
    df["ppt"] = df["total_persistence"] / df["seq_len"]
    res["n_sig_lengthnorm"] = int(paired(df, "ppt").sig.sum())

    sig_sets = {}
    for fam in sorted(df.family.unique()):
        rf = paired(df[df.family == fam], "total_persistence")
        res[f"{fam}_sig"] = int(rf.sig.sum())
        res[f"{fam}_tested"] = int(len(rf))
        s = rf[rf.sig]
        sig_sets[fam] = set(zip(s["layer"], s["head"]))
        gf = sl[sl.family == fam].pivot_table(index="pair_id", columns="hop", values="seq_len")
        res[f"{fam}_len_diff"] = round(float((gf[2] - gf[1]).dropna().mean()), 3)
    fams = sorted(sig_sets)
    res["family_overlap"] = int(len(sig_sets[fams[0]] & sig_sets[fams[1]]))

    res["model_accuracy"] = round(float(df.groupby("item_id")["correct"].first().mean()), 4)
    res["verdict"] = "GREEN" if (res["c1_nontriv_frac"] > 0.10 and res["c2_n_sig"] > 0) else "OTHER"

    with open("results/real_stats.json", "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE results/real_stats.json")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
