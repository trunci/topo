"""Adversarial confound checks on the GREEN verdict.

The danger: H1 features might just track sequence length (2-hop prompts differ in
token count from 1-hop). If so, the 'reasoning' signal is a graph-size artifact.
Also stratify by family (kinship is the clean minimal pair) and by the better
length-normalized metric.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

df = pd.read_parquet("results/spike.parquet")
out = open("results/confound_report.txt", "w")
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s); out.write(s + "\n")

p("=== shape ===", df.shape)

# --- 1. Sequence-length confound ---
# Per pair, does 2-hop have systematically different seq_len than 1-hop?
sl = df.drop_duplicates("item_id")[["pair_id", "hop", "seq_len", "family"]]
piv = sl.pivot_table(index="pair_id", columns="hop", values="seq_len")
dlen = (piv[2] - piv[1]).dropna()
p("\n[seq_len] mean 1-hop=%.2f  mean 2-hop=%.2f  mean diff(2-1)=%.3f  std=%.3f"
  % (piv[1].mean(), piv[2].mean(), dlen.mean(), dlen.std()))
p("[seq_len] pairs where 2hop longer: %d / %d ; identical length: %d"
  % ((dlen > 0).sum(), len(dlen), (dlen == 0).sum()))

# --- 2. Family stratification of the seq_len gap ---
for fam, g in sl.groupby("family"):
    gp = g.pivot_table(index="pair_id", columns="hop", values="seq_len")
    d = (gp[2] - gp[1]).dropna()
    p("[seq_len/%s] mean diff(2-1)=%.3f  std=%.3f  n=%d" % (fam, d.mean(), d.std(), len(d)))

# --- 3. Direction balance of significant heads (length artifact => one-sided) ---
def paired_table(d, metric):
    rows = []
    for (L, H), g in d.groupby(["layer", "head"]):
        pv = g.pivot_table(index="pair_id", columns="hop", values=metric).dropna()
        if 1 not in pv.columns or 2 not in pv.columns or len(pv) < 5:
            continue
        diff = pv[2].values - pv[1].values
        pval = 1.0 if np.allclose(diff, 0) else wilcoxon(pv[2].values, pv[1].values)[1]
        rows.append({"layer": L, "head": H, "mean_diff": diff.mean(), "p": pval})
    r = pd.DataFrame(rows)
    r["p_adj"] = multipletests(r["p"], method="fdr_bh")[1]
    r["sig"] = r["p_adj"] < 0.05
    return r

r = paired_table(df, "total_persistence")
sig = r[r.sig]
p("\n[direction] sig heads=%d  of which 2hop>1hop: %d  2hop<1hop: %d"
  % (len(sig), (sig.mean_diff > 0).sum(), (sig.mean_diff < 0).sum()))
p("[direction] => two-sided structure is evidence AGAINST a pure length artifact"
  if (sig.mean_diff < 0).sum() > 0 else "[direction] one-sided: length artifact PLAUSIBLE")

# --- 4. Length-controlled metric: cycles per edge-ish (normalize persistence by seq_len) ---
df["persist_per_tok"] = df["total_persistence"] / df["seq_len"]
r2 = paired_table(df, "persist_per_tok")
p("\n[length-normalized metric persist_per_tok] sig heads after BH = %d" % r2.sig.sum())

# --- 5. Kinship-only (cleanest minimal pair: 1-word question difference) ---
rk = paired_table(df[df.family == "kinship"], "total_persistence")
p("[kinship-only] sig heads after BH = %d (of %d tested)" % (rk.sig.sum(), len(rk)))
ro = paired_table(df[df.family == "ordering"], "total_persistence")
p("[ordering-only] sig heads after BH = %d (of %d tested)" % (ro.sig.sum(), len(ro)))

# --- 6. Overlap: are kinship-sig and ordering-sig heads the same? (robustness) ---
ks = set(zip(rk[rk.sig].layer, rk[rk.sig].head))
os_ = set(zip(ro[ro.sig].layer, ro[ro.sig].head))
p("[overlap] kinship-sig=%d ordering-sig=%d  intersection=%d"
  % (len(ks), len(os_), len(ks & os_)))

out.close()
print("\nWROTE results/confound_report.txt")
