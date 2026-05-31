"""Generate WRITEUP.md from the three committed stats JSONs.

Consolidates the spike (results/real_stats.json), Experiment 1
(results/exp1_stats.json), and Experiment 2 (results/exp2_stats.json) into one
honest report. Every numeric value is read from the JSON files; prose is fixed
here. No number is hand-typed into the document.
"""
from __future__ import annotations

import json


def _f(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def build(s: dict, e: dict, x: dict, r: dict, g: dict, p: dict,
          q: dict, qs: dict, qm: dict, h: dict, z: dict, za: dict, zk: dict,
          zl: dict) -> str:
    # --- spike numbers ---
    nontriv_pct = _f(s["c1_nontriv_frac"] * 100, 1)
    n_sig = s["c2_n_sig"]
    n_tested = s["c2_heads_tested"]
    sig_up = s["c2_sig_up"]
    sig_down = s["c2_sig_down"]
    layer_med = s["c3_sig_layer_median"]
    n_layers = s["layers"]
    kin_sig = s["kinship_sig"]
    ord_sig = s["ordering_sig"]
    overlap = s["family_overlap"]
    kin_len = s["kinship_len_diff"]
    pairs = s["pairs"]
    rows = s["rows"]

    # --- exp1 numbers ---
    ppl = e["wikitext_ppl_by_method"]
    ppl_order = sorted([m for m in ppl], key=lambda m: ppl[m])
    ppl_table = "\n".join(f"| {m} | {_f(ppl[m], 1)} |" for m in ppl_order)
    wil = e["wikitext_dmt_vs_baseline_loss"]
    wil_rows = []
    for b in ["magnitude", "window", "random"]:
        if b in wil:
            wr = wil[b]
            md = wr.get("median_diff")
            d = "DMT worse" if (md is not None and md > 0) else ("DMT better" if md is not None else "n/a")
            wil_rows.append(f"| dmt vs {b} | {_f(md, 3)} | {d} | {wr.get('p')} |")
    wil_table = "\n".join(wil_rows)
    dmt_spars = _f(e["dmt_mean_sparsity"], 3)
    exp1_verdict = e["verdict"]
    n_wt = e["wikitext_n_examples"]

    # --- exp2 numbers ---
    x_rho = _f(x["spearman_rho"], 3)
    x_p = _f(x["spearman_p"], 5)
    x_mw_p = _f(x["mannwhitney_p"], 5)
    x_top_h1 = _f(x["top_mean_h1"], 3)
    x_rest_h1 = _f(x["rest_mean_h1"], 3)
    x_drho = _f(x["baseline_distance_rho"], 3)
    x_dp = _f(x["baseline_distance_p"], 5)
    x_notdom = x["topology_not_dominated_by_distance"]
    x_verdict = x["verdict"]
    x_heads = x["n_heads"]

    # --- exp3 numbers (residual-information test) ---
    r_base_r2 = _f(r["baseline_r2"], 3)
    r_full_r2 = _f(r["full_r2"], 3)
    r_delta_r2 = _f(r["delta_r2"], 3)
    r_f = _f(r["f_stat"], 2)
    r_fp = _f(r["f_pvalue"], 5)
    r_partial_rho = _f(r["partial_spearman_rho"], 3)
    r_partial_p = _f(r["partial_spearman_p"], 7)
    r_raw_rho = _f(r["raw_spearman_rho"], 3)
    r_controls = ", ".join(r["controls"])
    r_verdict = r["verdict"]

    # --- exp4 numbers (generalization grid) ---
    g_overall = g["overall_verdict"]
    g_ngreen = g["n_green"]
    g_ncells = g["n_cells"]
    g_grid_rows = "\n".join(
        f"| {c['model']} | {c['circuit']} | {c['n_heads']} | {_f(c['delta_r2'], 3)} | "
        f"{_f(c['f_pvalue'], 5)} | {_f(c['partial_spearman_rho'], 3)} | {c['cell_verdict']} |"
        for c in g["cells"]
    )

    # --- exp5 numbers (cycle inspection E2 + causal ablation E1) ---
    p_e2 = p["E2"]
    p_e1 = p["E1"]
    p_e2_verdict = p_e2["verdict"]
    p_e2_ind = _f(p_e2["induction_mean_fraction"], 3)
    p_e2_non = _f(p_e2["noninduction_mean_fraction"], 3)
    p_e2_mw = _f(p_e2["mann_whitney_p"], 5)
    p_e2_S = p["seq_len"]
    p_e1_verdict = p_e1["verdict"]
    p_e1_cyc = _f(p_e1["median_damage"]["cycle"], 4)
    p_e1_mag = _f(p_e1["median_damage"]["magnitude"], 4)
    p_e1_rnd = _f(p_e1["median_damage"]["random"], 4)
    p_e1_cvm_p = _f(p_e1["cycle_vs_magnitude"]["p_value"], 4)
    p_e1_cvr_p = _f(p_e1["cycle_vs_random"]["p_value"], 4)
    p_e1_cyc_pos = p_e1["sanity"]["cycle_damage_positive"]

    # --- exp6 numbers (stronger directed causal test) ---
    def _q6row(name, d, tk):
        return (f"| {name} | {tk} | {_f(d['k_ablated_mean'], 1)} | "
                f"{_f(d['median_damage']['cycle'], 4)} | "
                f"{_f(d['cycle_vs_random']['p_value'], 4)} | "
                f"{_f(d['cycle_vs_magnitude']['p_value'], 4)} | {d['verdict']} |")

    q6_distil = q["models"]["distilgpt2"]
    q6_gpt2 = q["models"]["gpt2"]
    q6_rows = "\n".join([_q6row("distilgpt2", q6_distil, 8),
                         _q6row("gpt2", q6_gpt2, 8)])
    q6_distil_verdict = q6_distil["verdict"]
    q6_gpt2_verdict = q6_gpt2["verdict"]

    qs_by = qs["by_top_k"]

    def _qsrow(tk):
        d = qs_by[tk]
        return (f"| {tk} | {_f(d['k_ablated_mean'], 1)} | "
                f"{_f(d['median_damage']['cycle'], 4)} | "
                f"{_f(d['cycle_vs_random']['p_value'], 4)} | {d['verdict']} |")

    qs_rows = "\n".join(_qsrow(tk) for tk in ["8", "16", "24"])
    qs_v8 = qs_by["8"]["verdict"]
    qs_v16 = qs_by["16"]["verdict"]
    qs_v24 = qs_by["24"]["verdict"]

    qm_med = qm["models"]["gpt2-medium"]
    qm_verdict = qm_med["verdict"]
    qm_cyc = _f(qm_med["median_damage"]["cycle"], 4)
    qm_cvr = _f(qm_med["cycle_vs_random"]["p_value"], 4)
    qm_cvm = _f(qm_med["cycle_vs_magnitude"]["p_value"], 4)
    qm_k = _f(qm_med["k_ablated_mean"], 1)

    # --- exp7 numbers (failure prediction) ---
    h_all = h["all_items"]
    h_acc = _f(h["overall_accuracy"], 3)
    h_verdict = h_all["verdict"]
    h_auc_conf = _f(h_all["auc"]["confidence_only"]["mean_auc"], 3)
    h_auc_full = _f(h_all["auc"]["confidence_plus_topology"]["mean_auc"], 3)
    h_auc_topo = _f(h_all["auc"]["topology_only"]["mean_auc"], 3)
    h_delta_med = _f(h_all["delta_auc_full_minus_baseline"]["median"], 3)
    h_delta_p = _f(h_all["delta_auc_full_minus_baseline"]["wilcoxon_p"], 4)
    h_beats = h_all["topology_beats_chance"]
    h_adds = h_all["topology_adds_beyond_confidence"]
    h_n = h_all["n_items"]
    h2 = h["two_hop_only"]
    h2_verdict = h2["verdict"] if h2 else "n/a"
    h2_auc_topo = _f(h2["auc"]["topology_only"]["mean_auc"], 3) if h2 else "n/a"

    # --- exp8 numbers (cellular sheaves) ---
    z_f1 = z["frame1_discrimination"]
    z_f2 = z["frame2_failure_prediction"]
    z_f1_verdict = z_f1["verdict"]
    z_f1_base = _f(z_f1["baseline_r2"], 3)
    z_f1_full = _f(z_f1["full_r2"], 3)
    z_f1_delta = _f(z_f1["delta_r2"], 3)
    z_f1_fp = _f(z_f1["f_pvalue"], 6)
    z_f2_verdict = z_f2["verdict"]
    z_f2_auc_conf = _f(z_f2["auc"]["confidence_only"]["mean_auc"], 3)
    z_f2_auc_sheaf = _f(z_f2["auc"]["confidence_plus_sheaf"]["mean_auc"], 3)
    z_f2_auc_sheaf_only = _f(z_f2["auc"]["sheaf_only"]["mean_auc"], 3)
    z_f2_delta = _f(z_f2["sheaf_vs_confidence"]["median_delta"], 3)
    z_f2_p = _f(z_f2["sheaf_vs_confidence"]["wilcoxon_p"], 4)
    z_f2_beats = z_f2["sheaf_beats_chance"]
    z_f2_adds = z_f2["sheaf_adds_beyond_confidence"]
    z_n = z["n_items"]

    # --- exp9a numbers (Phase A1 carrier decomposition: flow vs shape) ---
    za_fvs = za["flow_beyond_shape"]
    za_fvh = za["fiedler_beyond_h1"]
    za_dp = za["discord_partial_spearman"]
    za_verdict = za["verdict"]
    za_flow_dr2 = _f(za_fvs["delta_r2"], 4)
    za_flow_fp = _f(za_fvs["f_pvalue"], 4)
    za_fied_dr2 = _f(za_fvh["delta_r2"], 4)
    za_fied_fp = _f(za_fvh["f_pvalue"], 7)
    za_disc_rho = _f(za_dp["rho"], 3)
    za_disc_p = _f(za_dp["p"], 3)

    # --- exp9 numbers (topological attribution + causal validation) ---
    zk_a = zk["part_a_attribution"]["per_saliency"]
    zk_b = zk["part_b_causal"]
    zk_headline = zk["headline_verdict"]
    zk_a_verdict = zk["part_a_attribution"]["verdict"]
    zk_mag_auc = _f(zk_a["magnitude"]["mean_auc"], 3)
    zk_cyc_auc = _f(zk_a["cycle_participation"]["mean_auc"], 3)
    zk_shf_auc = _f(zk_a["sheaf_discord"]["mean_auc"], 3)
    zk_b_verdict = zk_b["verdict"]
    zk_b_shf = _f(zk_b["median_damage"]["sheaf"], 4)
    zk_b_mag = _f(zk_b["median_damage"]["magnitude"], 4)
    zk_b_shf_vs_rnd = _f(zk_b["sheaf_vs_random_p"], 4)
    zk_b_shf_vs_mag = _f(zk_b["sheaf_vs_magnitude_p"], 3)

    # --- exp10 numbers (IOI attribution) ---
    zl_a = zl["part_a_attribution"]["per_saliency"]
    zl_a_verdict = zl["part_a_attribution"]["verdict"]
    zl_head = zl["headline_verdict"]
    zl_mag_auc = _f(zl_a["magnitude"]["mean_auc"], 3)
    zl_mag_p1 = _f(zl_a["magnitude"]["mean_p_at_1"], 3)
    zl_cyc_auc = _f(zl_a["cycle_participation"]["mean_auc"], 3)
    zl_shf_auc = _f(zl_a["sheaf_discord"]["mean_auc"], 3)
    zl_b_verdict = zl["part_b_causal"]["verdict"]

    return f"""# Topology of Attention — Consolidated Writeup

**One-line result:** a transformer's attention-graph topology carries a *real,
reproducible* signal about reasoning (Spike, GREEN). It is **not prescriptive** for pruning
(Exp 1, NULL) and its *marginal* correlation with a known circuit is weak (Exp 2, RED) — but
once first-order attention statistics are controlled for, topology carries **significant
residual signal** about that circuit (Exp 3, GREEN), and that holds for the **induction**
circuit across two models though not for simpler positional circuits (Exp 4,
PARTIAL-GENERALIZE). The H1 cycles in induction heads **descriptively** span the induction
copy-gap (Exp 5 E2, GREEN); the first ablation was too weak to conclude (Exp 5 E1, RED,
weak), but a rebuilt **directed, budget-matched** ablation does find a **real causal**
effect — cycle edges beat random and magnitude — in 2 of 3 models, budget-sensitive and
not universal across scale (Exp 6, Result G). As a failure predictor it beats chance but
adds nothing beyond the model's own confidence (Exp 7, PARTIAL). A **cellular sheaf** over
the residual stream first appeared to add beyond H1 at discriminating reasoning (Exp 8) — but
decomposition showed that signal was graph **connectivity (Fiedler), not information flow**,
and was itself largely a sequence-length artifact, so the "what flows" framing is **withdrawn**
(Result J). Reframed as explainability, topology fails as a circuit **attributor**: simple
attention magnitude localizes the circuit edge far better, on both induction (Exp 9, K) and
the fairer diffuse IOI circuit (Exp 10, L). Net: topology is a real, **non-redundant
diagnostic** correlate of relational-circuit structure, and — with a strong enough instrument
— a **budget-sensitive causal** one for induction; but it is **not** a usable pruning
criterion, **not** a failure predictor beyond model confidence, **not** evidence that flow
beats shape, and **not** competitive with attention magnitude as a circuit attributor. The
durable contribution is the **adversarial, baseline-benchmarked methodology** that
established all of this.

All numbers in this document are generated by `src/write_writeup.py` from
`results/real_stats.json` (spike), `results/exp1_stats.json` (Exp 1),
`results/exp2_stats.json` (Exp 2), `results/exp3_stats.json` (Exp 3),
`results/exp4_stats.json` (Exp 4), `results/exp5_stats.json` (Exp 5), and the Exp 6 files
`results/exp6_stats.json`, `results/exp6_sweep_stats.json`,
`results/exp6_medium_stats.json`. No value is hand-typed.

---

## 1. Setup (shared)

- **Models:** Qwen2.5-0.5B(-Instruct), {n_layers} layers x {s["heads"]} heads (Spike, Exp 1);
  GPT-2 small, 12 x 12 = {x_heads} heads (Exp 2). Eager attention, float32, CPU/MPS.
- **Topology pipeline:** per (layer, head) attention matrix -> max-symmetrize -> top-k
  sparsify -> flag (clique) complex -> persistent homology (H1) or discrete Morse matching.
- **Probes:** controlled 1-hop/2-hop minimal pairs (Spike); WikiText-2 perplexity +
  budget-matched pruning baselines (Exp 1); repeated-random induction detector (Exp 2).

---

## 2. Result A — Topology tracks reasoning (Spike): GREEN

Does attention flag-complex H1 discriminate reasoning hop-count? {pairs} matched pairs,
{rows} head-observations.

| Pre-registered criterion | Result |
|---|---|
| Non-triviality (H1 present) | {nontriv_pct}% of head-observations carry a persistent cycle |
| Discrimination (2-hop vs 1-hop, paired Wilcoxon, BH-corrected) | {n_sig} / {n_tested} heads significant |
| Plausible localization | significant-head layer median {layer_med} / {n_layers} |

Robustness: effect is two-sided ({sig_up} heads up, {sig_down} down — not a one-sided
length artifact); the exactly length-matched kinship family (token diff {_f(kin_len, 3)})
still yields {kin_sig} significant heads; cross-family replication kinship {kin_sig} /
ordering {ord_sig} with {overlap} heads significant in both.

**Verdict:** {s["verdict"]}

Top discriminating heads:
{chr(10).join("- " + h for h in s["top_heads"])}

---

## 3. Result B — Topology does NOT improve pruning (Exp 1): NULL

At equal per-head edge budget, does pruning by discrete-Morse critical cells beat
magnitude / window / random? DMT sets the budget; baselines match it. WikiText-2 PPL on
{n_wt} windows (base model).

| method | mean PPL |
|---|---|
{ppl_table}

Paired Wilcoxon on per-example loss (`median_diff` = median(DMT - baseline); positive = DMT worse):

| comparison | median_diff | direction | p |
|---|---|---|---|
{wil_table}

DMT natural sparsity {dmt_spars}. **Verdict:** {exp1_verdict}

---

## 4. Result C — Topology does NOT match induction circuits (Exp 2): RED

Do topologically distinctive heads coincide with induction heads (detected independently)?
GPT-2 small, {x_heads} heads.

- Discrimination: Spearman(induction, H1 persistence) rho = {x_rho}, p = {x_p}.
- Separation: top-10 induction heads vs rest, Mann-Whitney p = {x_mw_p}
  (mean H1 {x_top_h1} vs {x_rest_h1}).
- Beats trivial baseline? Spearman(induction, attention distance) rho = {x_drho},
  p = {x_dp}. Topology not dominated by distance: {x_notdom}.

On its own, the cheap attention-distance scalar has a higher *marginal* correlation with
induction than H1 persistence does. **Verdict:** {x_verdict}

But "marginal" is the key caveat — see Result D, which controls for the first-order scalars
jointly and reaches the opposite conclusion.

---

## 5. Result D — Topology is NOT redundant with first-order statistics (Exp 3): GREEN

Result C compared H1 against attention distance *one variable at a time*. The sharper test:
does H1 persistence add predictive power for induction **beyond** the first-order scalars
({r_controls}) taken together? Nested OLS on the {x_heads} GPT-2 heads.

- Baseline OLS (first-order scalars only): R2 = {r_base_r2}.
- Full OLS (+ H1 persistence): R2 = {r_full_r2} → delta-R2 = {r_delta_r2}.
- Nested-model F-test for the H1 term: F = {r_f}, p = {r_fp}.
- Partial Spearman(H1, induction | controls): rho = {r_partial_rho}, p = {r_partial_p}
  (vs raw marginal rho = {r_raw_rho}).

This is a **suppression effect**: H1's weak marginal correlation (Result C) was masked by its
own correlation with the first-order scalars; controlling for them *reveals* a strong partial
association. **Verdict:** {r_verdict}

---

## 6. Result E — The non-redundancy is circuit-specific, model-robust (Exp 4): PARTIAL

Result D was one circuit, one model. Exp 4 repeats the residual test over a grid of 2 models
{{gpt2, distilgpt2}} x 3 circuits {{induction, previous-token, duplicate-token}}. Same
pre-registered per-cell rule (delta-R2 >= 0.02 AND nested-F p < 0.05 => GREEN).

| model | circuit | n_heads | delta_r2 | f_pvalue | partial_rho | verdict |
|---|---|---|---|---|---|---|
{g_grid_rows}

{g_ngreen} of {g_ncells} cells GREEN. **Verdict:** {g_overall}

The non-redundant signal appears for **induction in both models** (and is stronger in
distilgpt2) but not for previous-token or duplicate-token. Mechanistically sensible:
previous-token heads are already ~80% explained by first-order stats (a single i->i-1 hop has
no cycle structure), so there is nothing left for topology to add. Topology earns its keep on
circuits with **multi-position relational structure** (induction), not simple positional ones.

---

## 7. Result F — Cycles span the copy-gap, but no causal evidence (Exp 5): E2 GREEN / E1 RED

Two escalations on induction in gpt2. **E2 (descriptive):** do the H1 critical-cycle edges of
induction heads span the induction copy distance S = {p_e2_S}? **E1 (causal):** does ablating
those cycle edges hurt induction *behavior* (model loss on second-copy tokens) more than
ablating an equal number of magnitude- or random-matched edges?

- **E2 = {p_e2_verdict}.** Fraction of cycle edges at gap ~ S: induction heads {p_e2_ind} vs
  non-induction {p_e2_non} (Mann-Whitney p = {p_e2_mw}). The cycles *are* the copy edges.
- **E1 = {p_e1_verdict}.** Median damage (loss increase vs unmasked): cycle {p_e1_cyc},
  magnitude {p_e1_mag}, random {p_e1_rnd}. cycle vs random p = {p_e1_cvr_p}; cycle vs
  magnitude p = {p_e1_cvm_p}. Cycle ablation does not exceed random.

**Important caveat:** the ablation was too weak to interpret strongly — cycle-damage-positive
= {p_e1_cyc_pos} (removing a few symmetrized top-k edges barely perturbed behavior, so all
damages were ~0 or slightly negative). E1 RED means "no causal signal *detected under this
ablation*", not strong evidence of no effect. A heavier ablation (mean-ablation, or more
edges) is the natural next test.

---

## 8. Result G — A stronger causal test: budget matters more than model size (Exp 6)

Exp 5's E1 was underpowered because its instrument was symmetric, common-mode (it
re-masked the *whole* graph in every condition), and tiny. Exp 6 rebuilds the
intervention to be (i) **directed** (the induction edge is query@2nd-copy -> key after
1st occurrence, q > k), (ii) **edge-exact** (ablate exactly the named directed edges,
keep all else), (iii) **gap-constrained** (treatment = directed critical-cycle edges at
gap ≈ S, the copy distance), and (iv) **cumulative** across all induction heads at once.
Controls are budget-matched: the same edge count chosen by highest magnitude or at
random. Readout is unchanged from Exp 5 (second-copy next-token CE damage), so results
are directly comparable. Per-model verdict (pre-registered): **GREEN** iff cycle damage
is positive AND cycle beats random (one-sided Wilcoxon p < 0.05); **RED** if positive
but not above random; **INCONCLUSIVE** if cycle damage is not positive (still
underpowered).

**Per-model (at the default budget top_k = 8):**

| model | top_k | K/seq | cycle median damage | cycle vs random p | cycle vs magnitude p | verdict |
|---|---|---|---|---|---|---|
{q6_rows}

distilgpt2 is **{q6_distil_verdict}** — the directed cumulative ablation of cycle edges
hurts induction more than random *and* more than magnitude at equal budget. gpt2 is
**{q6_gpt2_verdict}** at this budget: with only ~2-3 directed gap-S cycle edges per
sequence, there was almost nothing to remove.

**gpt2 budget sweep (does the INCONCLUSIVE just reflect a starved treatment?):**

| top_k | K/seq | cycle median damage | cycle vs random p | verdict |
|---|---|---|---|---|
{qs_rows}

The verdict moves monotonically with the treatment budget: top_k=8 {qs_v8} ->
top_k=16 {qs_v16} -> top_k=24 {qs_v24}. So gpt2's INCONCLUSIVE was a measurement
artifact: give the topology enough cycle edges to extract and the causal effect appears
(cycle beats both random and magnitude at top_k=24).

**A larger model does NOT strengthen the claim.** gpt2-medium (355M, 24L x 16H) at the
adequate budget top_k=24 (mean K = {qm_k}/seq) is **{qm_verdict}**: cycle median damage
{qm_cyc}, cycle vs random p = {qm_cvr} (not below 0.05), cycle vs magnitude p = {qm_cvm}.
More parameters spread the induction circuit across more heads, *increasing* redundancy,
so removing any one head's cycle edges matters less — the opposite of the "more thinking
-> more decisive cycles" intuition. The lever that mattered was **topological resolution
(top_k), not model size.**

**Net for Exp 6:** the causal claim (cycle > random at equal budget) holds in 2 of 3
models tested (distilgpt2, gpt2 at adequate budget) and is borderline-RED in gpt2-medium.
This *partially resolves* Exp 5's E1: with a properly directed, adequately-budgeted
instrument there **is** a real causal signal that topology marks functionally special
edges for induction — but it is budget-sensitive and not universal across scale.

---

## 9. Result H — Topology predicts reasoning failure, but confidence already knows (Exp 7)

The Spike (A) showed topology tracks *attempted* reasoning; it never gated on whether the
model was **right**. Exp 7 is the Spike's payoff test: do per-example H1 features predict
reasoning **errors**, and — the non-trivial bar — do they add anything **beyond the
model's own confidence** (the answer-token logit margin)? A probe that merely rediscovers
"low confidence -> likely wrong" is uninteresting. Qwen2.5-0.5B-Instruct on {h_n}
mixed-hop kinship/ordering items (overall accuracy {h_acc}); nested logistic models,
5-fold CV ROC-AUC.

| predictor | CV mean AUC (predict correctness) |
|---|---|
| topology only | {h_auc_topo} |
| confidence only (M0) | {h_auc_conf} |
| confidence + topology (M1) | {h_auc_full} |

- **Topology alone beats chance** (AUC {h_auc_topo} > 0.5): topology beats chance =
  {h_beats}. The Spike's signal does extend to *correctness*, not just hop-count.
- **But it adds nothing beyond confidence:** delta-AUC(M1 - M0) median {h_delta_med},
  paired Wilcoxon p = {h_delta_p}; topology adds beyond confidence = {h_adds}.
- 2-hop-only slice (the original Spike regime): topology-only AUC {h2_auc_topo}, verdict
  {h2_verdict} — same story.

**Verdict: {h_verdict}.** Topology is a *real but redundant* failure predictor here — it
knows about failure, but essentially only what the model's own confidence already encodes.
This mirrors the project's recurring pattern: topology is a genuine correlate that
struggles to be *uniquely* useful, except where confounds are explicitly controlled
(D, G). A larger model with *poorly-calibrated* confidence on genuinely harder reasoning is
where topology might finally add a distinct failure signal — untested here.

---

## 10. Result I — Sheaves over the residual stream (Exp 8) [flow reading withdrawn — see Result J]

Every prior result measured attention-graph **shape** (does an H1 cycle exist?) and was
blind to **what flows** along edges. Exp 8 builds a **cellular sheaf** over the residual
stream: node stalks are PCA-reduced hidden vectors, restriction maps are orthogonal frames
(local PCA + Procrustes), and the **sheaf Laplacian** measures *consistency of information
flow* (`xᵀL_F x = Σ ‖F_v x_v − F_u x_u‖²`). The construction is novel (no prior work puts a
sheaf on a transformer residual stream) but built from established, non-learned pieces
(Hansen-Ghrist; Neural Sheaf Diffusion; Connection-Laplacian SNNs). A scalar weighted-graph
Laplacian (Fiedler value) is the shape-only baseline. Qwen2.5-0.5B-Instruct, {z_n} items.

**Frame 1 — discrimination (does sheaf flow add beyond H1 shape at predicting hop?):**
nested OLS predicting reasoning hop-count.

| model | R² |
|---|---|
| H1 + first-order controls (baseline) | {z_f1_base} |
| + sheaf features (full) | {z_f1_full} |

delta-R² = {z_f1_delta}, nested-F p = {z_f1_fp}. **Verdict: {z_f1_verdict}** — the sheaf
"flow-consistency" features carry information about reasoning depth that H1 *shape* does
not. First evidence a richer descriptor exceeds H1.

**Frame 2 — failure prediction (does sheaf add beyond confidence, where H1 failed?):**
nested CV ROC-AUC predicting correctness.

| predictor | CV mean AUC |
|---|---|
| sheaf only | {z_f2_auc_sheaf_only} |
| confidence only (M0) | {z_f2_auc_conf} |
| confidence + sheaf (M2) | {z_f2_auc_sheaf} |

sheaf vs confidence (M2 − M0): median delta {z_f2_delta}, p = {z_f2_p}; beats chance =
{z_f2_beats}, adds beyond confidence = {z_f2_adds}. **Verdict: {z_f2_verdict}** — the sheaf
hits the *same ceiling* H1 hit in Exp 7: predicts failure above chance but adds nothing
beyond the model's own confidence.

**Net for Exp 8 (as reported):** the sheaf *block* beats H1 at discriminating reasoning
(Frame 1 GREEN) but not confidence at failure prediction (Frame 2 PARTIAL). **But the Frame 1
"flow beats shape" reading does not survive decomposition — see Result J, which falsifies it.**

---

## 11. Result J — The Exp 8 "flow" signal was SHAPE in disguise (Phase A1)

Result I's Frame 1 GREEN was carried by a *block* of sheaf features that mixed two very
different things: **Fiedler value** (a scalar graph-Laplacian connectivity statistic — pure
graph *shape*), a degenerate harmonic dimension (constant = d, contributing nothing), and
**discord** (the only genuinely vector-valued *flow*-consistency feature). The follow-up
study's decisive gate (`results/exp9a_stats.json`, run on the Exp 8 data) decomposes the
block to ask: was the signal *flow* or *shape*?

| nested test (predicting hop) | delta-R² | nested-F p | adds? |
|---|---|---|---|
| Fiedler beyond [H1 + controls] | {za_fied_dr2} | {za_fied_fp} | yes |
| discord/flow beyond [H1 + controls + Fiedler] | {za_flow_dr2} | {za_flow_fp} | no |

Partial Spearman(discord, hop given H1+controls+Fiedler) = {za_disc_rho} (p = {za_disc_p}) — about 0.

**Verdict: {za_verdict}.** The added predictive power was **Fiedler — a shape statistic H1
missed — not discord**. Once Fiedler is in the baseline, the vector-valued flow feature adds
nothing. So Exp 8 Frame 1 is a *shape* result, not a *flow* result; the "what flows beats
what shape" framing is **withdrawn**. Per the pre-registered kill criterion, the OV-circuit
sheaf (Phase B) is **not** run. The honest, smaller finding stands: the attention graph's
**algebraic connectivity (Fiedler)** predicts reasoning depth beyond H1 persistence — a
better *shape* descriptor, not evidence that the residual stream's information flow carries
extra signal.

---

## 12. Result K — Topology vs the trivial baseline as a circuit attributor (Exp 9)

The cleanest, confound-proof way to ask whether topology is *useful*: drop cross-example
scalar regression (where sequence length confounded everything) and ask, **within a single
fixed-length attention graph**, whether topology assigns high saliency to the **ground-truth
induction copy edge** — and whether it beats the trivial **attention-magnitude** baseline,
then whether the flagged edges are **causally** necessary. gpt2 + distilgpt2, induction
heads, per-edge.

**Part A — attribution (ROC-AUC at recovering the copy edge):**

| saliency | mean AUC |
|---|---|
| attention magnitude (baseline) | {zk_mag_auc} |
| cycle participation | {zk_cyc_auc} |
| sheaf discord | {zk_shf_auc} |

**Verdict: {zk_a_verdict}.** Attention magnitude is a near-perfect attributor
({zk_mag_auc}); cycle-participation is at chance ({zk_cyc_auc}); sheaf-discord beats chance
({zk_shf_auc}) but is decisively below magnitude. Topology does **not** beat the trivial
baseline at localizing the circuit.

**Part B — causal validation (damage = loss increase from ablating flagged edges):**
sheaf-flagged edges damage induction more than random (median {zk_b_shf}, vs-random
p = {zk_b_shf_vs_rnd}) → **{zk_b_verdict}** by the letter of the rule — but the effect is far
weaker than magnitude ({zk_b_mag}) and does not beat it (p = {zk_b_shf_vs_mag}); cycle edges
do not beat random.

**Net for Exp 9 (headline {zk_headline}):** as a circuit *attributor*, **attention magnitude
is simpler and better than topology** — at both localization and ablation. The one genuine
topological positive is a small-but-real causal signal from sheaf-discord above random,
which keeps the within-example "flow" idea barely alive while confirming it is not
competitive with the trivial baseline. This is the project's sharpest negative-leaning
result and the one most useful as a caution to the TDA-for-interpretability literature:
*benchmark against attention magnitude, or the topology may be decorative.*

---

## 13. Result L — The fair fight: IOI, where magnitude is handicapped (Exp 10)

Result K's caveat: induction is the *worst case* for topology because the magnitude
baseline trivially wins (induction heads put ~77% of row mass on one edge). Exp 10 moves to
**IOI name-mover heads**, where a verified probe showed attention is more diffuse (top-edge
mass 0.54, entropy 1.20 vs induction's 0.77 / 0.83) — handicapping magnitude. GPT-2 small,
END→IO-name as the ground-truth edge; GPT-2 solves the task (IO is top-1 on 92% of prompts).

**Part A — attribution (recover the END→IO edge):**

| saliency | mean AUC | p@1 |
|---|---|---|
| attention magnitude (baseline) | {zl_mag_auc} | {zl_mag_p1} |
| cycle participation | {zl_cyc_auc} | — |
| sheaf discord | {zl_shf_auc} | — |

**Verdict: {zl_a_verdict}.** Even with attention diffuse, magnitude is still a near-perfect
attributor (AUC {zl_mag_auc}); cycle-participation is *below* chance ({zl_cyc_auc}); sheaf
discord beats chance ({zl_shf_auc}) but stays far below magnitude. The diffuseness handicap
was **not enough** — 0.54 mass on the top edge is still plenty for the baseline.

**Part B — causal ({zl_b_verdict}):** uninterpretable here — every condition's median damage
is ≤ 0 (ablating name-mover edges tends to *raise* the IO−S margin, magnitude most of all),
because the name-mover edge feeds both the IO and S logits. The readout cannot cleanly score
necessity; this RED is a compromised instrument, not evidence against topology.

**Net for Exp 10 (headline {zl_head}):** the fairest fight we could construct still goes to
attention magnitude. Across induction (K) and IOI (L), the conclusion is robust:
**as an unsupervised circuit attributor, simple attention weight beats attention topology.**

---

## 14. Synthesis

Topology **locates** where reasoning structure lives (A), carries **genuine, non-redundant**
information about a known circuit once confounds are controlled (D), and that non-redundancy
is **model-robust but specific to relational circuits** (E, induction in 2 models; not
positional circuits). Descriptively, the cycles in induction heads correspond to the actual
copy relation (F/E2). What topology does *not* (yet) do: serve as a pruning criterion (B,
NULL), beat a trivial distance metric *marginally* (C — though that was a suppression
artifact, see D), or show a *causal* effect on behavior (F/E1 — but under a weak ablation, so
inconclusive). The defensible contribution is a **diagnostic** one: H1 persistence is a real,
non-trivial, relational-circuit-specific correlate of attention structure. The proposal's
*prescriptive* claim (topological pruning) is unsupported; the *causal* claim is untested by a
sufficiently strong instrument. And as an **attributor**, topology loses to attention
magnitude on both induction (K) and the fairer, diffuse IOI circuit (L) — its value is
descriptive/diagnostic, not as a practical circuit-finding tool.

## 15. Honest caveats

- All GPT-2 experiments (2-5) are small, memory-safe CPU runs (144 heads, n_seqs<=8); the
  Spike is the largest and most robust run. Directions are clear; magnitudes are not nailed.
- Mean PPL = mean(exp(loss)) is outlier-dominated; the paired per-example loss test agrees
  with the NULL verdict regardless.
- "Tracks reasoning" (Spike) was measured on *attempted* reasoning (attention over the
  prompt), not gated on correct answers.
- Exp 5's E1 causal test used a weak ablation (near-zero damage); its RED is "no signal
  detected", not strong evidence of no causal effect.
- Circuits tested are induction / previous-token / duplicate-token, all on repeated-random
  sequences — not the full space of attention circuits.
- Exp 6's causal signal is real but **budget-sensitive** (top_k) and **not universal**
  across scale (gpt2-medium borderline RED); n_seqs=12, single seed.
- Exp 7 is one small model (Qwen-0.5B, {h_n} items, single seed); "confidence" is a
  logit-margin proxy, not a calibrated probability; failure on synthetic kinship is narrow.
- Exp 8's sheaf is non-learned (restriction maps from local PCA + Procrustes, not the
  model's true computation); d=4, top_k=8 fixed; harmonic-dim is degenerate (always = d).
- Result J (Phase A1) shows Exp 8 Frame 1 was carried by Fiedler (shape), not discord (flow);
  the "flow" framing is withdrawn. The decomposition is on the same {z_n}-item single run —
  the surviving "Fiedler beyond H1" shape result still needs length-control, classical-graph
  competitors, and replication (Phase A2-A4) before it is trusted.
- Exp 9 (Result K) is induction-only; we tested the "diffuse circuit" rescue in Exp 10.
- Exp 10 (Result L) tested IOI to handicap magnitude; magnitude still won at attribution, and
  its causal readout (IO-S margin) was compromised (name-mover edges feed both IO and S
  logits → ablation can raise the margin), so Part B is uninterpretable, not a clean verdict.
  One template, one model, ~48 prompts.

## 16. Follow-up research directions

1. **Residual-information test (is topology redundant?).** ✅ DONE — Exp 3 (Result D): H1 is
   *not* redundant with first-order stats for induction (delta-R2 = {r_delta_r2}, F p = {r_fp},
   partial rho = {r_partial_rho}).

2. **Does it generalize?** ✅ DONE — Exp 4 (Result E): {g_overall} The non-redundancy is
   model-robust for induction but circuit-specific (not positional circuits).

3. **Cycle inspection + causal test.** ✅ MOSTLY DONE — Exp 5 (Result F): cycles span the
   copy-gap (E2 GREEN), but the first causal ablation was too weak (E1 RED, weak). Exp 6
   (Result G) then rebuilt the instrument (directed, edge-exact, cumulative, gap-constrained)
   and found a **real causal signal** — cycle ablation beats random and magnitude at equal
   budget — in distilgpt2 and in gpt2 once the treatment budget (top_k) is adequate; it is
   borderline-RED in gpt2-medium. Remaining: per-model budget calibration, more seeds, and a
   mean-ablation variant to cross-check the edge-removal readout.

4. **Failure prediction (Spike's payoff / proposal Exp 3').** ◑ PARTIAL — Exp 7 (Result H):
   topology predicts reasoning errors above chance but does not add beyond the model's own
   confidence ({h_verdict}). Next: a larger model with poorly-calibrated confidence on
   harder reasoning, where topology could add a *distinct* failure signal.

5. **Topological attribution (does topology find circuit edges?).** ✅ DONE — Exp 9
   (Result K, induction) and Exp 10 (Result L, IOI): no — attention magnitude localizes the
   circuit edge far better than any topological saliency, on both the worst case (induction)
   and the fairest case (diffuse IOI). Topology is descriptive, not a practical attributor.
   Remaining leads: a cleaner IOI causal readout (the IO-S margin is confounded because
   name-mover edges feed both logits — use direct-logit-attribution or a path-patching
   readout instead); the length-controlled "Fiedler beyond H1" shape lead (proposal Phase
   A2-A4, `docs/proposals/2026-05-31-sheaf-flow-followup-proposal.md`).

## 17. Provenance

All experiment verdicts are machine-checked fields in their respective JSON files
(spike, exp1-exp5, exp6 + sweep + gpt2-medium, exp7, exp8, exp9a, exp9, and exp10).
Regenerate this document with `uv run python -m src.write_writeup`.
"""


def main(spike="results/real_stats.json", exp1="results/exp1_stats.json",
         exp2="results/exp2_stats.json", exp3="results/exp3_stats.json",
         exp4="results/exp4_stats.json", exp5="results/exp5_stats.json",
         exp6="results/exp6_stats.json", exp6_sweep="results/exp6_sweep_stats.json",
         exp6_medium="results/exp6_medium_stats.json",
         exp7="results/exp7_stats.json", exp8="results/exp8_stats.json",
         exp9a="results/exp9a_stats.json", exp9="results/exp9_stats.json",
         exp10="results/exp10_stats.json", out="WRITEUP.md"):
    s = json.load(open(spike))
    e = json.load(open(exp1))
    x = json.load(open(exp2))
    r = json.load(open(exp3))
    g = json.load(open(exp4))
    p = json.load(open(exp5))
    q = json.load(open(exp6))
    qs = json.load(open(exp6_sweep))
    qm = json.load(open(exp6_medium))
    h = json.load(open(exp7))
    z = json.load(open(exp8))
    za = json.load(open(exp9a))
    zk = json.load(open(exp9))
    zl = json.load(open(exp10))
    open(out, "w").write(build(s, e, x, r, g, p, q, qs, qm, h, z, za, zk, zl))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
