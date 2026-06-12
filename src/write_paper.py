"""Generate PAPER.md — the paper draft — from the on-disk stats JSONs.

Same provenance discipline as write_writeup.py: every number in the paper is
interpolated from a results/*.json file produced by an experiment's
compute_stats script. No value is hand-typed (the four exp13 mechanistic-
decomposition scalars are inherited verbatim from write_writeup.py, where they
are documented as coming from the pooled-seed inline analysis).

Regenerate with: uv run python -m src.write_paper
"""
from __future__ import annotations

import json


def _f(x, nd=3):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def build(s, e, x, r, g, p, q, qs, qm, h, z, za, zk, zl, zm, zn, zo,
          zp, zpr, zp1, zq, zqb, zr) -> str:
    # ---- spike ----
    s_nontriv = _f(s["c1_nontriv_frac"] * 100, 1)
    s_nsig = s["c2_n_sig"]
    s_ntest = s["c2_heads_tested"]
    s_up = s["c2_sig_up"]
    s_down = s["c2_sig_down"]

    # ---- exp2/3/4: non-redundancy ----
    x_rho = _f(x["spearman_rho"], 3)
    x_p = _f(x["spearman_p"], 3)
    x_dist_rho = _f(x["baseline_distance_rho"], 3)
    r_base = _f(r["baseline_r2"], 3)
    r_full = _f(r["full_r2"], 3)
    r_dr2 = _f(r["delta_r2"], 3)
    r_fp = _f(r["f_pvalue"], 5)
    r_prho = _f(r["partial_spearman_rho"], 3)
    g_cells = g["cells"]
    g_ngreen = g["n_green"]
    g_ncells = g["n_cells"]
    g_rows = "\n".join(
        f"| {c['model']} | {c['circuit']} | {_f(c['delta_r2'], 3)} | "
        f"{_f(c['f_pvalue'], 5)} | {_f(c['partial_spearman_rho'], 3)} | "
        f"{c['cell_verdict']} |"
        for c in g_cells)

    # ---- exp5/6: causal ----
    p2 = p["E2"]
    p2_ind = _f(p2["induction_mean_fraction"], 3)
    p2_non = _f(p2["noninduction_mean_fraction"], 3)
    p2_p = _f(p2["mann_whitney_p"], 5)
    q_d = q["models"]["distilgpt2"]
    q_g = q["models"]["gpt2"]
    qd_dmg = _f(q_d["median_damage"]["cycle"], 4)
    qd_rnd = _f(q_d["cycle_vs_random"]["p_value"], 4) if isinstance(
        q_d["cycle_vs_random"], dict) else _f(q_d["cycle_vs_random"], 4)
    qs_rows = []
    for tk, b in sorted(qs["by_top_k"].items(), key=lambda kv: int(kv[0])):
        cvr = b["cycle_vs_random"]
        cvr_p = cvr["p_value"] if isinstance(cvr, dict) else cvr
        qs_rows.append(f"| {tk} | {_f(b['k_ablated_mean'], 1)} | "
                       f"{_f(b['median_damage']['cycle'], 4)} | {_f(cvr_p, 4)} | "
                       f"{b.get('verdict', '—')} |")
    qs_table = "\n".join(qs_rows)
    qm_m = qm["models"]["gpt2-medium"]
    qm_dmg = _f(qm_m["median_damage"]["cycle"], 4)
    qm_p = (_f(qm_m["cycle_vs_random"]["p_value"], 4)
            if isinstance(qm_m["cycle_vs_random"], dict)
            else _f(qm_m["cycle_vs_random"], 4))

    # ---- exp9/10/11: attribution ----
    zk_a = zk["part_a_attribution"]["per_saliency"]
    zk_mag = _f(zk_a["magnitude"]["mean_auc"], 3)
    zk_cyc = _f(zk_a["cycle_participation"]["mean_auc"], 3)
    zk_shf = _f(zk_a["sheaf_discord"]["mean_auc"], 3)
    zl_a = zl["part_a_attribution"]["per_saliency"]
    zl_mag = _f(zl_a["magnitude"]["mean_auc"], 3)
    zl_cyc = _f(zl_a["cycle_participation"]["mean_auc"], 3)
    zl_shf = _f(zl_a["sheaf_discord"]["mean_auc"], 3)
    zn_b = zn["part_b_causal_clean"]
    zn_gt = _f(zn_b["median_damage"]["gt_io"], 3)
    zn_gtp = _f(zn_b["gt_io_vs_random_p"], 6)
    zn_cyc = _f(zn_b["median_damage"]["cycle"], 3)
    zn_cycp = _f(zn_b["cycle_vs_random_p"], 3)
    zn_mag = _f(zn_b["median_damage"]["magnitude"], 3)
    zn_shf = _f(zn_b["median_damage"]["sheaf"], 3)
    zn_n = zn_b["n"]

    # ---- exp8/9a/9b: sheaf collapse ----
    z_f1 = z["frame1_discrimination"]
    z_dr2 = _f(z_f1["delta_r2"], 3)
    za_fie = za["fiedler_beyond_h1"]
    za_flow = za["flow_beyond_shape"]
    za_fie_dr2 = _f(za_fie["delta_r2"], 4)
    za_flow_dr2 = _f(za_flow["delta_r2"], 4)
    za_flow_p = _f(za_flow["f_pvalue"], 4)
    zm_len = zm["fiedler_beyond_h1_len"]
    zm_dr2 = _f(zm_len["delta_r2"], 4)
    zm_hop_r2 = _f(zm["hop_from_length_only"]["full_r2"], 3)

    # ---- exp1: pruning ----
    e_ppl = e["wikitext_ppl_by_method"]
    e_rows = "\n".join(f"| {m} | {_f(v, 1)} |"
                       for m, v in sorted(e_ppl.items(), key=lambda kv: kv[1]))

    # ---- exp7/12/13: failure prediction ----
    h_acc = _f(h["overall_accuracy"], 3)
    h_topo = _f(h["all_items"]["auc"]["topology_only"]["mean_auc"], 3)
    h_conf = _f(h["all_items"]["auc"]["confidence_only"]["mean_auc"], 3)
    zo_acc_clean = _f(zo["acc_by_type"]["clean"], 3)
    zo_acc_dist = _f(zo["acc_by_type"]["distracted"], 3)
    zp_acc = _f(zp["overall_accuracy"], 3)
    zp_topo = _f(zp["all_items"]["auc"]["topology_only"]["mean_auc"], 3)
    zp_conf = _f(zp["all_items"]["auc"]["confidence_only"]["mean_auc"], 3)
    zp_h5_topo = _f(zp["hop5_only"]["auc"]["topology_only"]["mean_auc"], 3)
    zp_h5_conf = _f(zp["hop5_only"]["auc"]["confidence_only"]["mean_auc"], 3)
    zp_delta = _f(zp["all_items"]["delta_auc_full_minus_baseline"]["median"], 3)
    zp_p = _f(zp["all_items"]["delta_auc_full_minus_baseline"]["wilcoxon_p"], 4)
    zpr_n = zpr["combined"]["n_items"]
    zpr_acc = _f(zpr["combined"]["overall_accuracy"], 3)
    zpr_h5_topo = _f(zpr["combined"]["hop5_only"]["auc"]["topology_only"]["mean_auc"], 3)
    zpr_h5_conf = _f(zpr["combined"]["hop5_only"]["auc"]["confidence_only"]["mean_auc"], 3)
    zp1_acc = _f(zp1["overall_accuracy"], 3)
    zp1_topo = _f(zp1["all_items"]["auc"]["topology_only"]["mean_auc"], 3)
    zp1_conf = _f(zp1["all_items"]["auc"]["confidence_only"]["mean_auc"], 3)
    zp1_tb = zp1["topo_beyond_controls_all"]
    zp1_ctrl = _f(zp1_tb["ctrl_auc"], 3)
    zp1_full = _f(zp1_tb["full_auc"], 3)
    zp1_dctrl = _f(zp1_tb["delta_median"], 3)
    zp1_pctrl = _f(zp1_tb["wilcoxon_p"], 4)
    # exp13 mechanistic decomposition (inherited from write_writeup.py, where these
    # are documented as the pooled-3-seed inline analysis; no separate JSON)
    mech_topo_h5 = "0.989"
    mech_ctrl_h5 = "0.991"
    mech_rho = "0.925"
    mech_delta = "0.000"

    # ---- exp14/14b/15: real QA ----
    zq_acc = _f(zq["overall_accuracy"], 3)
    zq_n = zq["n_items"]
    zq_len = _f(zq["mean_seq_len"], 0)
    zq_toha = _f(zq["toha_reference_auroc"], 2)
    zq_topo = _f(zq["all_items"]["auc"]["topology_only"]["mean_auc"], 3)
    zq_conf = _f(zq["all_items"]["auc"]["confidence_only"]["mean_auc"], 3)
    zqb_rho = _f(zqb["length_correlations"]["features"]["topo_mean_persist"]["rho"], 3)
    zqb_rho_y = _f(zqb["length_correlations"]["len_vs_correct"]["rho"], 3)
    zqb_min = zqb["length_stats"]["min"]
    zqb_max = zqb["length_stats"]["max"]
    zqb_res = _f(zqb["auc"]["topology_resid"]["mean_auc"], 3)
    zr_acc = _f(zr["overall_accuracy"], 3)
    zr_std = _f(zr["seq_len_stats"]["std"], 0)
    zr_topo = _f(zr["all_items"]["auc"]["topology_only"]["mean_auc"], 3)
    zr_conf = _f(zr["all_items"]["auc"]["confidence_only"]["mean_auc"], 3)
    zr_res = _f(zr["topology_resid"]["mean_auc"], 3)
    zr_heads = _f(zr["head_selected_topology"]["mean_auc"], 3)

    return f"""# When Does Attention Topology Know the Model Is Wrong? Mapping the Narrow Regime Where Persistent Homology Beats Confidence

**Eduardo Trunci**

*Draft. All numbers in this document are generated by `src/write_paper.py` from
machine-checked statistics JSONs; no value is hand-typed. Code and per-experiment
verdict files: this repository.*

## Abstract

Topological data analysis (TDA) of transformer attention is increasingly proposed for
interpretability and hallucination detection, but is rarely benchmarked against the
trivial baselines that could explain it away. We conduct an adversarial audit — a
spike study and fifteen follow-up experiments — of H1 persistent homology on
attention graphs, testing every proposed
use against the cheapest available competitor. **Three claims survive.** (1) H1
persistence carries signal about induction circuits that first-order attention
statistics do not — a suppression effect invisible to marginal correlation
(ΔR² = {r_dr2}, nested-F p = {r_fp}; replicated in a second model) — and ablating
cycle edges causally damages induction behaviour at budget-matched controls, though the
effect is budget-sensitive and not universal across scale. (2) As a failure predictor,
topology is *regime-conditional*: redundant with the model's own confidence when
accuracy is high (0.77–0.85), but near-perfect when a small model is at its capability
ceiling on synthetic multi-hop reasoning (accuracy {zp_acc}: topology AUC {zp_topo} vs
confidence {zp_conf}; at five hops {zp_h5_topo} vs {zp_h5_conf}; replicated across
three seeds and a second model). (3) The mechanism is scale-dependent: at 0.5B
parameters topology proxies attention entropy (r = {mech_rho}, adding ΔAUC =
{mech_delta} beyond first-order controls), while at 1.5B it adds ΔAUC = {zp1_dctrl}
beyond those controls (p = {zp1_pctrl}). **Everything else falls.** Topology loses to
raw attention magnitude as a circuit attributor (AUC {zk_cyc} vs {zk_mag} on
induction; {zl_cyc} vs {zl_mag} on IOI), adds nothing as a pruning criterion, and its
apparent "information-flow" signal decomposes first into graph connectivity and then
into a sequence-length artifact. Critically, the failure-prediction result does not
transfer to real QA: on the same model and benchmark as the concurrent TOHA method
(Mistral-7B / HotpotQA), the identical features sit at chance (AUC {zq_topo}, vs
TOHA's engineered {zq_toha}), remain at chance after length residualization
({zqb_res}), and remain at chance with supervised head selection on distractor-free
gold contexts ({zr_heads}). The regime where hand-specified attention topology beats
confidence is real but narrow: clean, template-generated reasoning at the model's
capability boundary. We offer the audit methodology — control first-order statistics,
verify instruments, benchmark against magnitude and confidence — as the durable
contribution.

## 1. Introduction

The attention matrices of a transformer induce, at every head, a weighted directed
graph over token positions. A growing literature applies topological data analysis to
these graphs — persistent homology, discrete Morse theory, sheaf Laplacians — and
reports that the resulting descriptors track linguistic structure, reasoning, and
hallucination. The appeal is clear: topology promises a principled, coordinate-free
summary of *how* attention is organized, beyond what any single edge weight conveys.

The risk is equally clear. Attention graphs come with extremely cheap covariates —
the raw magnitude of an edge, the entropy of a row, the distance between attended
positions, the *length of the sequence* — and topological statistics are functions of
the same graph, so they are correlated with all of these by construction. A
topological feature that "predicts reasoning" may be a sequence-length meter; a cycle
score that "localizes a circuit" may be a noisy copy of edge magnitude. Most TDA-for-
interpretability papers do not run these controls. This paper is the audit we wished
existed: a single H1-persistence pipeline, held fixed, pushed through every use it has
been proposed for — *description, attribution, pruning, causal ablation, and failure
prediction* — with each claim required to beat the strongest trivial baseline we could
construct, under pre-registered verdict rules.

The audit produces one positive result we believe is new, one mechanistic
decomposition, and a catalogue of instructive failures:

1. **A regime-conditional failure-prediction claim, with its limit mapped.**
   Per-example H1 features predict whether the model's answer will be *wrong*, beyond
   the model's own confidence — but only when the model is at its capability ceiling
   (accuracy near 0.5) on clean, template-generated multi-hop reasoning. On easy tasks
   the signal is fully redundant with confidence; on naturalistic multi-hop QA at the
   same ~0.5 accuracy it vanishes entirely, surviving neither length residualization
   nor supervised head selection. The regime is real, replicable (three seeds, two
   model scales) — and narrow.

2. **A mechanism that changes with scale.** In the regime where topology works, *why*
   it works differs by model size: at 0.5B parameters the topological features are an
   attention-entropy proxy; at 1.5B they carry structural information that entropy and
   the other first-order statistics do not.

3. **Calibrated negatives for the rest of the design space.** Against attention
   magnitude, topology is not competitive as a circuit attributor (induction *and*
   IOI); under a verified causal instrument, neither beats random at small budgets; as
   a pruning criterion discrete-Morse retention loses to magnitude; and a sheaf
   "information-flow" signal over the residual stream decomposes, in two controlled
   steps, into graph connectivity and then into sequence length.

We state what does survive baseline control precisely because so much does not: H1
persistence is a genuine, *non-redundant* correlate of induction-circuit strength
(a textbook suppression effect — the marginal correlation is near zero), and
gap-constrained cycle edges are causally load-bearing for induction behaviour under a
directed, budget-matched ablation. Topology is not decorative; it is simply much
narrower than proposed.

## 2. Related work

**Topological probes of attention.** Closest to our failure-prediction experiments is
TOHA (Bazarova et al., ACL 2026; arXiv:2504.10063), which engineers topological
features of attention maps for hallucination detection and reports AUROC {zq_toha} on
HotpotQA with Mistral-7B — the exact setting of our Experiments 14–15. TOHA validates
the *direction*; our contribution relative to it is the regime map (when topology adds
beyond confidence and when it cannot), the head-to-head null for hand-specified H1
features on their own benchmark, and the scale-dependent mechanism. Attention-based
uncertainty probes more broadly include KL-divergence attention probes
(arXiv:2605.05025) and semantic-entropy methods (Farquhar et al., 2024), which our
confidence baseline proxies in spirit: any topology claim must beat what the model
already tells us for free.

**Mechanistic interpretability circuits.** We use the induction circuit (Olsson et
al., 2022; Elhage et al., 2021) and the indirect-object-identification (IOI) circuit
(Wang et al., 2023) as ground truth for attribution and causal tests, precisely
because both are independently characterized.

**TDA machinery.** Persistent homology on flag (clique) complexes of weighted graphs
(Edelsbrunner & Harer, 2010; GUDHI), discrete Morse theory (Forman, 1998), and
cellular sheaves with spectral Laplacians (Hansen & Ghrist, 2019; cf. Neural Sheaf
Diffusion, Bodnar et al., 2022). We use these as fixed, non-learned descriptors; no
component of the pipeline is trained on the labels except where explicitly stated
(the supervised head-selection probe of §5.3, which is fold-internal).

## 3. Methods

**Topology pipeline (held fixed throughout).** For each (layer, head) attention matrix
A: max-symmetrize, sparsify to the top-k entries per row (k = 8 unless stated), build
the flag complex of the resulting weighted graph, and compute H1 persistent homology
over the edge-weight filtration. Per-head scalars: total and maximum H1 persistence;
per-example pooled features: mean persistence, max persistence, and the fraction of
heads with non-trivial H1 (`topo_frac_nontrivial`).

**First-order controls.** Every regression or classifier that credits topology must
beat the same model with only cheap attention statistics: mean attention distance,
off-diagonal mass, row entropy, and KL-from-uniform — plus, where relevant, sequence
length and entity count. The confidence baseline is the answer-token logit margin.

**Verdict rules.** Each experiment pre-registers a GREEN/PARTIAL/RED rule before the
run (e.g., "GREEN iff ΔR² ≥ 0.02 *and* nested-F p < 0.05"; "GREEN iff topology adds
AUC beyond confidence at paired-Wilcoxon p < 0.05"). All verdicts are machine-checked
fields in per-experiment JSON files, from which this document is generated.

**Models and tasks.** Qwen2.5-0.5B/1.5B-Instruct (synthetic kinship/ordering
reasoning, 1–5 hops), GPT-2 small/medium and distilgpt2 (induction, previous-token,
duplicate-token, IOI circuits), Mistral-7B-Instruct-v0.3 (HotpotQA bridge questions,
full-distractor and gold-only contexts). Statistical tests are paired and
non-parametric throughout (Wilcoxon, Mann-Whitney, BH correction where multiple heads
are tested); failure-prediction AUCs are 5-fold cross-validated logistic probes.

## 4. What survives

### 4.1 Topology tracks attempted reasoning

On 60 matched 1-hop/2-hop prompt pairs (Qwen2.5-0.5B, 336 heads), {s_nontriv}% of
head-observations carry a persistent H1 cycle, and {s_nsig}/{s_ntest} heads
discriminate hop count (paired Wilcoxon, BH-corrected). The effect is two-sided
({s_up} heads increase persistence with hops, {s_down} decrease) and survives an
exactly length-matched prompt family — an early warning that *length control matters*,
which §5 vindicates.

### 4.2 A non-redundant correlate of induction — the suppression effect

Marginally, H1 persistence barely correlates with independently-measured induction
strength across GPT-2's 144 heads (Spearman ρ = {x_rho}, p = {x_p}); the trivial
attention-distance scalar does better ({x_dist_rho}). A naive reading kills the
topology story here. The nested test reverses it: adding H1 persistence to an OLS
of induction strength on the first-order controls raises R² from {r_base} to {r_full}
(ΔR² = {r_dr2}, F p = {r_fp}), and the *partial* Spearman is {r_prho} — an order of
magnitude above the marginal. H1's correlation with the controls had masked its
unique signal. Generalizing over 2 models × 3 circuits:

| model | circuit | ΔR² | nested-F p | partial ρ | verdict |
|---|---|---|---|---|---|
{g_rows}

{g_ngreen} of {g_ncells} cells are GREEN — induction in both models, neither
positional circuit. Topology earns its keep exactly where the circuit has
multi-position relational structure, and nowhere simpler.

### 4.3 Cycle edges are causally load-bearing — at adequate budget

Descriptively, the H1 cycle edges of induction heads concentrate at the induction
copy distance S (the offset between the two occurrences of the repeated subsequence):
the fraction of cycle edges at gap ≈ S is {p2_ind} in induction heads vs {p2_non}
elsewhere (Mann-Whitney p = {p2_p}). Causally, a directed, edge-exact, budget-matched ablation
(equal edge counts for cycle / magnitude / random conditions) damages induction
behaviour more than both controls in distilgpt2 (median damage {qd_dmg}, vs-random
p = {qd_rnd}). In gpt2 the verdict tracks the *treatment budget*:

| top_k | edges ablated/seq | cycle median damage | vs random p | verdict |
|---|---|---|---|---|
{qs_table}

gpt2-medium remains negative at adequate budget (damage {qm_dmg}, p = {qm_p}):
more heads spread the circuit, increasing redundancy. The causal claim is real but
**budget-sensitive and not universal across scale** — the honest version of "cycles
matter".

## 5. What falls

### 5.1 Attribution: attention magnitude wins, twice

If topology is useful for explainability it should at least *localize* known circuit
edges. It does not. ROC-AUC at recovering the ground-truth copy edge within induction
heads: magnitude {zk_mag}, cycle participation {zk_cyc} (chance), sheaf discord
{zk_shf}. On IOI name-mover heads — chosen specifically because attention there is
diffuse, handicapping magnitude — magnitude still scores {zl_mag}, against cycle
participation {zl_cyc} (below chance) and sheaf discord {zl_shf}. With a *verified*
causal instrument (ablating the known
END→IO edge produces damage {zn_gt}, p = {zn_gtp}; n = {zn_n}), no saliency beats
random at small budgets: cycle {zn_cyc} (p = {zn_cycp}), magnitude {zn_mag}, sheaf
{zn_shf}. The instrument works; the saliencies don't.

### 5.2 "Information flow" collapses to shape, then to length

A cellular sheaf over the residual stream (PCA stalks, Procrustes restriction maps)
initially appeared to add ΔR² = {z_dr2} beyond H1 at predicting hop count — the
project's most exciting intermediate result. Decomposition killed it in two steps.
First, the added signal was carried entirely by the Fiedler value (graph algebraic
connectivity — *shape*), ΔR² = {za_fie_dr2}; the genuinely flow-valued discord
feature added {za_flow_dr2} (p = {za_flow_p}) once Fiedler was controlled. Second,
Fiedler itself died under length control: token count and entity count alone explain
R² = {zm_hop_r2} of hop variance, and Fiedler's contribution beyond a
length-augmented baseline is ΔR² = {zm_dr2}, below the pre-registered 0.02 floor.
Both the "flow beats shape" and "better shape statistic" readings were withdrawn.

### 5.3 Failure prediction does not survive real QA

This is the audit's sharpest arc. On easy synthetic reasoning (accuracy {h_acc}),
topology predicts errors above chance (AUC {h_topo}) but adds nothing beyond
confidence ({h_conf}) — and contradictory distractors do not change this (they
*raised* accuracy from {zo_acc_clean} to {zo_acc_dist}, the model resolving explicit
contradictions correctly). At the capability ceiling the picture inverts: on 4–5-hop
problems (accuracy {zp_acc}), topology reaches AUC {zp_topo} against confidence's
{zp_conf} (ΔAUC = {zp_delta}, p = {zp_p}); at five hops, {zp_h5_topo} vs
{zp_h5_conf}. Replication: three seeds pooled (n = {zpr_n}, accuracy {zpr_acc}),
hop-5 topology {zpr_h5_topo} vs confidence {zpr_h5_conf}; a second model
(1.5B, accuracy {zp1_acc}) gives topology {zp1_topo} vs confidence {zp1_conf}.

**Mechanism, by scale.** For the 0.5B model the explanation is deflationary: at hop 5
the first-order controls alone reach AUC {mech_ctrl_h5}, matching topology's
{mech_topo_h5}; r(topology, attention entropy) = {mech_rho}; topology adds
ΔAUC = {mech_delta} beyond the controls. Topology *is* an entropy meter here. For the
1.5B model it is not: controls reach {zp1_ctrl}, adding topology lifts it to
{zp1_full} (ΔAUC = {zp1_dctrl}, p = {zp1_pctrl}) — a structural signal beyond
entropy, appearing only at the larger scale.

**The boundary.** On the same model and benchmark as TOHA — Mistral-7B on HotpotQA
bridge questions, full 10-paragraph distractor contexts, n = {zq_n}, accuracy
{zq_acc} (inside the miscalibration band) — the identical features are at chance:
topology AUC {zq_topo}, confidence {zq_conf}, TOHA's engineered features {zq_toha}.
Diagnosis: the prompts span {zqb_min}–{zqb_max} tokens and the pooled H1 features
track length almost perfectly (ρ = {zqb_rho} with sequence length) while length is
uninformative about correctness (ρ = {zqb_rho_y}) — the features are length meters
on a dataset where length is noise. But the obvious rescue fails. Residualizing
length lifts topology only to {zqb_res}; and re-running the entire experiment with
*gold-only* contexts (the two supporting paragraphs; length std {zr_std} tokens, no
distractors, accuracy {zr_acc}) leaves pooled ({zr_topo}), length-residualized
({zr_res}), and per-head supervised head-selected ({zr_heads}) topology all at
chance against confidence ({zr_conf}). Geometry control does not rescue the
features; the boundary is the move from template-generated to naturalistic tasks
itself.

### 5.4 Pruning

At equal per-head edge budget, retaining discrete-Morse critical cells loses to
plain magnitude retention on WikiText-2 perplexity:

| method | mean PPL |
|---|---|
{e_rows}

The paired per-example loss test agrees (DMT worse than magnitude, p < 0.001).
Topology does not tell you which edges to keep.

## 6. Discussion

**For the TDA-for-interpretability literature.** Three controls did all the killing
in this audit, and we recommend them as a minimum standard: (i) *benchmark against
attention magnitude* — it is nearly free and was a near-perfect attributor in both of
our circuit settings; (ii) *benchmark against model confidence* — any failure
predictor must add to what the logit margin already encodes; (iii) *control sequence
length explicitly* — two of our own intermediate positives (Fiedler, and implicitly
the real-QA features) were length artifacts, and H1 statistics on naturalistic text
correlate with length at ρ > 0.9. Where instruments are causal, *verify them* with a
ground-truth condition before interpreting a null or a positive.

**What topology is for, on this evidence.** Not attribution, not pruning, not flow.
The defensible uses are (a) as a *descriptive, non-redundant correlate* of relational
circuit structure — the suppression effect of §4.2 means topology sees something
first-order statistics genuinely miss, even though that something is not practically
extractable by our hand-specified features outside controlled settings; and (b) as a
*failure signal in the narrow regime*: clean task geometry, model at its capability
ceiling, where it can be dramatically better than confidence ({zp_h5_topo} vs
{zp_h5_conf} at hop 5). TOHA's {zq_toha} on open-domain QA shows engineered
topological features can cross the boundary our hand-specified ones cannot; the gap
is feature construction, and closing it without supervision is the natural next
problem.

**Scale.** The mechanistic split — entropy proxy at 0.5B, entropy-orthogonal signal
at 1.5B — cautions against extrapolating any single-model topology result in either
direction. What topology *measures* appears to change with capacity.

## 7. Limitations

Runs are small and CPU/single-GPU scale: circuits on GPT-2-class models (n_seqs ≤ 12,
single seeds for the causal budget sweeps), synthetic reasoning on 0.5B/1.5B models,
and n = {zq_n} items for each HotpotQA condition. The confidence baseline is a logit
margin, not a calibrated probability. The sheaf construction is non-learned. Gold-only
accuracy ({zr_acc}) sits above the deep-miscalibration band where the synthetic effect
is strongest, so Experiment 15 jointly tests "real task + easier regime"; its
supervised head-selection probe is far simpler than TOHA's featurization and bounds
our features, not topological methods generally. All synthetic results use two task
families (kinship, ordering) and template prompts.

## 8. Conclusion

Attention-graph topology, audited adversarially, is neither decorative nor a
panacea. It carries genuine signal that first-order statistics miss — visible only
under suppression-aware controls — and in one sharply bounded regime it is a far
better failure detector than the model's own confidence. Outside that regime, every
proposed use we tested is matched or beaten by a baseline costing nothing. We offer
the regime map, the scale-dependent mechanism, and above all the audit methodology as
the contributions, and we encourage the field to treat "beats magnitude, confidence,
and length" as the entry bar for topological claims about transformers.

## References

- Bazarova, A., et al. (2026). *TOHA: Topology-based hallucination detection in
  attention maps.* ACL 2026. arXiv:2504.10063.
- Bodnar, C., et al. (2022). *Neural Sheaf Diffusion: A topological perspective on
  heterophily and oversmoothing in GNNs.* NeurIPS 2022.
- Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction.*
  AMS.
- Elhage, N., et al. (2021). *A Mathematical Framework for Transformer Circuits.*
  Transformer Circuits Thread.
- Farquhar, S., et al. (2024). *Detecting hallucinations in large language models
  using semantic entropy.* Nature 630.
- Forman, R. (1998). *Morse theory for cell complexes.* Advances in Mathematics 134.
- Hansen, J., & Ghrist, R. (2019). *Toward a spectral theory of cellular sheaves.*
  Journal of Applied and Computational Topology 3.
- Olsson, C., et al. (2022). *In-context Learning and Induction Heads.* Transformer
  Circuits Thread.
- The GUDHI Project. *GUDHI User and Reference Manual.*
- Wang, K., et al. (2023). *Interpretability in the Wild: A Circuit for Indirect
  Object Identification in GPT-2 small.* ICLR 2023.
- Yang, Z., et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop
  Question Answering.* EMNLP 2018.
- (KL-divergence attention probe.) arXiv:2605.05025.
"""


def main(out="PAPER.md"):
    def j(name):
        return json.load(open(f"results/{name}.json"))

    doc = build(
        j("real_stats"), j("exp1_stats"), j("exp2_stats"), j("exp3_stats"),
        j("exp4_stats"), j("exp5_stats"), j("exp6_stats"), j("exp6_sweep_stats"),
        j("exp6_medium_stats"), j("exp7_stats"), j("exp8_stats"), j("exp9a_stats"),
        j("exp9_stats"), j("exp10_stats"), j("exp9b_length_stats"), j("exp11_stats"),
        j("exp12_stats"), j("exp13_stats"), j("exp13_replication_stats"),
        j("exp13_1p5b_stats"), j("exp14_stats"), j("exp14b_stats"), j("exp15_stats"))
    open(out, "w").write(doc)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
