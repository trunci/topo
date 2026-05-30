# Discrete Morse Theory for Attention Sparsification and Mechanistic Interpretability in LLMs

## Motivation

Transformer attention is dense (O(n²)) but empirically redundant — existing sparse-attention methods (sliding window, BigBird, Native Sparse Attention) work because most attention mass is structurally unnecessary. Current sparsification methods use learned gating or fixed patterns; none use topological invariants of the attention graph itself as the sparsification criterion.

In parallel, mechanistic interpretability has produced strong local results (induction heads, IOI circuits, SAE features) but lacks a unifying mathematical framework for *where* and *why* circuits form. Circuits are typically discovered by ablation rather than predicted from first principles.

This proposal connects these two threads via Discrete Morse Theory (Forman, 1998), which provides a principled way to identify the topologically essential structure of a simplicial complex by constructing a discrete gradient vector field. Critical cells of the resulting Morse complex correspond to homology generators.

## Core Hypothesis

The flag complex of a transformer's attention graph has non-trivial topology — specifically, H₁ generators ("cycles") corresponding to multi-hop reasoning paths, coreference loops, and syntactic agreement structures. Discrete Morse Theory applied to this complex identifies the minimal subset of attention edges whose preservation is sufficient to maintain the model's computational topology.

This yields three concrete, testable claims:

1. **Sparsification:** Attention pruning based on DMT critical cells preserves multi-hop reasoning performance better than magnitude or learned-pattern baselines at equivalent sparsity.
2. **Mechanistic correspondence:** H₁ generators of the attention complex correspond to interpretable computational circuits (induction, coreference, agreement) identified independently by ablation-based MI methods.
3. **Failure prediction:** Topological features (β₁, persistence of H₁ classes) computed at inference time are predictive of reasoning failures on multi-hop benchmarks.

## Experimental Plan

Frozen open-weight model in the 7B–14B range (Llama 3.1 8B or Qwen 2.5 7B). Single H100 sufficient for the full plan.

**Experiment 1 — Pruning quality.** Compute discrete Morse matching on flag complex of attention graphs per layer/head. Use critical cells to define sparsified attention pattern. Compare against magnitude pruning, random pruning, and sliding window at matched sparsity, on WikiText perplexity, MMLU, MuSiQue, HotpotQA, and 2WikiMultihopQA.

**Experiment 2 — Cycle correlates of reasoning.** Compute β₁ per layer/head on multi-hop vs. single-hop benchmark subsets. Test whether β₁ separates by hop-count and whether the differential localizes to layers known to implement reasoning circuits.

**Experiment 3 — Failure prediction.** Train probes on per-example topological features to predict reasoning correctness on held-out multi-hop tasks.

**Tooling:** `transformer_lens` for attention extraction, `gudhi` for discrete Morse and persistence computation. Compute budget approximately 3–5 GPU-days across all experiments.

## Expected Outcomes

**Realistic (60%):** DMT pruning matches or modestly beats magnitude pruning on multi-hop benchmarks specifically; β₁ correlates with hop-count with moderate effect size; failure-prediction probe beats baseline by 3–7 AUC points. A solid contribution to the sparse-attention and MI literatures.

**Strong (15%):** DMT pruning meaningfully outperforms learned sparse attention at long context; H₁ generators map cleanly to named MI circuits; failure detector reaches deployment-relevant AUC. Notable conference paper.

**Best case (5%):** Empirical wins plus a theoretical result showing equivalence between DMT critical cells and circuits identified by path patching, establishing a coordinate-free mathematical framework for mechanistic interpretability.

**Null (20%):** Topology is informative diagnostically but doesn't beat existing sparsification or yield clean circuit correspondence. Still publishable as a careful negative result with diagnostic tooling of independent value.

## Fallback Directions

- Drop pruning, keep topological analysis as a runtime MI diagnostic.
- Lift from flag complex to cellular sheaf on the residual stream (stalks from OV-circuit subspaces); sheaf cohomology gives a richer obstruction theory.
- Move to training-time regularization via differentiable persistent homology rather than inference-time pruning.
- Apply to domain-specific reasoning (medical dialogue, legal argument graphs) where the topology of the task is more explicit than in general language.

## Why This Is Worth Doing

Three properties make this project unusually tractable:

1. **The mathematics is mature.** Forman's discrete Morse theory, persistent homology, and sheaf cohomology are well-developed. The novelty is in the application, not the foundations.
2. **The computational cost is modest.** Greedy/random Morse matchings on attention-graph flag complexes are tractable at LLM scale; this is not a method that requires frontier compute to evaluate.
3. **The interpretability community is ready for it.** SAE and circuit-analysis work has established the descriptive vocabulary; this would provide predictive structure on top of it.

The strongest version of the contribution is not "a better sparse attention method" (a crowded space) but "a topological framework that predicts where mechanistic-interpretability circuits form, with sparsification as a downstream application." Either result alone is a meaningful paper; together they would be a notable contribution.

## Practical Asks

A 3-month focused effort with single-GPU access (Modal or Lambda-class) and standard interpretability tooling is sufficient to complete Experiments 1–3 and produce a publication-ready result. The work is structured to yield partial results at each milestone, so commitment is staged rather than monolithic.

---

Want me to adapt the tone for a specific lab (more theoretical for an academic group, more applied for an industry research team), or produce a one-page version?