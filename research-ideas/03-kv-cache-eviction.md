# 3 — KV-cache eviction: a budget-matched audit on long context

## Question

Long-context inference SOTA (H2O, SnapKV, TOVA, …) evicts KV-cache entries by
*attention magnitude* heuristics. Our Exp 1 found magnitude beats topology for
attention-edge pruning at equal budget — on short WikiText with GPT-2. Does
that verdict hold at the setting people actually care about: 7B+ models,
4k–32k contexts, generation quality (not perplexity), and per-head budgets?
And is there *any* structural signal (position-in-cycle, attachment role,
head type) that adds on top of magnitude at fixed memory budget?

## Design

- Model: Mistral-7B (fits our harness); contexts: HotpotQA-long concatenations
  and needle-style synthetic tasks (both already buildable from data_hotpotqa).
- Conditions at equal per-head KV budget: (a) magnitude (SnapKV-style),
  (b) recency window, (c) magnitude + protected structural positions (cycle
  participants / induction-gap positions from our circuits work), (d) random.
- Metrics: answer accuracy on gold questions, degradation curves vs budget,
  per-head-type sensitivity (do retrieval/induction-like heads need bigger
  budgets — connecting to head-specific budget allocation, the current SOTA
  edge).
- Audit discipline as always: pre-registered verdicts, magnitude as the
  baseline to beat, budget-matched exactly.

## Honest prior

Magnitude likely wins again head-to-head; the publishable contribution is
(i) the controlled degradation map — *which heads and which context regions
break first under eviction* — and (ii) whether head-type-aware budget
allocation (informed by our per-head feature machinery) beats uniform
budgets, which is where current methods differentiate.

## Why it matters (attention/context SOTA)

KV eviction is the most directly SOTA-relevant thing this repo's tooling can
touch: it is *the* attention+context efficiency problem, benchmarks are
standard, and a rigorous negative ("structure adds nothing over magnitude at
matched budget, here is the audit") or positive ("head-aware budgets buy X%
context at equal memory") both land in an active conversation.

## Cost

Biggest of the pool: eviction requires generation-loop integration (real
engineering, ~2–3 days) plus sweep compute — **~$10–20 of A100 time**.
