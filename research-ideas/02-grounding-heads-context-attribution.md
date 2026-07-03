# 2 — Grounding heads: per-head answer→context attachment as a faithfulness meter

## Question

Exp 16 computed, for every one of Mistral-7B's 1024 heads, how cheaply the
generated answer's tokens attach to the *prompt* (MTop-Div). Failure
prediction from it is dead — but that was the wrong target. The natural
target for this feature is **context attribution**: did the model's answer
come *from the provided context* (grounded) or from parametric memory /
confabulation (unfaithful)? That is a different label than correctness — and
it is the question RAG verification, context-faithfulness, and the
retrieval-heads literature all care about.

## Why our data already speaks to it

We hold a controlled contrast nobody else has: the *same* 200 questions
answered under distractor contexts and gold-only contexts, with per-head
attachment scores persisted for both (`exp16_{distractor,gold}_features.parquet`).
Questions the existing data can answer for $0:

1. Are there heads whose answer→context attachment is stable across the two
   geometries (candidate "grounding heads"), and are they the same heads the
   retrieval-heads literature finds (induction-like, mid-depth)?
2. When the model answers the same question differently with vs without
   distractors (the divergent subset), do specific heads' attachment scores
   flip? That subset is exactly where context won or lost against memory.
3. Does per-head attachment to the *gold* region vs the *distractor* region
   (recomputable from answer→prompt rows if we persist column spans — small
   re-run) predict which source the answer copied?

## Design

Phase A ($0, offline): analyses 1–2 on existing parquets, plus a
"same-answer-under-both-geometries" grouping as a weak grounding label.
Phase B (one A100 session): re-run exp16-style capture persisting per-head
attachment split by context span (gold para 1, gold para 2, each distractor),
plus a NoContext condition (question only) whose answer agreement gives a
clean parametric-memory label — the model "knew it anyway" detector.
Phase C: probe — can span-resolved attachment predict (i) NoContext
agreement, (ii) LLM-judged faithfulness, better than confidence and better
than pooled attention mass to context (the magnitude baseline our audit
discipline demands)?

## Why it matters (attention/context SOTA)

Context faithfulness is arguably *the* applied long-context problem (RAG
verification, citation attribution, "did the model read the document").
Retrieval-head work shows a sparse set of heads carries copy-from-context
behavior; a per-head, budget-controlled attachment meter with our audit
discipline (magnitude + confidence baselines, pre-registered verdicts) would
be a genuinely useful contribution whether the result is positive or a
calibrated negative. And it converts exp16's RED infrastructure into a
second paper's foundation.

## Cost

Phase A: $0, ~a day of analysis. Phase B+C: ~1 A100-hour ≈ **$2–3**.
