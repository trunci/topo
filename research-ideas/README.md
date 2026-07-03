# Research ideas — follow-ups to the topology audit

Ranked pool of next experiments. Each file: motivation, what our existing
evidence says, concrete design, cost, and what SOTA conversation it feeds.
Per the project direction, ideas are weighted toward **attention** and
**context** — the two live research areas where this repo's tooling
(per-head attention capture, budget-matched ablation harness, pre-registered
audit discipline) can contribute beyond the topology question itself.

| # | idea | one line | GPU cost | SOTA area |
|---|---|---|---|---|
| 1 | [TOHA protocol diagnosis](01-toha-protocol-diagnosis.md) | why do we get chance where TOHA reports 0.71 — labels, items, decoding, or budget? | ~$5 | attention-based hallucination detection |
| 2 | [Grounding heads / context attribution](02-grounding-heads-context-attribution.md) | per-head answer→context attachment as a faithfulness meter; find the heads that know whether context was used | first pass $0 (data on disk) | context faithfulness, RAG verification, retrieval heads |
| 3 | [KV-cache eviction audit](03-kv-cache-eviction.md) | budget-matched audit of structure-aware vs magnitude KV eviction on long context | ~$10–20 | long-context inference efficiency |
| 4 | [Boundary interpolation](04-boundary-interpolation.md) | template → naturalistic gradient: which property of real context kills attention signals | ~$5 | context robustness / evaluation |
| 5 | [Scale ladder for the mechanism](05-scale-ladder-mechanism.md) | 3B/7B/14B: does the entropy-orthogonal signal grow, and does the regime band move? | ~$10 | attention dynamics at scale |
| 6 | [Directed, generation-time topology](06-directed-generation-time-topology.md) | path homology on directed attention; length-invariant by construction | ~$5 | attention structure methods |

Recommended order: **1 → 2** (1 completes the paper's story; 2 is the highest
upside per dollar and opens the context-faithfulness direction with data
already on disk).
