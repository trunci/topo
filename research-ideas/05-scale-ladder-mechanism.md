# 5 — Scale ladder: where does the entropy-orthogonal signal go?

## Question

The audit's mechanistic split is a two-point trend: at 0.5B, topology is an
attention-entropy proxy (r = 0.925, ΔAUC 0.000 beyond controls); at 1.5B it
carries structure entropy cannot see (ΔAUC 0.216, CI [0.147, 0.285]). Two
points define nothing. Qwen2.5 offers 3B / 7B / 14B with identical tasks:
does the entropy-orthogonal component grow, saturate, or vanish — and does
the capability-ceiling band (where topology beats confidence) move up in
hops as capacity grows?

## Design

- Same synthetic kinship/ordering generator, hop count re-tuned per model so
  accuracy lands in the 0.45–0.60 band (the regime's precondition); the
  hop-at-ceiling *itself* becomes a capacity measurement.
- Same features, probes, controls; per-model pre-registered verdicts on
  (i) topology beyond confidence, (ii) topology beyond first-order controls,
  (iii) r(persistence, entropy).
- Deliverable: three curves vs parameter count — regime band location,
  ΔAUC-beyond-controls, entropy-proxy correlation.

## Why it matters (attention/context SOTA)

Attention entropy dynamics at scale (attention sinks, entropy collapse,
long-context attention concentration) is an active mechanistic area. A clean
measurement of "what per-head attention organization knows about impending
failure, as a function of scale, with entropy controlled" plugs into that
conversation and de-risks every single-model attention-probe result in the
literature — the audit paper's §6 already flags this caution; this experiment
would quantify it.

## Cost

3B/7B fit L4; 14B wants the A100 (bf16 ~28GB). Reuses run_exp13 harness with
a model-name sweep. **~$10** total; the main cost is hop re-tuning runs.
