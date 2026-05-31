# Experiment 10 Findings

Topological attribution + causal validation on **IOI** (name-mover heads, GPT-2), where attention is more diffuse so the magnitude baseline is handicapped. Generated mechanically from `results/exp10_stats.json`; do not edit by hand.

- task = IOI, model = `gpt2`, alpha = 0.05
- **Headline verdict: PARTIAL**

## Part A - attribution (recover the END->IO name-mover edge)

**Verdict: PARTIAL** (n = 384 head x prompt)

| saliency | mean AUC | mean p@1 | p vs chance | p vs magnitude |
|---|---|---|---|---|
| magnitude | 0.963 | 0.609 | 0.0000 | n/a |
| cycle_participation | 0.458 | 0.000 | 1.0000 | 1.0000 |
| sheaf_discord | 0.606 | 0.000 | 0.0000 | 1.0000 |

## Part B - causal (damage = IO-S margin drop from ablation)

**Verdict: RED** (n = 48 prompts)

| condition | median damage |
|---|---|
| cycle | -0.181 |
| sheaf | -0.378 |
| magnitude | -0.852 |
| random | 0.019 |

- cycle: damage>0 = False, vs random p = 0.9467, vs magnitude p = 0.0015
- sheaf: damage>0 = False, vs random p = 0.9914, vs magnitude p = 0.0040

## Interpretation

Part A PARTIAL: magnitude (AUC 0.963) is not beaten; topological sheaf_discord beat chance but not magnitude. Part B RED: ablating flagged edges does not reduce the IO margin beyond random. CAUTION: every condition's median damage is <=0 (ablation tends to RAISE the IO-S margin, magnitude most of all). The name-mover edge feeds both IO and S logits, so removing it within these heads can net-help the margin -- this readout cannot cleanly score 'necessity' here; the Part B RED reflects a compromised instrument, not strong evidence against topology.
