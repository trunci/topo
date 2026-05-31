# Experiment 9 Findings

Topological attribution + causal validation: does per-edge topology localize the induction copy edge beyond attention magnitude, and is it causal? Generated mechanically from `results/exp9_stats.json`; do not edit by hand.

- alpha = 0.05
- **Headline verdict: PARTIAL**

## Part A - Attribution (ROC-AUC at recovering ground-truth copy edges)

**Verdict: PARTIAL** (n = 240 head x seq)

| saliency | mean AUC | mean p@k | p vs chance | p vs magnitude |
|---|---|---|---|---|
| magnitude | 0.976 | 0.288 | 0.0000 | n/a |
| cycle_participation | 0.501 | 0.006 | 1.0000 | 1.0000 |
| sheaf_discord | 0.581 | 0.031 | 0.0000 | 1.0000 |

## Part B - Causal validation (damage = loss(cond) - loss(unmasked))

**Verdict: GREEN** (n = 24 seq)

| condition | median damage |
|---|---|
| cycle | -0.007 |
| sheaf | 0.001 |
| magnitude | 0.021 |
| random | -0.002 |

- cycle: damage>0 = False, vs random p = 0.7721, vs magnitude p = 0.9553
- sheaf: damage>0 = True, vs random p = 0.0002, vs magnitude p = 0.3216

## Interpretation

Part A PARTIAL: attention magnitude is a near-perfect attributor of the copy edge (AUC 0.976); topology does NOT beat it. Of the topological saliencies, sheaf_discord beat chance but stayed well below magnitude; cycle_participation was at chance. Part B GREEN (narrow): sheaf-flagged edges damage induction above random (p=0.0002), but the effect (median 0.001) is ~19x weaker than magnitude (0.021) and does not beat it (p=0.32); cycle edges do not beat random at all.
