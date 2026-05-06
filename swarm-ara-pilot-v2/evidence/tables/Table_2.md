# Table 2: Scenario Configurations

**Source**: Table 2, Section 5 "Experimental Setup" (PDF lines 406–419)
**Caption**: "Scenario configurations. Agent types: H=honest, O=opportunistic, D=deceptive, A=adversarial, C=cautious, AA=adaptive adversarial, CR=cautious reciprocator, TD=threshold dancer. Governance: tax rate, circuit breaker (CB) threshold, audit probability, reputation decay rate."
**Extraction type**: raw_table

| Scenario | Agents | Epochs | Steps | Tax | CB | Audit | Decay |
|---|---|---|---|---|---|---|---|
| Baseline | 3H+1O+1D | 20 | 15 | 0.00 | — | — | 1.00 |
| Strict Governance | 3H+2O+1D+1A | 20 | 15 | 0.10 | 0.35 | 0.25 | 0.85 |
| Adaptive Governance | 4H+2O+1A | 20 | 15 | 0.05 | 0.70 | 0.10 | 0.95 |
| Adversarial Red Team | 4H+2A+2AA | 20 | 15 | 0.05 | 0.60 | 0.15 | 0.95 |
| Misalignment Sweep | 4H+2O+2A+1D+1C | 20 | 15 | 0.05 | 0.70 | 0.10 | 0.95 |
| Threshold Dancer | 2H+3CR+3TD | 20 | 15 | 0.05 | 0.80 | 0.10 | 0.95 |
| Collusion Detection | 3H+2O+3A | 20 | 15 | 0.05 | 0.60 | — | 0.95 |
