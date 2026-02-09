## Add SWARM: distributional safety simulation framework
Hi! I'd like to add [SWARM](https://github.com/swarm-ai-safety/swarm) to the Environment section of this awesome list.

### What is SWARM?
**SWARM (System-Wide Assessment of Risk in Multi-agent systems)** is an open-source Python framework for studying emergent risks in multi-agent AI systems using probabilistic (soft) labels rather than binary classifications.

### Why it fits this list
SWARM provides a simulation environment directly relevant to multi-agent learning research:
- **Distributional safety**: Studies how catastrophic failures emerge from interactions of many sub-AGI agents, even when none are individually dangerous.
- **Soft probabilistic labels**: Interactions carry `p = P(v = +1)` instead of binary good/bad, enabling richer analysis of adverse selection, toxicity, and quality gaps.
- **Governance mechanisms**: Built-in levers (taxes, reputation decay, circuit breakers, audits, staking, collusion detection) for studying multi-agent governance.
- **Agent archetypes**: Honest, opportunistic, deceptive, adversarial, and LLM-backed agents.
- **Reproducible experiments**: YAML scenario definitions, parameter sweeps, and replay capabilities.

### Key metrics studied
| Metric | What it measures |
|--------|-----------------|
| Toxicity rate | Expected harm among accepted interactions |
| Quality gap | Adverse selection indicator (negative = system preferentially accepts low-quality) |
| Conditional loss | Selection effect on payoffs |
| Incoherence | Variance-to-error ratio across replays |

### Links
- Repository: https://github.com/swarm-ai-safety/swarm
- License: MIT
- Install: `pip install swarm-safety`

This framework connects well with many papers in this list, particularly those studying emergent complexity, multi-agent competition, and social influence in multi-agent systems.
