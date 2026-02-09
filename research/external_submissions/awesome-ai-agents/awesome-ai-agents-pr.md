# PR: Add SWARM to awesome-ai-agents

**Target repo:** https://github.com/e2b-dev/awesome-ai-agents
**PR Title:** Add SWARM: safety simulation framework for multi-agent AI interactions

## PR Body

### What is SWARM?

[SWARM](https://github.com/swarm-ai-safety/swarm) (System-Wide Assessment of Risk in Multi-agent systems) is a research framework for studying emergent risks in multi-agent AI systems. It uses soft probabilistic labels to reveal how catastrophic failures can emerge from agent interactions — even when no individual agent is dangerous.

### Why it fits this list

SWARM belongs in the **Multi-agent** category. It provides:
- Simulation of multi-agent interactions with measurable safety metrics (toxicity, adverse selection, incoherence)
- Testable governance mechanisms (circuit breakers, reputation decay, collusion detection)
- Multiple agent types (Honest, Opportunistic, Deceptive, Adversarial, LLM-based)
- YAML-defined scenarios with full reproducibility

### Placement

Added alphabetically between **Sweep** and **Taxy AI** in the open-source projects section.

### Entry added

```markdown
## [SWARM](https://github.com/swarm-ai-safety/swarm)
Safety simulation framework for multi-agent AI interactions

<details>

![Image](https://raw.githubusercontent.com/swarm-ai-safety/swarm/main/docs/swarm-logo.png)

### Category
Multi-agent, Research, Safety, Build your own

### Description
- SWARM (System-Wide Assessment of Risk in Multi-agent systems) is a research framework for studying emergent risks in multi-agent AI systems
- Uses soft probabilistic labels instead of binary classifications to measure whether agent interactions are beneficial
- Reveals how catastrophic failures can emerge from interactions between many sub-AGI agents, even when none are individually dangerous
- Provides measurable safety metrics: toxicity rate, quality gap (adverse selection), and incoherence across replays
- Includes testable governance mechanisms: transaction taxes, reputation decay, circuit breakers, random audits, staking, and collusion detection
- Supports multiple agent types: Honest, Opportunistic, Deceptive, Adversarial, and LLM-based agents
- YAML-defined scenarios with parameter sweeps and full reproducibility from seed + scenario config

### Links
- [GitHub](https://github.com/swarm-ai-safety/swarm)
- [Website](https://swarm-ai.org)
- [Documentation](https://docs.swarm-ai.org)

</details>
```
