## Add SWARM to Multi-Agent Simulation Projects

Hi! I'd like to suggest adding [SWARM](https://github.com/swarm-ai-safety/swarm) to the **Multi-Agent Simulation Projects** section.

### What is SWARM?

**SWARM** (System-Wide Assessment of Risk in Multi-agent systems) is a research framework for studying emergent risks and distributional safety in multi-agent AI systems. It focuses on how harmful dynamics can emerge from the *interaction* of many LLM-powered agents — even when no single agent is individually dangerous.

### Key Features

- **LLM Agent Support** — First-class integration with Anthropic, OpenAI, and Ollama; configurable personas (Honest, Strategic, Adversarial); prompt audit logging and cost tracking.
- **Probabilistic Safety Metrics** — Soft (probabilistic) labels instead of binary classification: toxicity rate, quality gap (adverse selection detection), conditional loss, and incoherence.
- **Configurable Governance Levers** — Transaction taxes, reputation decay, circuit breakers, random audits, staking, collusion detection, sybil detection, and more.
- **Multi-Agent Simulation** — Heterogeneous agent populations (honest, opportunistic, deceptive, adversarial, LLM-backed) interacting in a marketplace environment.
- **Red-Teaming Framework** — Systematically test governance mechanisms against adaptive adversaries.
- **Reproducible Research** — YAML scenario definitions, seed-based reproducibility, append-only event logs, CSV/JSON exports.

### Why it fits this list

SWARM directly addresses **multi-agent cooperation and competition** among LLM-powered agents, with a specific focus on safety and governance — a critical gap in the current ecosystem. It complements existing entries like AgentVerse and ChatArena by focusing on *emergent risk measurement* rather than general collaboration or game environments.

### Entry

```markdown
* ![SWARM Stars](https://img.shields.io/github/stars/swarm-ai-safety/swarm) [SWARM](https://github.com/swarm-ai-safety/swarm) - A research framework for studying emergent risks and governance in multi-agent LLM systems, featuring probabilistic safety metrics, configurable governance levers, and first-class LLM agent support.
```

Thank you for maintaining this excellent resource!
