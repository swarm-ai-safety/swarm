# Awesome List Submission Drafts

Drafts for submitting SWARM to curated awesome lists. Each section includes the entry text, target section, and submission command.

---

## 1. Giskard-AI/awesome-ai-safety

**Status:** Ready to submit
**Section:** General ML Testing
**Format:** `* [Title](URL) (Author, Year) \`#Tag\``

**Entry:**

```
* [SWARM: Distributional Safety for Multi-Agent AI Systems](https://github.com/swarm-ai-safety/swarm) (Savitt, 2026) `#Safety` `#MultiAgent` `#Robustness`
```

**PR title:** Add SWARM: multi-agent AI safety simulation framework

**PR body:**

```
## New Resource

SWARM (System-Wide Assessment of Risk in Multi-agent systems) is an open-source simulation framework for studying emergent failures in multi-agent AI ecosystems. It applies financial market theory (adverse selection, market microstructure) to build probabilistic safety metrics that capture dynamics invisible to binary safe/unsafe labels.

Key features: 20+ governance mechanisms, 51 scenario configs, 2800+ tests, factorial sweep infrastructure with built-in statistical analysis. Findings include sharp phase transitions at ~40-50% adversarial agent fraction and that transaction tax is a stronger governance lever than circuit breakers (eta2=0.324 vs d=-0.02).

GitHub: https://github.com/swarm-ai-safety/swarm
```

**Command:**

```bash
/submit_to_list Giskard-AI/awesome-ai-safety "General ML Testing"
```

---

## 2. kaushikb11/awesome-llm-agents

**Status:** Ready to submit (stretch fit — SWARM is a safety testing framework, not an agent-building framework)
**Section:** Frameworks
**Format:** Detailed entry with stars/features

**Entry:**

```
- [SWARM](https://github.com/swarm-ai-safety/swarm) - Simulation framework
  for studying emergent safety failures in multi-agent AI systems

  Python · MIT

  - Probabilistic (soft-label) safety metrics from financial market theory
  - 20+ governance mechanisms (taxes, audits, collusion detection, circuit breakers)
  - 51 scenario configs with parameter sweep infrastructure
  - Bridges to Concordia, GasTown, AgentXiv, ClawXiv, Claude Code, Prime Intellect
  - No API keys needed for core simulation
```

**PR title:** Add SWARM: multi-agent safety simulation framework

**PR body:**

```
## New Framework

SWARM is a simulation framework for studying when multi-agent AI systems fail. Unlike agent orchestration frameworks, it focuses on safety measurement — using probabilistic metrics from financial market theory to detect adverse selection, ecosystem collapse, and coordinated exploitation in multi-agent environments.

2800+ tests, 51 scenario configs, 7 framework bridges, MIT license.

GitHub: https://github.com/swarm-ai-safety/swarm
```

**Command:**

```bash
/submit_to_list kaushikb11/awesome-llm-agents "Frameworks"
```

---

## 3. kyegomez/awesome-multi-agent-papers

**Status:** Already submitted (merged via PR #30)

---

## Notes

- Giskard-AI/awesome-ai-safety is the strongest fit (safety-focused list, SWARM is a safety framework)
- kaushikb11/awesome-llm-agents is a stretch (agent-building list, SWARM is safety-testing) but has 6000+ stars and high visibility
- Both lists accept PRs from the community
- Run `/submit_to_list` with the commands above to fork, branch, and open PRs via `gh` CLI
