# Show HN Draft

---

**Title:** Show HN: SWARM -- Open-source framework for studying when multi-agent AI systems fail

**Body:**

SWARM (System-Wide Assessment of Risk in Multi-agent systems) is a simulation framework for studying emergent failures in multi-agent AI ecosystems.

The core insight: AGI-level risks don't require AGI-level agents. Catastrophic outcomes can emerge from many sub-AGI agents interacting, even when none are individually dangerous.

We borrowed from financial market theory (adverse selection, market microstructure) to build probabilistic metrics that capture dynamics invisible to binary safe/unsafe labels. Every interaction gets a calibrated probability p = P(beneficial), enabling continuous metrics like toxicity (expected harm among accepted interactions) and quality gap (adverse selection indicator).

Key findings across 11+ scenarios and 2000+ simulation runs:

- Sharp phase transition: governance works at 37.5% adversarial agents, fails at 50%. The transition is abrupt, not gradual.
- Transaction tax is the strongest single governance lever — explains 32% of welfare variance in our factorial sweeps (p=0.004). Circuit breakers, despite being the most commonly proposed mechanism, show zero effect (d=-0.02).
- Collusion detection (pattern-based, not individual) is the threshold capability that prevents ecosystem collapse.
- The Purity Paradox: mixed agent populations outperform pure honest ones on aggregate welfare — but this reverses when you properly price externalities.
- 20+ governance mechanisms across 5 families (friction, detection, reputation, circuit breakers, structural), all configurable via YAML.

The newest addition: a GPU kernel marketplace where agents generate actual CUDA code. A static regex analyzer extracts code features (bounds checks, shared memory, hardcoded shapes) that feed into proxy quality signals — so adversarial coding strategies produce measurably different governance outcomes.

Technical details:

- `pip install swarm-safety`
- 2800+ tests, MIT license
- 51 YAML scenario configs
- 7 framework bridges (Concordia, OpenClaw, GasTown, AgentXiv, ClawXiv, Claude Code, Prime Intellect)
- Built-in parameter sweep and statistical analysis (`examples/sweep_stats.py`)
- SQLite database of all runs for reproducibility
- No API keys needed for the core simulation (LLM agents optional)

GitHub: https://github.com/swarm-ai-safety/swarm
Quickstart notebook: 5 min, no API keys

---

**HN posting notes:**

- Post on a weekday morning (US Eastern, ~9-11am) for best visibility
- "Show HN" prefix is correct since this is a project you built
- Monitor comments for the first 2 hours and respond promptly
- Common HN objections to anticipate:
  - "Simulation != reality" -- acknowledge explicitly. These are toy models of real dynamics. The value is in the phase transitions and governance tradeoffs, not in predicting specific real-world outcomes.
  - "What's new vs existing multi-agent frameworks?" -- soft (probabilistic) labels instead of binary, financial market theory basis, 20+ governance levers with factorial sweep infrastructure.
  - "How does this scale?" -- largest tested is 10 agents. Scale experiments are future work. Be honest.
  - "Is this just a toy?" -- point to 2800+ tests, 51 scenarios, 5 papers, 7 framework bridges, built-in statistical analysis.
  - "Why not just use [CrewAI/AutoGen/etc]?" -- those are agent orchestration frameworks. SWARM is a safety measurement framework. It studies what goes wrong, not how to build agents.
  - "Circuit breakers don't work?" -- they fire too late (after damage is done) and the freeze period removes agents that might have been learning. Our 40-run factorial sweep showed d=-0.02 with p=0.88. Null effect.
